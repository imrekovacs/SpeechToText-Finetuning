"""
Fine-tune OpenAI Whisper Tiny on a custom dataset.

Supported fine-tuning strategies
--------------------------------
- **full**   – update every model parameter (highest quality, most VRAM)
- **lora**   – low-rank adapters via PEFT (~0.6 % trainable params)
- **qlora**  – 4-bit quantised base model + LoRA (lowest VRAM)

Additional techniques
---------------------
- **freeze_encoder** – freeze the audio encoder, train only the decoder
- **gradient_checkpointing** – trade compute for VRAM savings (~40 %)

Dataset layout expected in `data/`:
  data/
    audio1.wav
    audio2.wav
    ...
    metadata.csv        # columns: file_name, transcription

If metadata.csv doesn't exist, the script will auto-generate one
by looking for .txt files that share the same stem as the audio files.
"""

import os
import csv
import yaml
import torch
import torchaudio
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Any

from datasets import Dataset, DatasetDict
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    WhisperFeatureExtractor,
    WhisperTokenizer,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    BitsAndBytesConfig,
)
import evaluate


# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------

def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Dataset preparation
# ---------------------------------------------------------------------------

def _ensure_metadata(data_dir: str, audio_format: str) -> str:
    """Create metadata.csv from paired audio + .txt files if not present."""
    meta_path = os.path.join(data_dir, "metadata.csv")
    if os.path.exists(meta_path):
        return meta_path

    rows = []
    for audio_file in sorted(Path(data_dir).glob(f"*.{audio_format}")):
        txt_file = audio_file.with_suffix(".txt")
        if txt_file.exists():
            transcription = txt_file.read_text(encoding="utf-8").strip()
            rows.append({"file_name": audio_file.name, "transcription": transcription})

    if not rows:
        raise FileNotFoundError(
            f"No paired audio (.{audio_format}) + .txt files found in {data_dir}. "
            "Please provide a metadata.csv or matching .txt transcription files."
        )

    with open(meta_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["file_name", "transcription"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Auto-generated metadata.csv with {len(rows)} entries.")
    return meta_path


def load_dataset_from_dir(cfg: dict) -> DatasetDict:
    """Load audio + transcription data into a HuggingFace DatasetDict."""
    data_dir = cfg["data"]["train_dir"]
    audio_fmt = cfg["data"].get("audio_format", "wav")
    test_split = cfg["data"].get("test_split", 0.2)
    sample_rate = cfg["data"].get("sample_rate", 16000)

    meta_path = _ensure_metadata(data_dir, audio_fmt)

    # Read metadata
    entries = []
    with open(meta_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            audio_path = os.path.join(data_dir, row["file_name"])
            if os.path.exists(audio_path):
                entries.append({
                    "audio_path": audio_path,
                    "transcription": row["transcription"],
                })

    if not entries:
        raise ValueError("No valid audio entries found in metadata.csv")

    # Load audio waveforms
    records = []
    for entry in entries:
        waveform, sr = torchaudio.load(entry["audio_path"])
        if sr != sample_rate:
            waveform = torchaudio.transforms.Resample(sr, sample_rate)(waveform)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        records.append({
            "audio": waveform.squeeze().numpy(),
            "transcription": entry["transcription"],
        })

    ds = Dataset.from_dict({
        "audio": [r["audio"] for r in records],
        "transcription": [r["transcription"] for r in records],
    })

    # Split
    if len(ds) < 2:
        # Too few samples to split; use same data for both train and test
        return DatasetDict({"train": ds, "test": ds})

    split = ds.train_test_split(test_size=test_split, seed=42)
    return DatasetDict({"train": split["train"], "test": split["test"]})


# ---------------------------------------------------------------------------
# Data collator
# ---------------------------------------------------------------------------

@dataclass
class DataCollatorSpeechSeq2Seq:
    processor: Any
    decoder_start_token_id: int

    def __call__(self, features: list[dict]) -> dict:
        # Extract audio arrays and tokenize transcriptions
        audio_arrays = [f["audio"] for f in features]
        transcriptions = [f["transcription"] for f in features]

        # Process audio
        batch = self.processor.feature_extractor(
            audio_arrays, sampling_rate=16000, return_tensors="pt"
        )

        # Tokenize labels
        labels = self.processor.tokenizer(
            transcriptions, return_tensors="pt", padding=True, truncation=True
        )
        label_ids = labels["input_ids"]

        # Replace padding token id with -100 so it's ignored in loss
        label_ids = label_ids.masked_fill(
            label_ids == self.processor.tokenizer.pad_token_id, -100
        )

        batch["labels"] = label_ids
        return batch


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------

def make_compute_metrics(processor):
    wer_metric = evaluate.load("wer")

    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids

        # Replace -100 with pad token
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

        pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        wer = wer_metric.compute(predictions=pred_str, references=label_str)
        return {"wer": wer}

    return compute_metrics


# ---------------------------------------------------------------------------
# Training entrypoint
# ---------------------------------------------------------------------------

def _count_parameters(model) -> tuple[int, int]:
    """Return (trainable, total) parameter counts."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def _apply_finetuning_strategy(model, cfg: dict):
    """
    Apply the selected fine-tuning strategy to *model* **in-place** and
    return the (possibly wrapped) model.

    Strategies
    ----------
    full  – no changes, all parameters remain trainable.
    lora  – attach LoRA adapters via PEFT.
    qlora – model must already be loaded in 4-bit; attach LoRA adapters.
    """
    ft = cfg.get("finetuning", {})
    strategy = ft.get("strategy", "full").lower()

    # ── Freeze encoder ────────────────────────────────────────────────
    if ft.get("freeze_encoder", False):
        for param in model.model.encoder.parameters():
            param.requires_grad = False
        print("Encoder frozen — only decoder parameters will be trained.")

    # ── Gradient checkpointing ────────────────────────────────────────
    if ft.get("gradient_checkpointing", False):
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        print("Gradient checkpointing enabled.")

    # ── LoRA / QLoRA ──────────────────────────────────────────────────
    if strategy in ("lora", "qlora"):
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType

        lora_cfg = ft.get("lora", {})
        r = lora_cfg.get("r", 16)
        alpha = lora_cfg.get("lora_alpha", 32)
        dropout = lora_cfg.get("lora_dropout", 0.05)
        target = lora_cfg.get("target_modules", ["q_proj", "v_proj"])
        bias = lora_cfg.get("bias", "none")

        # Map string task type to PEFT enum
        task_str = lora_cfg.get("task_type", "SEQ_2_SEQ_LM")
        task_type = getattr(TaskType, task_str, TaskType.SEQ_2_SEQ_LM)

        if strategy == "qlora":
            # prepare 4-bit quantised model for training
            model = prepare_model_for_kbit_training(model)

        peft_config = LoraConfig(
            r=r,
            lora_alpha=alpha,
            lora_dropout=dropout,
            target_modules=target,
            bias=bias,
            task_type=task_type,
        )

        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
    else:
        trainable, total = _count_parameters(model)
        pct = trainable / total * 100 if total else 0
        print(f"Strategy: full — trainable params: {trainable:,} / {total:,} ({pct:.1f}%)")

    return model


def _get_quantization_config(cfg: dict):
    """Return a BitsAndBytesConfig for QLoRA, or None."""
    ft = cfg.get("finetuning", {})
    strategy = ft.get("strategy", "full").lower()
    if strategy != "qlora":
        return None
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )


def run_training(config_path: str):
    cfg = load_config(config_path)

    model_name = cfg["model"]["name"]
    language = cfg["model"].get("language", "en")
    task = cfg["model"].get("task", "transcribe")
    tcfg = cfg["training"]
    ft = cfg.get("finetuning", {})
    strategy = ft.get("strategy", "full").lower()

    print(f"Loading model: {model_name}")
    print(f"Fine-tuning strategy: {strategy}")

    processor = WhisperProcessor.from_pretrained(
        model_name, language=language, task=task
    )

    # Load model — optionally quantised for QLoRA
    quant_config = _get_quantization_config(cfg)
    load_kwargs: dict[str, Any] = {}
    if quant_config is not None:
        load_kwargs["quantization_config"] = quant_config
        load_kwargs["device_map"] = "auto"

    model = WhisperForConditionalGeneration.from_pretrained(
        model_name, **load_kwargs
    )

    # Force decoder settings
    model.generation_config.language = language
    model.generation_config.task = task
    model.generation_config.forced_decoder_ids = None

    # Apply fine-tuning strategy (LoRA, freezing, grad ckpt)
    model = _apply_finetuning_strategy(model, cfg)

    print("Loading dataset...")
    dataset = load_dataset_from_dir(cfg)
    print(f"  Train samples: {len(dataset['train'])}")
    print(f"  Test samples:  {len(dataset['test'])}")

    # Data collator
    data_collator = DataCollatorSpeechSeq2Seq(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,
    )

    # Adjust learning rate for LoRA (higher works better with adapters)
    lr = float(tcfg.get("learning_rate", 1e-5))
    if strategy in ("lora", "qlora") and lr <= 1e-5:
        lr = 1e-3  # LoRA benefits from a higher lr
        print(f"  Auto-raised learning rate to {lr} for {strategy.upper()}.")

    # Training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=tcfg["output_dir"],
        num_train_epochs=tcfg.get("num_train_epochs", 3),
        per_device_train_batch_size=tcfg.get("per_device_train_batch_size", 4),
        per_device_eval_batch_size=tcfg.get("per_device_eval_batch_size", 4),
        gradient_accumulation_steps=tcfg.get("gradient_accumulation_steps", 2),
        learning_rate=lr,
        warmup_steps=tcfg.get("warmup_steps", 50),
        logging_steps=tcfg.get("logging_steps", 10),
        eval_steps=tcfg.get("eval_steps", 50),
        save_steps=tcfg.get("save_steps", 100),
        save_total_limit=tcfg.get("save_total_limit", 2),
        fp16=tcfg.get("fp16", torch.cuda.is_available()),
        dataloader_num_workers=tcfg.get("dataloader_num_workers", 0),
        push_to_hub=tcfg.get("push_to_hub", False),
        report_to=tcfg.get("report_to", "none"),
        eval_strategy="steps",
        predict_with_generate=True,
        generation_max_length=225,
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        remove_unused_columns=False,  # Keep audio/transcription for data collator
    )

    # Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        data_collator=data_collator,
        compute_metrics=make_compute_metrics(processor),
        processing_class=processor.feature_extractor,
    )

    print("Starting training...")
    trainer.train()

    # Save final model + processor
    final_dir = os.path.join(tcfg["output_dir"], "final")

    if strategy in ("lora", "qlora"):
        # Save only the adapter weights (very small, ~2-5 MB)
        model.save_pretrained(final_dir)
        processor.save_pretrained(final_dir)
        print(f"Adapter weights saved to {final_dir}")

        # Also save a merged copy for easy inference
        merged_dir = os.path.join(tcfg["output_dir"], "merged")
        print("Merging adapter weights into base model...")
        merged_model = model.merge_and_unload()
        merged_model.save_pretrained(merged_dir)
        processor.save_pretrained(merged_dir)
        print(f"Merged model saved to {merged_dir}")
    else:
        model.save_pretrained(final_dir)
        processor.save_pretrained(final_dir)
        print(f"Model saved to {final_dir}")
