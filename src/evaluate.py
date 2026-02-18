"""
Evaluate a fine-tuned Whisper model on the test split.
Reports Word Error Rate (WER).

Supports both full fine-tuned models and LoRA/QLoRA adapter models.
If the model path contains `adapter_config.json`, it will automatically
load the adapter weights on top of the base model.
"""

import os
import yaml
import torch
import numpy as np
from rich.console import Console
from rich.table import Table
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import evaluate as hf_evaluate

from src.train import load_config, load_dataset_from_dir

console = Console()


def _load_model_auto(model_path: str, cfg: dict | None = None):
    """
    Load a Whisper model from *model_path*.

    If *model_path* contains ``adapter_config.json`` (i.e. a PEFT adapter
    checkpoint), the base model is loaded first and the adapter is applied.
    Otherwise the path is treated as a regular HuggingFace model directory.
    """
    adapter_cfg = os.path.join(model_path, "adapter_config.json")

    if os.path.isfile(adapter_cfg):
        # LoRA / QLoRA adapter checkpoint
        from peft import PeftModel
        import json

        with open(adapter_cfg, "r") as f:
            acfg = json.load(f)
        base_name = acfg.get("base_model_name_or_path", "openai/whisper-tiny")
        console.print(f"Loading base model: [bold]{base_name}[/bold]")
        base_model = WhisperForConditionalGeneration.from_pretrained(base_name)
        console.print(f"Applying adapter from: [bold]{model_path}[/bold]")
        model = PeftModel.from_pretrained(base_model, model_path)
        model = model.merge_and_unload()  # merge for fast inference
        processor = WhisperProcessor.from_pretrained(model_path)
    else:
        processor = WhisperProcessor.from_pretrained(model_path)
        model = WhisperForConditionalGeneration.from_pretrained(model_path)

    return processor, model


def run_evaluation(config_path: str, model_path: str | None = None) -> dict:
    """
    Evaluate a Whisper model on the test split defined in config.

    Args:
        config_path: Path to the training config YAML.
        model_path: Path to the fine-tuned model. If None, auto-detect from config output_dir.

    Returns:
        Dictionary with evaluation metrics (e.g. {"wer": 0.12}).
    """
    cfg = load_config(config_path)

    # Resolve model path — prefer merged copy, then adapter, then base
    if model_path is None:
        merged_path = os.path.join(cfg["training"]["output_dir"], "merged")
        final_path = os.path.join(cfg["training"]["output_dir"], "final")
        if os.path.exists(merged_path):
            model_path = merged_path
        elif os.path.exists(final_path):
            model_path = final_path
        else:
            model_path = cfg["model"]["name"]
            console.print(
                f"[yellow]No fine-tuned model found. Evaluating base model: {model_path}[/yellow]"
            )

    console.print(f"Loading model from: [bold]{model_path}[/bold]")
    processor, model = _load_model_auto(model_path, cfg)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()

    # Load test dataset
    dataset = load_dataset_from_dir(cfg)
    test_ds = dataset["test"]

    wer_metric = hf_evaluate.load("wer")

    predictions = []
    references = []

    console.print(f"Evaluating on {len(test_ds)} samples...")

    for i, sample in enumerate(test_ds):
        audio_array = np.array(sample["audio"], dtype=np.float32)
        reference = sample["transcription"]

        input_features = processor(
            audio_array, sampling_rate=16000, return_tensors="pt"
        ).input_features.to(device)

        with torch.no_grad():
            predicted_ids = model.generate(input_features)

        pred_text = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0].strip()

        predictions.append(pred_text)
        references.append(reference)

    # Compute WER
    wer = wer_metric.compute(predictions=predictions, references=references)

    # Display results table
    table = Table(title="Evaluation Results")
    table.add_column("Sample #", style="cyan", justify="right")
    table.add_column("Reference", style="green")
    table.add_column("Prediction", style="yellow")

    for i, (ref, pred) in enumerate(zip(references, predictions)):
        table.add_row(str(i + 1), ref, pred)

    console.print(table)

    return {"wer": wer, "predictions": predictions, "references": references}
