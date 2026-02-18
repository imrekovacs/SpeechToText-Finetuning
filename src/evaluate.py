"""
Evaluate a fine-tuned Whisper model on the test split.
Reports Word Error Rate (WER).
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

    # Resolve model path
    if model_path is None:
        model_path = os.path.join(cfg["training"]["output_dir"], "final")
        if not os.path.exists(model_path):
            # Fallback to base model
            model_path = cfg["model"]["name"]
            console.print(
                f"[yellow]No fine-tuned model found. Evaluating base model: {model_path}[/yellow]"
            )

    console.print(f"Loading model from: [bold]{model_path}[/bold]")
    processor = WhisperProcessor.from_pretrained(model_path)
    model = WhisperForConditionalGeneration.from_pretrained(model_path)
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
