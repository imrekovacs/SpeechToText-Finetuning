"""
Whisper Tiny Fine-tuning CLI
=============================
Commands:
  transcribe <audio_file>   - Transcribe an audio file using the model
  train <config_yaml>       - Fine-tune Whisper Tiny on your dataset
  evaluate                  - Evaluate the fine-tuned model
"""

import typer
from rich.console import Console

app = typer.Typer(
    name="whisper-cli",
    help="CLI for fine-tuning and using OpenAI Whisper Tiny model",
    add_completion=False,
)
console = Console()


@app.command()
def transcribe(
    audio_path: str = typer.Argument(..., help="Path to the audio file to transcribe"),
    model_path: str = typer.Option(
        None,
        "--model", "-m",
        help="Path to fine-tuned model directory (uses base whisper-tiny if omitted)",
    ),
):
    """Transcribe an audio file using Whisper Tiny (or a fine-tuned checkpoint)."""
    from src.transcribe import transcribe_audio

    console.print(f"[bold blue]Transcribing:[/bold blue] {audio_path}")
    text = transcribe_audio(audio_path, model_path=model_path)
    console.print(f"\n[bold green]Transcription:[/bold green]\n{text}")


@app.command()
def train(
    config_path: str = typer.Argument("config.yaml", help="Path to training config YAML"),
    strategy: str = typer.Option(
        None,
        "--strategy", "-s",
        help="Override fine-tuning strategy: full, lora, qlora",
    ),
    freeze_encoder: bool = typer.Option(
        None,
        "--freeze-encoder/--no-freeze-encoder",
        help="Override encoder freezing setting from config",
    ),
    gradient_checkpointing: bool = typer.Option(
        None,
        "--grad-ckpt/--no-grad-ckpt",
        help="Override gradient checkpointing setting from config",
    ),
):
    """Fine-tune Whisper Tiny on the dataset specified in the config."""
    from src.train import run_training, load_config
    import yaml

    # Apply CLI overrides to the config before training
    if strategy or freeze_encoder is not None or gradient_checkpointing is not None:
        cfg = load_config(config_path)
        ft = cfg.setdefault("finetuning", {})
        if strategy:
            ft["strategy"] = strategy
        if freeze_encoder is not None:
            ft["freeze_encoder"] = freeze_encoder
        if gradient_checkpointing is not None:
            ft["gradient_checkpointing"] = gradient_checkpointing
        # Write patched config to a temp file
        import tempfile, os
        tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False, encoding="utf-8"
        )
        yaml.dump(cfg, tmp, default_flow_style=False)
        tmp.close()
        config_path = tmp.name

    console.print(f"[bold blue]Starting training with config:[/bold blue] {config_path}")
    run_training(config_path)
    console.print("[bold green]Training complete![/bold green]")


@app.command()
def evaluate(
    config_path: str = typer.Option("config.yaml", "--config", "-c", help="Config YAML"),
    model_path: str = typer.Option(
        None,
        "--model", "-m",
        help="Path to fine-tuned model (auto-detected from config if omitted)",
    ),
):
    """Evaluate the fine-tuned model and report WER."""
    from src.evaluate import run_evaluation

    console.print("[bold blue]Running evaluation...[/bold blue]")
    results = run_evaluation(config_path, model_path=model_path)
    console.print(f"\n[bold green]Word Error Rate (WER):[/bold green] {results['wer']:.2%}")


if __name__ == "__main__":
    app()
