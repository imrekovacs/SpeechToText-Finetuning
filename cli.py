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
):
    """Fine-tune Whisper Tiny on the dataset specified in the config."""
    from src.train import run_training

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
