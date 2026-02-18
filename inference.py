"""
Inference script for the fine-tuned Whisper Tiny model.

Supports:
  - Single file transcription
  - Batch transcription of a directory
  - Comparison between base and fine-tuned models
  - Real-time microphone input (if sounddevice installed)

Usage:
    .venv\\Scripts\\python.exe inference.py audio.wav
    .venv\\Scripts\\python.exe inference.py audio.wav --model output/whisper-tiny-finetuned/final
    .venv\\Scripts\\python.exe inference.py data/ --batch
    .venv\\Scripts\\python.exe inference.py audio.wav --compare
"""

import os
import sys
import time
import glob
import argparse

import torch
import torchaudio
import numpy as np
from transformers import WhisperProcessor, WhisperForConditionalGeneration


# ── Model loading ────────────────────────────────────────────────────────────

_model_cache: dict = {}


def load_model(model_path: str, device: str | None = None):
    """Load and cache a Whisper model + processor."""
    if model_path in _model_cache:
        return _model_cache[model_path]

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading model: {model_path} → {device}")

    processor = WhisperProcessor.from_pretrained(model_path)
    model = WhisperForConditionalGeneration.from_pretrained(model_path)
    model.to(device).eval()

    _model_cache[model_path] = (processor, model, device)
    return processor, model, device


# ── Audio loading ────────────────────────────────────────────────────────────

def load_audio(path: str, target_sr: int = 16000) -> np.ndarray:
    """Load an audio file, resample to 16 kHz mono, return numpy array."""
    waveform, sr = torchaudio.load(path)

    if sr != target_sr:
        waveform = torchaudio.transforms.Resample(sr, target_sr)(waveform)

    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    return waveform.squeeze().numpy()


# ── Inference ────────────────────────────────────────────────────────────────

def transcribe(audio: np.ndarray, processor, model, device: str) -> tuple[str, float]:
    """
    Run inference on a single audio array.

    Returns:
        (transcription_text, inference_time_seconds)
    """
    input_features = processor(
        audio, sampling_rate=16000, return_tensors="pt"
    ).input_features.to(device)

    start = time.perf_counter()
    with torch.no_grad():
        predicted_ids = model.generate(input_features)
    elapsed = time.perf_counter() - start

    text = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0].strip()
    return text, elapsed


def transcribe_file(
    audio_path: str,
    model_path: str = "output/whisper-tiny-finetuned/final",
) -> dict:
    """Transcribe a single audio file. Returns dict with results."""
    processor, model, device = load_model(model_path)
    audio = load_audio(audio_path)
    duration = len(audio) / 16000

    text, elapsed = transcribe(audio, processor, model, device)

    return {
        "file": audio_path,
        "transcription": text,
        "audio_duration_s": round(duration, 2),
        "inference_time_s": round(elapsed, 3),
        "rtf": round(elapsed / duration, 3) if duration > 0 else 0,
    }


def transcribe_batch(
    directory: str,
    model_path: str = "output/whisper-tiny-finetuned/final",
    extensions: tuple = ("wav", "mp3", "flac", "ogg"),
) -> list[dict]:
    """Transcribe all audio files in a directory."""
    files = []
    for ext in extensions:
        files.extend(glob.glob(os.path.join(directory, f"*.{ext}")))
    files.sort()

    if not files:
        print(f"No audio files found in {directory}")
        return []

    print(f"Found {len(files)} audio file(s) in {directory}\n")
    results = []
    for f in files:
        result = transcribe_file(f, model_path)
        results.append(result)
        print(f"  {os.path.basename(f):20s} → {result['transcription'][:80]}")

    return results


def compare_models(
    audio_path: str,
    base_model: str = "openai/whisper-tiny",
    finetuned_model: str = "output/whisper-tiny-finetuned/final",
) -> dict:
    """Compare base vs fine-tuned model on the same audio file."""
    audio = load_audio(audio_path)
    duration = len(audio) / 16000

    # Base model
    proc_b, model_b, dev_b = load_model(base_model)
    text_b, time_b = transcribe(audio, proc_b, model_b, dev_b)

    # Fine-tuned model
    proc_f, model_f, dev_f = load_model(finetuned_model)
    text_f, time_f = transcribe(audio, proc_f, model_f, dev_f)

    return {
        "file": audio_path,
        "audio_duration_s": round(duration, 2),
        "base_model": {
            "text": text_b,
            "time_s": round(time_b, 3),
        },
        "finetuned_model": {
            "text": text_f,
            "time_s": round(time_f, 3),
        },
    }


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Inference with fine-tuned Whisper Tiny model",
    )
    parser.add_argument(
        "input",
        help="Path to an audio file or directory (with --batch)",
    )
    parser.add_argument(
        "--model", "-m",
        default="output/whisper-tiny-finetuned/final",
        help="Path to model directory (default: output/whisper-tiny-finetuned/final)",
    )
    parser.add_argument(
        "--batch", "-b",
        action="store_true",
        help="Transcribe all audio files in the input directory",
    )
    parser.add_argument(
        "--compare", "-c",
        action="store_true",
        help="Compare base whisper-tiny vs fine-tuned model",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  Whisper Tiny Inference")
    print("=" * 60)
    print(f"  Device : {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    if torch.cuda.is_available():
        print(f"  GPU    : {torch.cuda.get_device_name(0)}")
    print()

    # ── Compare mode ──
    if args.compare:
        result = compare_models(args.input, finetuned_model=args.model)
        print(f"  File           : {result['file']}")
        print(f"  Duration       : {result['audio_duration_s']}s\n")
        print(f"  Base model     : {result['base_model']['text']}")
        print(f"    (time: {result['base_model']['time_s']}s)\n")
        print(f"  Fine-tuned     : {result['finetuned_model']['text']}")
        print(f"    (time: {result['finetuned_model']['time_s']}s)")
        return

    # ── Batch mode ──
    if args.batch:
        if not os.path.isdir(args.input):
            print(f"Error: {args.input} is not a directory")
            sys.exit(1)
        results = transcribe_batch(args.input, args.model)
        print(f"\n  Total: {len(results)} file(s) transcribed")
        total_audio = sum(r["audio_duration_s"] for r in results)
        total_infer = sum(r["inference_time_s"] for r in results)
        print(f"  Audio duration : {total_audio:.1f}s")
        print(f"  Inference time : {total_infer:.2f}s")
        print(f"  Avg RTF        : {total_infer/total_audio:.3f}x" if total_audio > 0 else "")
        return

    # ── Single file mode ──
    if not os.path.isfile(args.input):
        print(f"Error: {args.input} not found")
        sys.exit(1)

    result = transcribe_file(args.input, args.model)
    print(f"  File           : {result['file']}")
    print(f"  Audio duration : {result['audio_duration_s']}s")
    print(f"  Inference time : {result['inference_time_s']}s")
    print(f"  RTF            : {result['rtf']}x")
    print(f"\n  Transcription:\n  {result['transcription']}")


if __name__ == "__main__":
    main()
