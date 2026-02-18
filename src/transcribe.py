"""
Transcribe audio files using Whisper Tiny (base or fine-tuned).
"""

import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import torchaudio


def load_model(model_path: str | None = None):
    """Load the Whisper model and processor."""
    model_name = model_path or "openai/whisper-tiny"
    processor = WhisperProcessor.from_pretrained(model_name)
    model = WhisperForConditionalGeneration.from_pretrained(model_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()
    return processor, model, device


def transcribe_audio(audio_path: str, model_path: str | None = None) -> str:
    """
    Transcribe a single audio file.

    Args:
        audio_path: Path to the audio file (wav, mp3, flac, etc.)
        model_path: Optional path to a fine-tuned model directory.

    Returns:
        Transcribed text string.
    """
    processor, model, device = load_model(model_path)

    # Load audio
    waveform, sample_rate = torchaudio.load(audio_path)

    # Resample to 16kHz if needed
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
        waveform = resampler(waveform)

    # Convert to mono if stereo
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    waveform = waveform.squeeze().numpy()

    # Process through model
    input_features = processor(
        waveform, sampling_rate=16000, return_tensors="pt"
    ).input_features.to(device)

    with torch.no_grad():
        predicted_ids = model.generate(input_features)

    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    return transcription.strip()
