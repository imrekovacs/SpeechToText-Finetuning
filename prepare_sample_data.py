"""
Generate a small synthetic sample dataset for proof-of-concept fine-tuning.

This downloads a few samples from the HuggingFace 'mozilla-foundation/common_voice_17_0'
or falls back to generating synthetic audio samples if download fails.

Usage:
    python prepare_sample_data.py
"""

import os
import csv
import numpy as np

DATA_DIR = "data"
SAMPLE_RATE = 16000
NUM_SAMPLES = 10  # Small dataset for PoC


# Sample sentences for synthetic data
SENTENCES = [
    "Hello, how are you doing today?",
    "The weather is really nice outside.",
    "I would like to order some coffee please.",
    "Can you tell me the time right now?",
    "This is a test of the speech recognition system.",
    "Fine tuning makes the model work much better.",
    "The quick brown fox jumps over the lazy dog.",
    "Artificial intelligence is changing the world.",
    "Please turn off the lights when you leave.",
    "Thank you very much for your help today.",
]


def try_download_common_voice():
    """Try to download a small subset from a public speech dataset."""
    try:
        from datasets import load_dataset, Audio
        import soundfile as sf
        import io

        print("Attempting to download sample data from 'hf-internal-testing/librispeech_asr_dummy'...")

        # Load WITHOUT audio decoding to avoid torchcodec dependency
        ds = load_dataset(
            "hf-internal-testing/librispeech_asr_dummy",
            "clean",
            split="validation",
        ).cast_column("audio", Audio(decode=False))

        os.makedirs(DATA_DIR, exist_ok=True)
        entries = []

        # Take up to NUM_SAMPLES
        for i, sample in enumerate(ds):
            if i >= NUM_SAMPLES:
                break

            # Read audio bytes from the parquet-embedded data
            audio_info = sample["audio"]
            audio_bytes = audio_info.get("bytes")

            if audio_bytes is not None:
                audio_array, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32")
            else:
                # Try reading from path as fallback
                audio_path_src = audio_info.get("path", "")
                if audio_path_src and os.path.exists(audio_path_src):
                    audio_array, sr = sf.read(audio_path_src, dtype="float32")
                else:
                    print(f"  Skipping sample {i}: no audio bytes or valid path")
                    continue

            # Resample if needed
            if sr != SAMPLE_RATE:
                from scipy.signal import resample as scipy_resample
                num_samples_new = int(len(audio_array) * SAMPLE_RATE / sr)
                audio_array = scipy_resample(audio_array, num_samples_new).astype(np.float32)

            filename = f"sample_{i:03d}.wav"
            filepath = os.path.join(DATA_DIR, filename)
            sf.write(filepath, audio_array, SAMPLE_RATE)

            transcription = sample["text"].strip()
            entries.append({"file_name": filename, "transcription": transcription})
            print(f"  Saved {filename}: \"{transcription[:60]}...\"")

        # Write metadata.csv
        meta_path = os.path.join(DATA_DIR, "metadata.csv")
        with open(meta_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["file_name", "transcription"])
            writer.writeheader()
            writer.writerows(entries)

        print(f"\nDataset ready: {len(entries)} samples in '{DATA_DIR}/'")
        print(f"Metadata written to '{meta_path}'")
        return True

    except Exception as e:
        print(f"Download failed: {e}")
        return False


def generate_synthetic_data():
    """Generate synthetic audio with tone patterns as a last-resort fallback."""
    import soundfile as sf

    os.makedirs(DATA_DIR, exist_ok=True)
    entries = []

    print("Generating synthetic audio samples...")
    for i, sentence in enumerate(SENTENCES[:NUM_SAMPLES]):
        # Create a pseudo-speech signal (varying frequencies to simulate speech patterns)
        duration = 2.0 + len(sentence) * 0.05  # Longer text = longer audio
        t = np.linspace(0, duration, int(SAMPLE_RATE * duration), dtype=np.float32)

        # Mix of frequencies to simulate speech-like audio
        np.random.seed(i)
        signal = np.zeros_like(t)
        for _ in range(5):
            freq = np.random.uniform(100, 400)
            amp = np.random.uniform(0.1, 0.3)
            signal += amp * np.sin(2 * np.pi * freq * t)

        # Add envelope (speech has pauses)
        envelope = np.ones_like(t)
        num_words = len(sentence.split())
        word_duration = duration / num_words
        for w in range(num_words):
            start = int(w * word_duration * SAMPLE_RATE)
            gap_start = int((w * word_duration + word_duration * 0.8) * SAMPLE_RATE)
            gap_end = int(((w + 1) * word_duration) * SAMPLE_RATE)
            gap_end = min(gap_end, len(envelope))
            gap_start = min(gap_start, len(envelope))
            envelope[gap_start:gap_end] *= 0.1

        signal *= envelope
        signal = signal / (np.max(np.abs(signal)) + 1e-8) * 0.8  # Normalize

        filename = f"sample_{i:03d}.wav"
        filepath = os.path.join(DATA_DIR, filename)
        sf.write(filepath, signal, SAMPLE_RATE)

        entries.append({"file_name": filename, "transcription": sentence})
        print(f"  Created {filename}: \"{sentence}\"")

    # Write metadata.csv
    meta_path = os.path.join(DATA_DIR, "metadata.csv")
    with open(meta_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["file_name", "transcription"])
        writer.writeheader()
        writer.writerows(entries)

    print(f"\nSynthetic dataset ready: {len(entries)} samples in '{DATA_DIR}/'")
    print(f"Metadata written to '{meta_path}'")


if __name__ == "__main__":
    print("=" * 60)
    print("  Preparing sample dataset for Whisper fine-tuning PoC")
    print("=" * 60)
    print()

    # Try downloading real speech data first, fall back to synthetic
    if not try_download_common_voice():
        print("\nFalling back to synthetic data generation...")
        generate_synthetic_data()

    print("\nDone! You can now run: python cli.py train config.yaml")
