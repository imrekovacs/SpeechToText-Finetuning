

# Whisper Tiny Fine-Tuning CLI

A beginner-friendly Python CLI application for **fine-tuning OpenAI's Whisper Tiny** speech-to-text model on your own audio data. Train the model on a custom dataset, run inference, and evaluate accuracy — all from the command line.

---

## What Does This Project Do?

[OpenAI Whisper](https://github.com/openai/whisper) is a powerful automatic speech recognition (ASR) model. The **"tiny"** variant (~39M parameters, ~150 MB) is the smallest and fastest, making it ideal for fine-tuning on a single consumer GPU.

This project lets you:

1. **Fine-tune** Whisper Tiny on your own audio + transcript pairs
2. **Transcribe** audio files using the fine-tuned (or base) model
3. **Evaluate** model accuracy using Word Error Rate (WER)
4. **Run inference** with benchmarking (single file, batch, or side-by-side comparison)

Fine-tuning is useful when you need Whisper to perform better on:
- A specific accent or dialect
- Domain-specific vocabulary (medical, legal, technical)
- Low-resource languages
- Noisy audio environments

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                    CLI (cli.py)                      │
│         transcribe | train | evaluate               │
└──────────┬──────────────┬──────────────┬────────────┘
           │              │              │
     ┌─────▼─────┐  ┌────▼─────┐  ┌─────▼──────┐
     │transcribe  │  │  train   │  │  evaluate   │
     │   .py      │  │   .py    │  │    .py      │
     └─────┬──────┘  └────┬─────┘  └─────┬───────┘
           │              │              │
     ┌─────▼──────────────▼──────────────▼────────┐
     │        HuggingFace Transformers            │
     │   WhisperProcessor + WhisperForConditional │
     │              Generation                    │
     └─────────────────┬──────────────────────────┘
                       │
              ┌────────▼────────┐
              │   PyTorch +     │
              │   torchaudio    │
              │   (CUDA GPU)    │
              └─────────────────┘
```

**Key components:**

| File | Purpose |
|------|---------|
| `cli.py` | Main entry point — Typer-based CLI with `transcribe`, `train`, `evaluate` commands |
| `src/train.py` | Fine-tuning pipeline using HuggingFace `Seq2SeqTrainer` |
| `src/transcribe.py` | Audio loading, resampling, and model inference |
| `src/evaluate.py` | WER evaluation on the test split |
| `inference.py` | Standalone inference script with batch & comparison modes |
| `prepare_sample_data.py` | Downloads a small LibriSpeech sample dataset for testing |
| `config.yaml` | All training hyperparameters and data paths |

---

## Project Structure

```
SpeechToText-Finetuning/
├── cli.py                  # Main CLI entry point
├── inference.py            # Standalone inference script
├── prepare_sample_data.py  # Download/generate sample dataset
├── config.yaml             # Training configuration
├── requirements.txt        # Python dependencies
├── src/
│   ├── __init__.py
│   ├── train.py            # Fine-tuning logic
│   ├── transcribe.py       # Transcription logic
│   └── evaluate.py         # WER evaluation
├── data/                   # Your audio files + metadata.csv
│   ├── metadata.csv
│   ├── sample_000.wav
│   └── ...
└── output/                 # Fine-tuned model checkpoints (git-ignored)
    └── whisper-tiny-finetuned/
        └── final/
```

---

## Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **GPU** | NVIDIA GPU with 4 GB VRAM (e.g. GTX 1650) | NVIDIA GPU with 8+ GB VRAM (e.g. RTX 2080, 3060, 4060) |
| **RAM** | 8 GB | 16 GB |
| **Disk** | 5 GB free (model + deps) | 20 GB free |
| **CUDA** | CUDA 12.1+ | CUDA 12.4+ |
| **OS** | Windows 10/11, Linux | Windows 10/11, Linux |
| **CPU-only** | Possible but very slow | Not recommended for training |

> **Note:** Whisper Tiny is the smallest Whisper model (~150 MB). Fine-tuning it on a small dataset (10-100 samples) takes only seconds to minutes on a modern GPU. Inference runs at ~30x real-time on an RTX 2080.

---

## Installation

### Prerequisites

- **Python 3.10+** installed
- **NVIDIA GPU** with [CUDA drivers](https://developer.nvidia.com/cuda-downloads) installed
- **[uv](https://docs.astral.sh/uv/getting-started/installation/)** package manager (recommended) or pip

### Step 1: Clone the Repository

```bash
git clone https://github.com/imrekovacs/SpeechToText-Finetuning.git
cd SpeechToText-Finetuning
```

### Step 2: Create a Virtual Environment

```bash
uv venv .venv --python 3.10
```

Activate it:

- **Windows (PowerShell):** `.venv\Scripts\activate`
- **Linux/macOS:** `source .venv/bin/activate`

### Step 3: Install PyTorch with CUDA

```bash
uv pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124
```

> Change `cu124` to match your CUDA version. Check with `nvidia-smi`. Common options: `cu121`, `cu124`.

### Step 4: Install Dependencies

```bash
uv pip install -r requirements.txt --extra-index-url https://pypi.org/simple/ --index-url https://download.pytorch.org/whl/cu124
```

### Step 5: Prepare Sample Data (Optional)

To quickly test with a small LibriSpeech dataset (10 samples):

```bash
.venv\Scripts\python.exe prepare_sample_data.py
```

---

## Usage

All commands use the venv Python. On Windows: `.venv\Scripts\python.exe`, on Linux: `.venv/bin/python`.

### Prepare Your Dataset

Place your audio files (`.wav`) and a `metadata.csv` in the `data/` folder:

```
data/
├── metadata.csv
├── audio1.wav
├── audio2.wav
└── ...
```

**metadata.csv format:**

```csv
file_name,transcription
audio1.wav,This is the transcription of the first audio file.
audio2.wav,Another transcription goes here.
```

> **Tip:** If you have `.txt` files with the same name as your audio files (e.g. `audio1.txt`), the system will auto-generate `metadata.csv` for you.

### Train (Fine-Tune)

```bash
.venv\Scripts\python.exe cli.py train config.yaml
```

Edit `config.yaml` to adjust epochs, batch size, learning rate, etc.

### Transcribe

```bash
# Using the fine-tuned model
.venv\Scripts\python.exe cli.py transcribe data/sample_000.wav

# Using a specific model
.venv\Scripts\python.exe cli.py transcribe data/sample_000.wav --model output/whisper-tiny-finetuned/final
```

### Evaluate

```bash
.venv\Scripts\python.exe cli.py evaluate
```

Reports Word Error Rate (WER) on the test split.

### Inference Script (Advanced)

The standalone `inference.py` provides additional features:

```bash
# Single file
.venv\Scripts\python.exe inference.py audio.wav

# Batch — transcribe all files in a directory
.venv\Scripts\python.exe inference.py data/ --batch

# Compare base model vs fine-tuned model side-by-side
.venv\Scripts\python.exe inference.py audio.wav --compare
```

---

## Configuration

All training parameters are in `config.yaml`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model.name` | `openai/whisper-tiny` | Base model from HuggingFace |
| `model.language` | `en` | Target language |
| `data.train_dir` | `data` | Folder with audio + metadata |
| `data.test_split` | `0.2` | Fraction reserved for evaluation |
| `training.num_train_epochs` | `3` | Number of training passes |
| `training.learning_rate` | `1e-5` | Learning rate |
| `training.fp16` | `true` | Mixed precision (faster, less VRAM) |
| `training.per_device_train_batch_size` | `4` | Batch size per GPU |

---

## Tips for Beginners

- **Start small:** Use the sample dataset (`prepare_sample_data.py`) to verify everything works before using your own data.
- **More data = better results:** 10 samples is a proof of concept. For real improvements, use 100-1000+ audio clips.
- **Audio format:** Whisper expects 16 kHz mono WAV. The scripts auto-resample, but starting with 16 kHz avoids quality loss.
- **VRAM errors:** Reduce `per_device_train_batch_size` to `1` or `2` in `config.yaml` if you get CUDA out-of-memory errors.
- **CPU fallback:** Training will work on CPU but will be much slower. Set `fp16: false` in `config.yaml` when using CPU.

---

## License

MIT
