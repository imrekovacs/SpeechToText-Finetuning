

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
│    transcribe | train [--strategy] | evaluate       │
└──────────┬──────────────┬──────────────┬────────────┘
           │              │              │
     ┌─────▼─────┐  ┌────▼─────┐  ┌─────▼──────┐
     │transcribe  │  │  train   │  │  evaluate   │
     │   .py      │  │   .py    │  │    .py      │
     └─────┬──────┘  └────┬─────┘  └─────┬───────┘
           │              │              │
           │         ┌────▼──────────┐   │
           │         │  PEFT (LoRA)  │   │
           │         │  + Quantize   │   │
           │         │  + Freeze     │   │
           │         │  + Grad Ckpt  │   │
           │         └────┬──────────┘   │
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

This installs all dependencies including `peft` (for LoRA/QLoRA) and `bitsandbytes` (for 4-bit quantization).

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

## Fine-Tuning Techniques

This project supports multiple fine-tuning strategies that let you trade off between quality, speed, and VRAM usage. Configure them in `config.yaml` under the `finetuning:` section, or override via CLI flags.

### Strategy Comparison

| Strategy | Trainable Params | VRAM Usage | Training Speed | When to Use |
|----------|-----------------|------------|----------------|-------------|
| **full** | 100% (~39M) | ~3-4 GB | Baseline | Best quality, enough VRAM |
| **lora** | ~0.6% (~230K) | ~1.5-2 GB | 2-3× faster | Limited VRAM, avoid overfitting |
| **qlora** | ~0.6% (~230K) | ~1-1.5 GB | Similar to LoRA | Very limited VRAM (4 GB cards) |

### LoRA (Low-Rank Adaptation)

Instead of updating all 39M parameters, LoRA injects small trainable matrices (rank `r`) into the attention layers (`q_proj`, `v_proj`). The base model weights stay frozen.

**Advantages:**
- Trains ~0.6% of parameters → much faster, less VRAM
- Adapter weights are tiny (~2-5 MB vs 150 MB for full model)
- Less prone to catastrophic forgetting on small datasets
- Multiple adapters can be swapped without reloading the base model

**Config:**
```yaml
finetuning:
  strategy: "lora"
  lora:
    r: 16              # Rank — higher = more capacity, more params
    lora_alpha: 32     # Scaling factor (typically 2× rank)
    lora_dropout: 0.05 # Regularization
    target_modules:    # Which layers to adapt
      - "q_proj"
      - "v_proj"
```

### QLoRA (Quantized LoRA)

QLoRA loads the base model in **4-bit precision** (NF4 quantization) and trains LoRA adapters on top. This cuts VRAM usage by ~50% compared to full precision.

**Config:**
```yaml
finetuning:
  strategy: "qlora"
  # Same lora: section as above
```

> **Note:** QLoRA requires `bitsandbytes`. The first run may take longer due to quantization overhead.

### Encoder Freezing

Whisper's encoder converts audio spectrograms to embeddings. If your audio domain is similar to the pre-training data (clean English speech), the encoder features are already excellent — freezing it prevents catastrophic forgetting and halves trainable parameters.

**Config:**
```yaml
finetuning:
  freeze_encoder: true
```

**When to freeze:**
- Clean English speech (encoder already optimised)
- Small datasets (< 1000 samples) where overfitting is a risk
- When combining with LoRA for maximum efficiency

**When NOT to freeze:**
- Very different audio domain (e.g. noisy factory environments)
- New language the model hasn't seen
- Large dataset (> 10,000 samples) where you want maximum adaptation

### Gradient Checkpointing

Normally, all intermediate activations are stored in VRAM for the backward pass. Gradient checkpointing discards them and recomputes on-the-fly, cutting VRAM ~40% at the cost of ~20% slower training.

**Config:**
```yaml
finetuning:
  gradient_checkpointing: true
```

### CLI Overrides

You can override any fine-tuning setting from the command line without editing `config.yaml`:

```bash
# Train with LoRA strategy
.venv\Scripts\python.exe cli.py train config.yaml --strategy lora

# Train with full strategy, frozen encoder, gradient checkpointing
.venv\Scripts\python.exe cli.py train config.yaml --strategy full --freeze-encoder --grad-ckpt

# Train with QLoRA
.venv\Scripts\python.exe cli.py train config.yaml --strategy qlora
```

### Combining Techniques

The techniques are composable. A recommended setup for consumer GPUs:

```yaml
finetuning:
  strategy: "lora"
  freeze_encoder: true
  gradient_checkpointing: true
  lora:
    r: 16
    lora_alpha: 32
    lora_dropout: 0.05
    target_modules: ["q_proj", "v_proj"]
```

This combination trains only ~230K parameters (LoRA adapters on the decoder's attention layers), uses gradient checkpointing to reduce VRAM, and freezes the encoder — ideal for an RTX 2080 or similar 8 GB card.

---

## Configuration

All training parameters are in `config.yaml`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model.name` | `openai/whisper-tiny` | Base model from HuggingFace |
| `model.language` | `en` | Target language |
| `data.train_dir` | `data` | Folder with audio + metadata |
| `data.test_split` | `0.2` | Fraction reserved for evaluation |
| `finetuning.strategy` | `lora` | `full`, `lora`, or `qlora` |
| `finetuning.freeze_encoder` | `true` | Freeze the audio encoder |
| `finetuning.gradient_checkpointing` | `true` | Enable gradient checkpointing |
| `finetuning.lora.r` | `16` | LoRA rank (higher = more capacity) |
| `finetuning.lora.lora_alpha` | `32` | LoRA scaling factor |
| `finetuning.lora.lora_dropout` | `0.05` | Dropout on adapter weights |
| `finetuning.lora.target_modules` | `["q_proj","v_proj"]` | Layers to attach adapters to |
| `training.num_train_epochs` | `3` | Number of training passes |
| `training.learning_rate` | `1e-5` | Learning rate (auto-raised to 1e-3 for LoRA) |
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
