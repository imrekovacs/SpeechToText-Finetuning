

# Test Model
.venv\Scripts\python.exe cli.py evaluate
.venv\Scripts\python.exe cli.py transcribe data/sample_000.wav
.venv\\Scripts\\python.exe inference.py data/sample_000.wav --model output/whisper-tiny-finetuned/final
