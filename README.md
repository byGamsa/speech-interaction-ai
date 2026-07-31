# speech-interaction-ai

Evaluation pipeline for Text-to-Speech (Piper) robustness under noisy conditions, transcribed and scored with OpenAI Whisper.

## Pipeline

1. **Synthesize** – Harvard sentences (`data/harvard_sentences.txt`) are synthesized with one or more [Piper](https://github.com/rhasspy/piper) voices (`voices/*.onnx`).
2. **Mix with noise** – Each synthesized clip is mixed with white, pink and blue noise at several SNR levels (0–20 dB).
3. **Transcribe** – The noisy clips are transcribed with Whisper.
4. **Evaluate** – Word Error Rate (WER) and Character Error Rate (CER) are computed against the reference sentences, plus a word-level confusion analysis (substitutions, deletions, insertions).
5. **Report** – Results are saved as CSV and visualized as line plots and heatmaps of WER/CER vs. SNR per noise type and voice.

## Requirements

- Python 3.10+
- A CUDA-capable GPU is used automatically if available, otherwise CPU
- Packages: `piper-tts`, `openai-whisper`, `torch`, `jiwer`, `numpy`, `scipy`, `pandas`, `matplotlib`, `imageio-ffmpeg`

Install them, e.g.:

```bash
pip install piper-tts openai-whisper torch jiwer numpy scipy pandas matplotlib imageio-ffmpeg
```

## Usage

Place Piper voice files (`.onnx` + `.onnx.json`) in `voices/`, then run:

```bash
python main.py --voices en_US-amy-medium en_US-ryan-medium
```

Available voices out of the box: `en_US-amy-medium`, `en_US-danny-low`, `en_US-joe-medium`, `en_US-kristin-medium`, `en_US-lessac-medium`, `en_US-ryan-medium`.

## Output

All results are written to `output/`:

- `output/wav/` – synthesized and noise-mixed audio
- `output/results/` – `results.csv` and per-voice `word_confusions_*.csv`
- `output/charts/` – WER/CER line plots and heatmaps (overall and per voice)

## Project structure

| File | Purpose |
|---|---|
| `main.py` | Entry point and pipeline configuration |
| `speech_pipeline.py` | Synthesis, noise mixing, Whisper transcription |
| `reporting.py` | CSV export and word confusion analysis |
| `plotting.py` | Chart generation |
