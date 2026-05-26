import csv
import os
import re
import wave
import random
import jiwer
import subprocess
import imageio_ffmpeg
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from piper import PiperVoice, SynthesisConfig
from scipy.fft import rfft, irfft
from scipy.io import wavfile
import whisper

SENTENCES_COUNT = 10
OUTPUT_DIR = Path("output")
VOICES_DIR = Path("voices")
DATA_DIR = Path("data")

FFMPEG_BIN = imageio_ffmpeg.get_ffmpeg_exe() 

NOISE_TYPES = ["white", "pink", "blue"]
SNR_LEVELS = [0, 5, 10, 15, 20]

BASE_NOISE_AMPLITUDE = 0.3

syn_config = SynthesisConfig(
    volume=0.5,  # half as loud
    length_scale=1.0, 
    #noise_scale=1.0, 
    #noise_w_scale=1.0, 
    #normalize_audio=False, 
)

# load the voice
voice = PiperVoice.load(f"{VOICES_DIR}/en_US-ryan-medium.onnx")

# load sentences from harvard_sentences
with open(f"{DATA_DIR}/harvard_sentences.txt", "r", encoding="utf-8") as f:
    sentences = [line.strip() for line in f if line.strip()]

selected_sentences = random.sample(sentences, SENTENCES_COUNT)

OUTPUT_DIR.mkdir(exist_ok=True)

# clear the output directory
for file in OUTPUT_DIR.glob("*.wav"):
    file.unlink()

speech_files = []
speech_references = {}

# synthesize each selected sentence and save as a wav file
for index, sentence in enumerate(selected_sentences, start=1):
    print(f"Synthesizing: {sentence}")

    filename = f"speech_{index:02d}.wav"
    filepath = OUTPUT_DIR / filename

    with wave.open(str(filepath), "wb") as wav_file:
        voice.synthesize_wav(sentence, wav_file, syn_config=syn_config)

    speech_files.append(filepath)
    speech_references[filepath.stem] = sentence

mixed_files = []

for speech_file in speech_files:
    for noise_type in NOISE_TYPES:
        for snr_db in SNR_LEVELS:
            noise_amplitude = BASE_NOISE_AMPLITUDE * (10 ** (-snr_db / 20))
            output_file = OUTPUT_DIR / f"{speech_file.stem}_{noise_type}_noise_snr_{snr_db}db.wav"

            command = [
                FFMPEG_BIN,
                "-y",
                "-i", str(speech_file),
                "-filter_complex",
                (
                    f"anoisesrc=color={noise_type}:sample_rate=16000:amplitude={noise_amplitude}[noise];"
                    f"[0:a][noise]amix=inputs=2:duration=first:dropout_transition=0,"
                    f"alimiter=limit=0.95[out]"
                ),
                "-map", "[out]",
                str(output_file)
            ]

            result = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True
            )

            if result.returncode != 0:
                print("FFmpeg command failed:")
                print("Command:", " ".join(command))
                print("STDOUT:", result.stdout)
                print("STDERR:", result.stderr)
                raise RuntimeError(f"FFmpeg failed with exit code {result.returncode}")

            print(f"Saved mixed file: {output_file}")

            mixed_files.append({
                "file": output_file,
                "speech_id": speech_file.stem,
                "noise_type": noise_type,
                "snr_db": snr_db,
            })

def normalize_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

print("Loading Whisper model")
whisper_model = whisper.load_model("small")


results = []

for item in mixed_files:
    audio_file = item["file"]
    speech_id = item["speech_id"]

    reference = speech_references[speech_id]

    print(f"Transcribing: {audio_file}")

    # load audio as numpy array to avoid Whisper's internal ffmpeg dependency
    sample_rate, audio_data = wavfile.read(str(audio_file))
    audio_np = audio_data.astype(np.float32) / 32768.0

    # transcribe the audio file using Whisper
    transcription = whisper_model.transcribe(
        audio_np,
        language="en",
        fp16=False
    )

    hypothesis = transcription["text"]

    # normalize both reference and hypothesis
    reference_norm = normalize_text(reference)
    hypothesis_norm = normalize_text(hypothesis)

    wer_score = jiwer.wer(reference_norm, hypothesis_norm)
    cer_score = jiwer.cer(reference_norm, hypothesis_norm)

    results.append({
        "file": audio_file,
        "reference": reference,
        "hypothesis": hypothesis,
        "reference_norm": reference_norm,
        "hypothesis_norm": hypothesis_norm,
        "noise_type": item["noise_type"],
        "snr_db": item["snr_db"],
        "wer": wer_score,
        "cer": cer_score,
    })

# print results
for result in results:
    print(f"File: {result['file']}")
    print(f"Reference: {result['reference']}")
    print(f"Hypothesis: {result['hypothesis']}")
    print(f"Noise Type: {result['noise_type']}, SNR: {result['snr_db']} dB")
    print(f"WER: {result['wer']:.3f}")
    print(f"CER: {result['cer']:.3f}")
    print("-" * 40)

results_csv = OUTPUT_DIR / "results.csv"

with open(results_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=[
        "file",
        "reference",
        "hypothesis",
        "reference_norm",
        "hypothesis_norm",
        "noise_type",
        "snr_db",
        "wer",
        "cer",
    ])
    writer.writeheader()
    writer.writerows(results)

print(f"Saved results to {results_csv}")

df = pd.DataFrame(results)


# wer vs snr plot
fig, ax = plt.subplots(figsize=(8, 5))

for noise_type in NOISE_TYPES:
    subset = df[df["noise_type"] == noise_type].groupby("snr_db")["wer"].mean()
    ax.plot(subset.index, subset.values * 100, marker="o", label=noise_type)

ax.set_xlabel("SNR (dB)")
ax.set_ylabel("WER (%)")
ax.set_title("WER vs SNR by Noise Type")
ax.set_xticks(SNR_LEVELS)
ax.legend()
ax.grid(True)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "wer_vs_snr.png", dpi=150)
plt.close()

print("Saved wer_vs_snr.png")


# cer vs snr plot
fig, ax = plt.subplots(figsize=(8, 5))

for noise_type in NOISE_TYPES:
    subset = df[df["noise_type"] == noise_type].groupby("snr_db")["cer"].mean()
    ax.plot(subset.index, subset.values * 100, marker="o", label=noise_type)

ax.set_xlabel("SNR (dB)")
ax.set_ylabel("CER (%)")
ax.set_title("CER vs SNR by Noise Type")
ax.set_xticks(SNR_LEVELS)
ax.legend()
ax.grid(True)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "cer_vs_snr.png", dpi=150)
plt.close()

print("Saved cer_vs_snr.png")


# wer heatmap (noise type × snr)
fig, ax = plt.subplots(figsize=(8, 4))

pivot = df.groupby(["noise_type", "snr_db"])["wer"].mean().unstack()
im = ax.imshow(pivot.values, aspect="auto", cmap="RdYlGn_r", vmin=0, vmax=1)

ax.set_xticks(range(len(pivot.columns)))
ax.set_xticklabels(pivot.columns)
ax.set_yticks(range(len(pivot.index)))
ax.set_yticklabels(pivot.index)
ax.set_xlabel("SNR (dB)")
ax.set_ylabel("Noise Type")
ax.set_title("WER Heatmap (Noise Type × SNR)")


for i in range(len(pivot.index)):
    for j in range(len(pivot.columns)):
        ax.text(j, i, f"{pivot.values[i, j]:.2f}", ha="center", va="center", fontsize=9)

plt.colorbar(im, ax=ax, label="WER")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "wer_heatmap.png", dpi=150)
plt.close()

print("Saved wer_heatmap.png")

# cer heatmap (noise type × snr)
fig, ax = plt.subplots(figsize=(8, 4))

pivot_cer = df.groupby(["noise_type", "snr_db"])["cer"].mean().unstack()
im = ax.imshow(pivot_cer.values, aspect="auto", cmap="RdYlGn_r", vmin=0, vmax=1)

ax.set_xticks(range(len(pivot_cer.columns)))
ax.set_xticklabels(pivot_cer.columns)
ax.set_yticks(range(len(pivot_cer.index)))
ax.set_yticklabels(pivot_cer.index)
ax.set_xlabel("SNR (dB)")
ax.set_ylabel("Noise Type")
ax.set_title("CER Heatmap (Noise Type × SNR)")

for i in range(len(pivot_cer.index)):
    for j in range(len(pivot_cer.columns)):
        ax.text(j, i, f"{pivot_cer.values[i, j]:.2f}", ha="center", va="center", fontsize=9)

plt.colorbar(im, ax=ax, label="CER")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "cer_heatmap.png", dpi=150)
plt.close()

print("Saved cer_heatmap.png")