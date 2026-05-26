import csv
import os
import re
import wave
import random
import jiwer
import numpy as np
import subprocess
import imageio_ffmpeg

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
SNR_DB = 10

BASE_NOISE_AMPLITUDE = 0.3
NOISE_AMPLITUDE = BASE_NOISE_AMPLITUDE * (10 ** (-SNR_DB / 20))

syn_config = SynthesisConfig(
    volume=0.5,  # half as loud
    length_scale=1.0, 
    #noise_scale=1.0, 
    #noise_w_scale=1.0, 
    #normalize_audio=False, 
)

# load the voice
voice = PiperVoice.load(f"{VOICES_DIR}/en_US-lessac-medium.onnx")

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
        output_file = OUTPUT_DIR / f"{speech_file.stem}_{noise_type}_noise_snr_{SNR_DB}db.wav"

        command = [
            FFMPEG_BIN,
            "-y",
            "-i", str(speech_file),
            "-filter_complex",
            (
                f"anoisesrc=color={noise_type}:sample_rate=16000:amplitude={NOISE_AMPLITUDE}[noise];"
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
            "snr_db": SNR_DB,
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