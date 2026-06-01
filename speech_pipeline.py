import random
import re
import subprocess
import wave

import jiwer
import numpy as np
import whisper

from piper import PiperVoice, SynthesisConfig
from scipy.io import wavfile


def load_sentences(data_dir, sentences_count):
    with open(data_dir / "harvard_sentences.txt", "r", encoding="utf-8") as file:
        sentences = [line.strip() for line in file if line.strip()]
    if sentences_count == 0:
        return sentences
    return random.sample(sentences, sentences_count)


def synthesize_speech(selected_voices, selected_sentences, voices_dir, wav_dir, syn_config):
    speech_items = []

    for voice_name in selected_voices:
        voice_path = voices_dir / f"{voice_name}.onnx"
        if not voice_path.exists():
            raise FileNotFoundError(f"Voice file not found: {voice_path}")

        voice = PiperVoice.load(str(voice_path))

        for index, sentence in enumerate(selected_sentences, start=1):
            print(f"Synthesizing with {voice_name}: {sentence}")

            filename = f"{voice_name}_speech_{index:02d}.wav"
            filepath = wav_dir / filename

            with wave.open(str(filepath), "wb") as wav_file:
                voice.synthesize_wav(sentence, wav_file, syn_config=syn_config)

            speech_items.append(
                {
                    "file": filepath,
                    "speech_id": filepath.stem,
                    "voice_name": voice_name,
                    "reference": sentence,
                }
            )

    return speech_items


def mix_with_noise(speech_items, ffmpeg_bin, wav_dir, noise_types, snr_levels, base_noise_amplitude):
    mixed_files = []

    for speech_item in speech_items:
        speech_file = speech_item["file"]
        for noise_type in noise_types:
            for snr_db in snr_levels:
                noise_amplitude = base_noise_amplitude * (10 ** (-snr_db / 20))
                output_file = wav_dir / f"{speech_file.stem}_{noise_type}_noise_snr_{snr_db}db.wav"

                command = [
                    ffmpeg_bin,
                    "-y",
                    "-i",
                    str(speech_file),
                    "-filter_complex",
                    (
                        f"anoisesrc=color={noise_type}:sample_rate=16000:amplitude={noise_amplitude}[noise];"
                        f"[0:a][noise]amix=inputs=2:duration=first:dropout_transition=0,"
                        f"alimiter=limit=0.95[out]"
                    ),
                    "-map",
                    "[out]",
                    str(output_file),
                ]

                result = subprocess.run(command, check=False, capture_output=True, text=True)
                if result.returncode != 0:
                    print("FFmpeg command failed:")
                    print("Command:", " ".join(command))
                    print("STDOUT:", result.stdout)
                    print("STDERR:", result.stderr)
                    raise RuntimeError(f"FFmpeg failed with exit code {result.returncode}")

                print(f"Saved mixed file: {output_file}")

                mixed_files.append(
                    {
                        "file": output_file,
                        "speech_id": speech_item["speech_id"],
                        "voice_name": speech_item["voice_name"],
                        "reference": speech_item["reference"],
                        "noise_type": noise_type,
                        "snr_db": snr_db,
                    }
                )

    return mixed_files


def normalize_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def transcribe_and_evaluate(mixed_files, whisper_model_name):
    print("Loading Whisper model")
    whisper_model = whisper.load_model(whisper_model_name)

    results = []
    for item in mixed_files:
        audio_file = item["file"]
        reference = item["reference"]

        print(f"Transcribing: {audio_file}")

        sample_rate, audio_data = wavfile.read(str(audio_file))
        audio_np = audio_data.astype(np.float32) / 32768.0

        transcription = whisper_model.transcribe(audio_np, language="en", fp16=False)
        hypothesis = transcription["text"]

        reference_norm = normalize_text(reference)
        hypothesis_norm = normalize_text(hypothesis)

        results.append(
            {
                "file": audio_file,
                "voice_name": item["voice_name"],
                "reference": reference,
                "hypothesis": hypothesis,
                "reference_norm": reference_norm,
                "hypothesis_norm": hypothesis_norm,
                "noise_type": item["noise_type"],
                "snr_db": item["snr_db"],
                "wer": jiwer.wer(reference_norm, hypothesis_norm),
                "cer": jiwer.cer(reference_norm, hypothesis_norm),
            }
        )

    return results
