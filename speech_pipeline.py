import random
import re
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


def generate_colored_noise(color, num_samples): 
    # standard white noise (random values, normal distribution)
    white_noise = np.random.randn(num_samples)

    if color == "white":
        return white_noise

    # transform into frequency domain with FFT
    frequencies = np.fft.rfftfreq(num_samples, d=1.0)
    frequencies[0] = 1.0  # avoid division by zero
    spectrum = np.fft.rfft(white_noise)

    # shape the spectrum according to the desired noise color
    if color == "pink":
        spectrum = spectrum / np.sqrt(frequencies)
    elif color == "blue":
        spectrum = spectrum * np.sqrt(frequencies)

    # transform back to time domain
    colored_noise = np.fft.irfft(spectrum, n=num_samples)
    return colored_noise


def mix_with_noise(speech_items, wav_dir, noise_types, snr_levels): 
    mixed_files = []

    for speech_item in speech_items:
        speech_file = speech_item["file"]

        # read the speech audio and convert to float in range [-1, 1]
        sample_rate, audio_data = wavfile.read(str(speech_file))
        speech = audio_data.astype(np.float32) / 32768.0

        # calculate the average energy of the speech signal with root mean square
        rms_speech = np.sqrt(np.mean(speech ** 2))

        for noise_type in noise_types:
            # generate noise with the same length as the speech
            noise = generate_colored_noise(noise_type, len(speech))

            # measure the current RMS of the generated noise
            rms_noise = np.sqrt(np.mean(noise ** 2))
            if rms_noise == 0:
                rms_noise = 1e-10  # safety check to avoid division by zero

            for snr_db in snr_levels:
                # calculate the desired noise RMS based on the target SNR
                # SNR(dB) = 20 * log10(RMS_speech / RMS_noise)
                # => RMS_noise = RMS_speech / 10^(SNR/20)
                desired_noise_rms = rms_speech / (10 ** (snr_db / 20))

                # scale the noise so its RMS matches the desired level
                scale_factor = desired_noise_rms / rms_noise
                scaled_noise = noise * scale_factor

                # add speech and noise together
                mixed = speech + scaled_noise

                # prevent clipping: if the signal exceeds 0.95, scale it down
                peak = np.max(np.abs(mixed))
                if peak > 0.95:
                    mixed = mixed * (0.95 / peak)

                # save the mixed audio as 16-bit WAV
                output_file = wav_dir / f"{speech_file.stem}_{noise_type}_noise_snr_{snr_db}db.wav"
                mixed_int16 = np.clip(mixed * 32768, -32768, 32767).astype(np.int16)
                wavfile.write(str(output_file), sample_rate, mixed_int16)

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
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading Whisper model on {device}")
    whisper_model = whisper.load_model(whisper_model_name, device=device)

    use_fp16 = device == "cuda"

    results = []
    for item in mixed_files:
        audio_file = item["file"]
        reference = item["reference"]

        print(f"Transcribing: {audio_file}")

        sample_rate, audio_data = wavfile.read(str(audio_file))
        audio_np = audio_data.astype(np.float32) / 32768.0

        transcription = whisper_model.transcribe(audio_np, language="en", fp16=use_fp16)
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
