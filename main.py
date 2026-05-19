import wave
import random
import numpy as np

from pathlib import Path
from piper import PiperVoice, SynthesisConfig
from scipy.fft import rfft, irfft

SENTENCES_COUNT = 10
OUTPUT_DIR = Path("output")
VOICES_DIR = Path("voices")
DATA_DIR = Path("data")

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

# clear the output directory
for file in OUTPUT_DIR.glob("*.wav"):
    file.unlink()

# synthesize each selected sentence and save as a wav file
for sentence in selected_sentences:
    print(f"Synthesizing: {sentence}")
    
    with wave.open(f'{OUTPUT_DIR}/{sentence[:10]}.wav', "wb") as wav_file:
        voice.synthesize_wav(sentence, wav_file, syn_config=syn_config)

def generate_white_noise(num_samples): 
    # generating white noise using a normal distribution
    noise_samples = np.random.normal(0, 0.5, num_samples) 
    return noise_samples

def generate_pink_noise(num_samples):  
    # generating pink noise using the Real Fast Fourier Transform (RFFT)
    white_noise = np.random.normal(0, 1, num_samples)

    X = rfft(white_noise)
 
    S = np.zeros_like(X)
    freqs = np.fft.rfftfreq(num_samples, d=1/num_samples)

    S[1:] = 1 / np.sqrt(freqs[1:])

    pink_fft = X * S
    pink_noise = irfft(pink_fft)

    # Normalize the pink noise to be in the range [-1, 1]
    pink_noise = pink_noise / np.max(np.abs(pink_noise)) 
    return pink_noise

def generate_blue_noise(num_samples):
    # generating blue noise using a high-pass filter on white noise
    white_noise = np.random.normal(0, 1, num_samples)

    X = rfft(white_noise)

    freqs = np.fft.rfftfreq(num_samples, d=1/num_samples)
    S = np.sqrt(freqs)

    blue_fft = X * S
    blue_noise = irfft(blue_fft)

    blue_noise = blue_noise / np.max(np.abs(blue_noise))
    return blue_noise

def create_noise_wav(noise_samples, filename):
    with wave.open(f'{OUTPUT_DIR}/{filename}', "wb") as wav_file:
        wav_file.setnchannels(1)  # mono
        wav_file.setsampwidth(2)  # 16-bit audio
        wav_file.setframerate(16000)  # 16kHz
        wav_file.writeframes((noise_samples * 32767).astype(np.int16).tobytes())

num_samples = 16000  # 1 second of noise at 16kHz

white_noise = generate_white_noise(num_samples)
pink_noise = generate_pink_noise(num_samples)
blue_noise = generate_blue_noise(num_samples)

# save noise samples as a wav file
create_noise_wav(white_noise, "white_noise.wav") 
create_noise_wav(pink_noise, "pink_noise.wav") 
create_noise_wav(blue_noise, "blue_noise.wav")

# mix the synthesized sentences with noise and save the results 