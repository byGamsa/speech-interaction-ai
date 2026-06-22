import argparse
import imageio_ffmpeg

from dataclasses import dataclass
from pathlib import Path

from piper import SynthesisConfig

from plotting import save_all_plots
from reporting import analyze_word_confusions, print_results, results_to_dataframe, save_results_csv
from speech_pipeline import (
    load_sentences,
    mix_with_noise,
    synthesize_speech,
    transcribe_and_evaluate,
)


SENTENCES_COUNT = 0
NOISE_TYPES = ["white", "pink", "blue"]
SNR_LEVELS = [0, 5, 10, 15, 20]
WHISPER_MODEL_NAME = "turbo"


@dataclass
class AppConfig:
    output_dir: Path
    charts_dir: Path
    wav_dir: Path
    results_dir: Path
    tmp_dir: Path
    voices_dir: Path
    data_dir: Path
    ffmpeg_bin: str
    selected_voices: list[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Piper synthesis, noise mixing, Whisper transcription and WER/CER evaluation."
    )
    parser.add_argument(
        "--voices",
        nargs=2,
        required=True,
        help="Exactly two Piper voice names without .onnx, e.g. en_US-ryan-medium en_US-amy-medium",
    )
    return parser.parse_args()


def build_config() -> AppConfig:
    args = parse_args()

    output_dir = Path("output")
    charts_dir = output_dir / "charts"
    wav_dir = output_dir / "wav"
    results_dir = output_dir / "results"
    tmp_dir = output_dir / "tmp"

    output_dir.mkdir(exist_ok=True)
    charts_dir.mkdir(exist_ok=True)
    wav_dir.mkdir(exist_ok=True)
    results_dir.mkdir(exist_ok=True)
    tmp_dir.mkdir(exist_ok=True)

    for file in wav_dir.glob("*.wav"):
        file.unlink()

    return AppConfig(
        output_dir=output_dir,
        charts_dir=charts_dir,
        wav_dir=wav_dir,
        results_dir=results_dir,
        tmp_dir=tmp_dir,
        voices_dir=Path("voices"),
        data_dir=Path("data"),
        ffmpeg_bin=imageio_ffmpeg.get_ffmpeg_exe(),
        selected_voices=args.voices,
    )


def main():
    config = build_config()
    syn_config = SynthesisConfig(
        volume=0.5,
        length_scale=1.0,
    )

    selected_sentences = load_sentences(config.data_dir, SENTENCES_COUNT)

    speech_items = synthesize_speech(
        selected_voices=config.selected_voices,
        selected_sentences=selected_sentences,
        voices_dir=config.voices_dir,
        wav_dir=config.wav_dir,
        syn_config=syn_config,
    )

    mixed_files = mix_with_noise(
        speech_items=speech_items,
        wav_dir=config.wav_dir,
        noise_types=NOISE_TYPES,
        snr_levels=SNR_LEVELS,
    )

    results = transcribe_and_evaluate(
        mixed_files=mixed_files,
        whisper_model_name=WHISPER_MODEL_NAME,
    )

    print_results(results)
    save_results_csv(results, config.results_dir)
    analyze_word_confusions(results, config.results_dir)

    df = results_to_dataframe(results)
    save_all_plots(
        df=df,
        selected_voices=config.selected_voices,
        charts_dir=config.charts_dir,
        noise_types=NOISE_TYPES,
        snr_levels=SNR_LEVELS,
    )


if __name__ == "__main__":
    main()
