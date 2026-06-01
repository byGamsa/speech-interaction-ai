import csv

import pandas as pd


def print_results(results):
    for result in results:
        print(f"File: {result['file']}")
        print(f"Reference: {result['reference']}")
        print(f"Hypothesis: {result['hypothesis']}")
        print(f"Noise Type: {result['noise_type']}, SNR: {result['snr_db']} dB")
        print(f"WER: {result['wer']:.3f}")
        print(f"CER: {result['cer']:.3f}")
        print("-" * 40)


def save_results_csv(results, results_dir):
    results_csv = results_dir / "results.csv"
    with open(results_csv, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "file",
                "voice_name",
                "reference",
                "hypothesis",
                "reference_norm",
                "hypothesis_norm",
                "noise_type",
                "snr_db",
                "wer",
                "cer",
            ],
        )
        writer.writeheader()
        writer.writerows(results)

    print(f"Saved results to {results_csv}")
    return results_csv


def results_to_dataframe(results):
    return pd.DataFrame(results)
