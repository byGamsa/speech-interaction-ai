import csv
from collections import Counter

import jiwer
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


def analyze_word_confusions(results, results_dir, top_n=20):
    """
    Compares reference and hypothesis word by word using alignment.
    Counts how often each word was substituted, inserted or deleted.
    Groups results by voice and saves a CSV for each voice.
    """
    # collect confusions per voice
    substitutions_by_voice = {}
    deletions_by_voice = {}
    insertions_by_voice = {}

    for result in results:
        voice = result["voice_name"]
        ref_text = result["reference_norm"]
        hyp_text = result["hypothesis_norm"]
        ref_words = ref_text.split()
        hyp_words = hyp_text.split()

        if voice not in substitutions_by_voice:
            substitutions_by_voice[voice] = Counter()
            deletions_by_voice[voice] = Counter()
            insertions_by_voice[voice] = Counter()

        if not ref_words or not hyp_words:
            continue

        # use jiwer alignment to get word-level operations
        alignment = jiwer.process_words(ref_text, hyp_text)

        for chunk in alignment.alignments[0]:
            if chunk.type == "substitute":
                for i_ref, i_hyp in zip(
                    range(chunk.ref_start_idx, chunk.ref_end_idx),
                    range(chunk.hyp_start_idx, chunk.hyp_end_idx),
                ):
                    ref_word = ref_words[i_ref]
                    hyp_word = hyp_words[i_hyp]
                    substitutions_by_voice[voice][(ref_word, hyp_word)] += 1

            elif chunk.type == "delete":
                for i_ref in range(chunk.ref_start_idx, chunk.ref_end_idx):
                    deletions_by_voice[voice][ref_words[i_ref]] += 1

            elif chunk.type == "insert":
                for i_hyp in range(chunk.hyp_start_idx, chunk.hyp_end_idx):
                    insertions_by_voice[voice][hyp_words[i_hyp]] += 1

    # print and save results per voice
    for voice in substitutions_by_voice:
        print(f"\n{'=' * 60}")
        print(f"Word confusion analysis for: {voice}")
        print(f"{'=' * 60}")

        top_subs = substitutions_by_voice[voice].most_common(top_n)
        top_dels = deletions_by_voice[voice].most_common(top_n)
        top_ins = insertions_by_voice[voice].most_common(top_n)

        if top_subs:
            print(f"\nTop {top_n} substitutions (reference -> hypothesis):")
            for (ref_word, hyp_word), count in top_subs:
                print(f"  '{ref_word}' -> '{hyp_word}' ({count}x)")

        if top_dels:
            print(f"\nTop {top_n} deleted words (in reference but missed by Whisper):")
            for word, count in top_dels:
                print(f"  '{word}' ({count}x)")

        if top_ins:
            print(f"\nTop {top_n} inserted words (not in reference but added by Whisper):")
            for word, count in top_ins:
                print(f"  '{word}' ({count}x)")

        # save to CSV
        rows = []
        for (ref_word, hyp_word), count in top_subs:
            rows.append({"voice": voice, "type": "substitution", "reference": ref_word, "hypothesis": hyp_word, "count": count})
        for word, count in top_dels:
            rows.append({"voice": voice, "type": "deletion", "reference": word, "hypothesis": "", "count": count})
        for word, count in top_ins:
            rows.append({"voice": voice, "type": "insertion", "reference": "", "hypothesis": word, "count": count})

        csv_path = results_dir / f"word_confusions_{voice}.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["voice", "type", "reference", "hypothesis", "count"])
            writer.writeheader()
            writer.writerows(rows)

        print(f"\nSaved word confusions to {csv_path}")


def results_to_dataframe(results):
    return pd.DataFrame(results)
