import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import os
import json
import numpy as np
import pandas as pd

from test_rhythm_align import align_word_characters, map_words_to_phonemes

MANIFEST_PATH = "benchmarks/manifest.json"


def compute_robust_metrics(phoneme_events: list):
    """Calculates robust rhythm and C/V metrics for benchmark aggregation."""
    vowels = [e for e in phoneme_events if e["is_vowel"]]
    consonants = [e for e in phoneme_events if not e["is_vowel"]]

    sum_v = sum(e["dur_ms"] for e in vowels)
    sum_c = sum(e["dur_ms"] for e in consonants)
    total_ms = sum_v + sum_c

    pct_v = (sum_v / total_ms * 100.0) if total_ms > 0 else 0.0
    pct_c = 100.0 - pct_v

    # 30ms floor clipped nPVI-V
    raw_durs = [e["dur_ms"] for e in vowels if e["dur_ms"] > 5.0]
    clipped_durs = [max(30.0, d) for d in raw_durs]
    m = len(clipped_durs)

    npvi_raw = 0.0
    npvi_clipped = 0.0
    if m > 1:
        npvi_raw = (100.0 / (m - 1)) * sum(
            abs(raw_durs[k] - raw_durs[k + 1]) / ((raw_durs[k] + raw_durs[k + 1]) / 2.0)
            for k in range(m - 1)
        )
        npvi_clipped = (100.0 / (m - 1)) * sum(
            abs(clipped_durs[k] - clipped_durs[k + 1]) / ((clipped_durs[k] + clipped_durs[k + 1]) / 2.0)
            for k in range(m - 1)
        )

    s_durs = [e["dur_ms"] for e in vowels if e["stress"] in (1, 2)]
    u_durs = [e["dur_ms"] for e in vowels if e["stress"] == 0]

    s_mean = float(np.mean(s_durs)) if s_durs else 0.0
    u_mean = float(np.mean(u_durs)) if u_durs else 1.0
    u_median = float(np.percentile(u_durs, 50)) if u_durs else 0.0
    u_p75 = float(np.percentile(u_durs, 75)) if u_durs else 0.0
    u_p90 = float(np.percentile(u_durs, 90)) if u_durs else 0.0

    return {
        "pct_v": round(pct_v, 2),
        "pct_c": round(pct_c, 2),
        "npvi_raw": round(npvi_raw, 2),
        "npvi_clipped": round(npvi_clipped, 2),
        "s_u_ratio_mean": round(s_mean / u_mean, 2) if u_mean > 0 else 0.0,
        "stressed_mean_ms": round(s_mean, 1),
        "schwa_mean_ms": round(u_mean, 1),
        "schwa_median_ms": round(u_median, 1),
        "schwa_p75_ms": round(u_p75, 1),
        "schwa_p90_ms": round(u_p90, 1),
    }


def main():
    if not os.path.exists(MANIFEST_PATH):
        print("Manifest file not found. Run generate_benchmarks.py first.")
        return

    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    records = []
    print(f"Processing alignment for {len(manifest)} items...")

    for item in manifest:
        wav_path = item["wav_path"]
        ref_text = item["reference_en"]
        fname = os.path.basename(wav_path)

        try:
            w_aligns = align_word_characters(wav_path, ref_text)
            phones = map_words_to_phonemes(w_aligns)
            m = compute_robust_metrics(phones)
            m["sentence_key"] = item["sentence_key"]
            m["cohort"] = item["cohort"]
            m["file"] = fname
            records.append(m)
            print(f"  [OK] [{item['sentence_key']}] {fname:<32} | %V: {m['pct_v']:>5.1f}% | P75: {m['schwa_p75_ms']:>5.1f}ms")
        except Exception as e:
            print(f"  [ERR] {fname}: {e}")

    df = pd.DataFrame(records)
    csv_path = "benchmarks/benchmark_results_all.csv"
    df.to_csv(csv_path, index=False)

    print("\n--- Summary per Sentence & Cohort ---")
    summary = df.groupby(["sentence_key", "cohort"])[["pct_v", "pct_c", "schwa_p75_ms"]].agg(["mean", "std"]).round(2)
    print(summary.to_string())


if __name__ == "__main__":
    main()