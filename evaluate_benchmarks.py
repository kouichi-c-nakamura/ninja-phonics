import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import os
import glob
import numpy as np
import pandas as pd

# 既存の test_rhythm_align.py から関数をインポート
from test_rhythm_align import (
    DEFAULT_SCRIPT,
    align_word_characters,
    map_words_to_phonemes
)


def compute_robust_metrics(phoneme_events: list):
    """ベンチマーク統計用にロバスト化したリズム・C/V指標を算出"""
    vowels = [e for e in phoneme_events if e["is_vowel"]]
    consonants = [e for e in phoneme_events if not e["is_vowel"]]

    sum_v = sum(e["dur_ms"] for e in vowels)
    sum_c = sum(e["dur_ms"] for e in consonants)
    total_ms = sum_v + sum_c

    pct_v = (sum_v / total_ms * 100.0) if total_ms > 0 else 0.0
    pct_c = 100.0 - pct_v

    # 1. 最小フレーム(20.1ms)による分母歪みを抑えるため 30ms で下限クリップした nPVI-V
    raw_durs = [e["dur_ms"] for e in vowels if e["dur_ms"] > 5.0]
    clipped_durs = [max(30.0, d) for d in raw_durs]
    m = len(clipped_durs)

    npvi_raw = 0.0
    npvi_clipped = 0.0
    if m > 1:
        npvi_raw = (100.0 / (m - 1)) * sum(
            abs(raw_durs[k] - raw_durs[k+1]) / ((raw_durs[k] + raw_durs[k+1]) / 2.0)
            for k in range(m - 1)
        )
        npvi_clipped = (100.0 / (m - 1)) * sum(
            abs(clipped_durs[k] - clipped_durs[k+1]) / ((clipped_durs[k] + clipped_durs[k+1]) / 2.0)
            for k in range(m - 1)
        )

    # 2. 強勢母音 (Stress 1,2) vs 弱勢母音 (Stress 0) のパーセンタイル分布
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
    wav_files = sorted(glob.glob("benchmarks/*/*.wav"))
    if not wav_files:
        print("benchmarks/ 内にWAVファイルが見つかりません。先に generate_benchmarks.py を実行してください。")
        return

    records = []
    print(f"📊 全 {len(wav_files)} ファイルのアライメントと音響計測を開始します...")

    for wav_path in wav_files:
        cohort = os.path.basename(os.path.dirname(wav_path))
        fname = os.path.basename(wav_path)
        try:
            # 強制カタカナ音声であっても、英語のお題スクリプト(DEFAULT_SCRIPT)に対してアライメントを取る
            w_aligns = align_word_characters(wav_path, DEFAULT_SCRIPT)
            phones = map_words_to_phonemes(w_aligns)
            m = compute_robust_metrics(phones)
            m["cohort"] = cohort
            m["file"] = fname
            records.append(m)
            print(f"  [OK] {fname:<38} | %V: {m['pct_v']:>5.1f}% | nPVI(clip): {m['npvi_clipped']:>5.1f} | Schwa P75: {m['schwa_p75_ms']:>5.1f}ms")
        except Exception as e:
            print(f"  [ERR] {fname}: {e}")

    df = pd.DataFrame(records)
    csv_path = "benchmarks/benchmark_results.csv"
    df.to_csv(csv_path, index=False)

    print("\n" + "=" * 85)
    print("📈 コホート別 統計サマリー (Mean ± SD)")
    print("=" * 85)
    target_cols = ["pct_v", "pct_c", "npvi_raw", "npvi_clipped", "s_u_ratio_mean", "schwa_median_ms", "schwa_p75_ms", "schwa_p90_ms"]
    summary = df.groupby("cohort")[target_cols].agg(["mean", "std"]).round(2)
    print(summary.to_string())
    print("=" * 85)
    print(f"\n💾 詳細データ保存先: {csv_path}")


if __name__ == "__main__":
    main()