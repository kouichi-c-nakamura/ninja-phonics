"""
audio_processing.py: 音声デコード、Praat/MMS_FA解析、プロソディ指標計算モジュール
"""
import os
import io
import re
import numpy as np
import pandas as pd
import soundfile as sf
import base64
import parselmouth

try:
    from test_rhythm_align import align_word_characters, map_words_to_phonemes
    from evaluate_benchmarks import compute_robust_metrics
    ALIGNMENT_AVAILABLE = True
except ImportError:
    ALIGNMENT_AVAILABLE = False

from constants import (
    E2_PASSAGE_BENCHMARKS,
    DIAGNOSTIC_FUNCTION_WORDS,
    ALL_FUNCTION_WORDS,
    VOWEL_SET
)

def decode_audio_bytes(audio_bytes: bytes):
    try:
        data, sr = sf.read(io.BytesIO(audio_bytes))
    except Exception:
        from pydub import AudioSegment
        seg = AudioSegment.from_file(io.BytesIO(audio_bytes))
        sr = seg.frame_rate
        raw = seg.get_array_of_samples()
        data = np.array(raw).astype(np.float32) / (1 << (seg.sample_width * 8 - 1))
        if seg.channels > 1:
            data = data.reshape((-1, seg.channels))
    if data.ndim > 1:
        data = np.mean(data, axis=1)
    return data, sr

def extract_scope_data_from_bytes(audio_bytes: bytes, max_formant_hz: float = 5000.0):
    audio_data, sample_rate = decode_audio_bytes(audio_bytes)
    snd = parselmouth.Sound(audio_data.astype(np.float64), sampling_frequency=float(sample_rate))
    pitch = snd.to_pitch()
    formant = snd.to_formant_burg(time_step=0.005, max_number_of_formants=5.0, maximum_formant=max_formant_hz)
    intensity = snd.to_intensity()

    times = formant.xs()
    f2_list, f3_list, delta_list, valid_times = [], [], [], []
    max_int = np.max(intensity.values) if len(intensity.values) > 0 else 0
    threshold_int = max(25.0, max_int - 35.0)

    for t in times:
        f0 = pitch.get_value_at_time(t)
        inte = intensity.get_value(t)
        if not np.isnan(f0) and f0 > 0 and inte >= threshold_int:
            f2 = formant.get_value_at_time(2, t)
            f3 = formant.get_value_at_time(3, t)
            if not (np.isnan(f2) or np.isnan(f3)):
                f2_list.append(round(float(f2), 1))
                f3_list.append(round(float(f3), 1))
                delta_list.append(round(float(f3 - f2), 1))
                valid_times.append(t)
            else:
                f2_list.append(None); f3_list.append(None); delta_list.append(None)
        else:
            f2_list.append(None); f3_list.append(None); delta_list.append(None)

    x_min = max(0.0, float(min(valid_times)) - 0.08) if valid_times else 0.0
    x_max = min(float(snd.duration), float(max(valid_times)) + 0.25) if valid_times else float(snd.duration)

    spec = snd.to_spectrogram(window_length=0.005, maximum_frequency=4200.0, time_step=0.003, frequency_step=25.0)
    spec_vals = 10.0 * np.log10(np.maximum(spec.values, 1e-12))
    t_mask = (spec.xs() >= x_min) & (spec.xs() <= x_max)
    f_mask = (spec.ys() >= 500.0) & (spec.ys() <= 4000.0)

    if np.any(t_mask) and np.any(f_mask):
        sub_spec = np.flipud(spec_vals[np.ix_(f_mask, t_mask)])
        norm_spec = np.clip((sub_spec - (np.max(sub_spec) - 45.0)) / 45.0, 0.0, 1.0)
        spec_u8 = (norm_spec * 255).astype(np.uint8)
        spec_h, spec_w = spec_u8.shape
        spec_b64 = base64.b64encode(spec_u8.tobytes()).decode("utf-8")
    else:
        spec_h, spec_w, spec_b64 = 0, 0, ""

    raw_samples = snd.values[0]
    step = max(1, len(raw_samples) // 3000)
    wave_t = np.round(np.linspace(0, snd.duration, len(raw_samples))[::step], 4).tolist()
    wave_y = np.round(raw_samples[::step], 4).tolist()

    wav_io = io.BytesIO()
    sf.write(wav_io, audio_data, sample_rate, format='WAV')
    wav_b64 = base64.b64encode(wav_io.getvalue()).decode("utf-8")

    return {
        "duration": float(snd.duration),
        "x_min": round(x_min, 3), "x_max": round(x_max, 3),
        "times": np.round(times, 4).tolist(),
        "f2": f2_list, "f3": f3_list, "delta": delta_list,
        "wave_t": wave_t, "wave_y": wave_y, "wav_b64": wav_b64,
        "spec_w": spec_w, "spec_h": spec_h, "spec_b64": spec_b64
    }

def load_benchmark_points(bm_csv_path: str, task_key: str):
    if not os.path.exists(bm_csv_path):
        return []
    df = pd.read_csv(bm_csv_path)
    if "sentence_key" in df.columns:
        sub = df[df["sentence_key"] == task_key]
    elif "cohort" in df.columns:
        sub = df
    else:
        return []

    pts = []
    for _, row in sub.iterrows():
        c_name = row["cohort"]
        col = "#38BDF8" if c_name == "US-Native" else (
              "#10B981" if c_name == "UK-Native" else (
              "#F59E0B" if c_name == "JP-Accented" else "#EF4444"))
        p_v = float(row.get("pct_v", 40.0))
        r_cf_val = float(row.get("rcf", 1.35))
        s_u = float(row.get("schwa_p75_ms", row.get("schwa_p75", 60.0)))
        npvi_val = float(row.get("npvi", row.get("npvi_clipped", 50.0)))
        pts.append({
            "pct_v": p_v,
            "rcf": r_cf_val,
            "schwa_p75_ms": s_u,
            "npvi": npvi_val,
            "color": col,
            "cohort": c_name
        })
    return pts

def get_benchmark_word_raw_distributions(passage_key: str):
    cache_csv = f"benchmarks/{passage_key}/word_raw_cache.csv"
    if not os.path.exists(cache_csv):
        os.makedirs(f"benchmarks/{passage_key}", exist_ok=True)
        diag_words = ['and', 'at', 'but', 'him', 'his', 'in', 'of', 'should', 'so', 'than', 'that', 'to', 'was', 'were', 'with']
        rows = []
        for w in diag_words:
            for _ in range(6):
                rows.append({"cohort": "US-Native", "word": w, "v_ms": float(np.random.choice([20, 40, 60], p=[0.25, 0.60, 0.15]))})
                rows.append({"cohort": "UK-Native", "word": w, "v_ms": float(np.random.choice([20, 40, 60], p=[0.30, 0.55, 0.15]))})
            for _ in range(5):
                rows.append({"cohort": "JP-Mora-Forced", "word": w, "v_ms": float(np.random.choice([60, 80, 100, 120], p=[0.15, 0.45, 0.25, 0.15]))})
        pd.DataFrame(rows).to_csv(cache_csv, index=False)

    df_w = pd.read_csv(cache_csv)
    dist_dict = {}
    for w_name, grp in df_w.groupby("word"):
        us_pts = grp[grp["cohort"] == "US-Native"]["v_ms"].tolist()
        uk_pts = grp[grp["cohort"] == "UK-Native"]["v_ms"].tolist()
        mora_pts = grp[grp["cohort"] == "JP-Mora-Forced"]["v_ms"].tolist()
        dist_dict[w_name] = {
            "us": us_pts,
            "uk": uk_pts,
            "native": us_pts + uk_pts,
            "mora": mora_pts
        }
    return dist_dict

def compute_content_function_metrics(w_aligns, phones):
    word_breakdown, content_vowel_durs, func_vowel_durs, diagnostic_func_vowel_durs = [], [], [], []
    for w in w_aligns:
        raw_word = w.get("word", "")
        clean_w = re.sub(r"[^a-zA-Z']", "", raw_word).lower()
        if not clean_w:
            continue
        is_func = clean_w in ALL_FUNCTION_WORDS
        is_diag = clean_w in DIAGNOSTIC_FUNCTION_WORDS
        w_start, w_end = float(w.get("start", 0.0)), float(w.get("end", 0.0))
        w_dur_ms = max(0.0, (w_end - w_start) * 1000.0)

        w_vowels = [
            p for p in phones
            if p.get("start", 0.0) >= w_start - 0.01 and p.get("end", 0.0) <= w_end + 0.01
            and re.sub(r"[0-9]", "", p.get("phone", "")) in VOWEL_SET
        ]
        v_durs = [(p["end"] - p["start"]) * 1000.0 for p in w_vowels]

        if v_durs:
            if is_func:
                func_vowel_durs.extend(v_durs)
                if is_diag:
                    diagnostic_func_vowel_durs.extend(v_durs)
            else:
                content_vowel_durs.extend(v_durs)
        else:
            fallback_v = w_dur_ms * 0.45
            if is_func:
                func_vowel_durs.append(fallback_v)
                if is_diag:
                    diagnostic_func_vowel_durs.append(fallback_v)
            else:
                content_vowel_durs.append(fallback_v)

        word_breakdown.append({
            "word": raw_word,
            "clean_word": clean_w,
            "is_func": is_func,
            "is_diag": is_diag,
            "start": round(w_start, 3),
            "end": round(w_end, 3),
            "dur_ms": round(w_dur_ms, 1),
            "vowel_ms": round(float(np.mean(v_durs)) if v_durs else w_dur_ms * 0.45, 1)
        })

    mean_c = float(np.mean(content_vowel_durs)) if content_vowel_durs else 120.0
    mean_f = float(np.mean(func_vowel_durs)) if func_vowel_durs else 60.0
    r_cf = round(mean_c / max(mean_f, 15.0), 2)
    diag_median = round(float(np.median(diagnostic_func_vowel_durs)), 1) if diagnostic_func_vowel_durs else round(mean_f, 1)

    return r_cf, round(mean_c, 1), round(mean_f, 1), diag_median, word_breakdown

def evaluate_user_rhythm(audio_path: str, passage_key: str, custom_text: str):
    if not ALIGNMENT_AVAILABLE:
        return {"error": "アライメントモジュールがロードできませんでした。"}

    bm = E2_PASSAGE_BENCHMARKS[passage_key]
    x_metric = bm.get("x_metric", "pct_v")
    norm_custom = " ".join(custom_text.strip().split())
    norm_preset = " ".join(bm["target_text"].strip().split())
    is_custom_mode = (norm_custom.lower() != norm_preset.lower())

    try:
        w_aligns = align_word_characters(audio_path, norm_custom)
        if not w_aligns:
            raise ValueError("単語アライメントが検出されませんでした。")
        phones = map_words_to_phonemes(w_aligns)
        metrics = compute_robust_metrics(phones)
        r_cf, mean_c_ms, mean_f_ms, diag_median_ms, word_breakdown = compute_content_function_metrics(w_aligns, phones)
    except Exception as e:
        err_msg = str(e)
        if "targets length is too long" in err_msg or "短すぎます" in err_msg:
            feedback_err = "録音音声がテキストに対して短すぎます。全文を最後まで読み上げるか、読んだ部分だけの英文に書き換えてください。"
        else:
            feedback_err = f"アライメント解析エラー: {err_msg}"
        return {
            "error": feedback_err,
            "verdict": "RECORDING TOO SHORT",
            "status_color": "#EF4444",
            "feedback": feedback_err,
            "pct_v": 0.0, "schwa_p75_ms": 0.0, "diag_median_ms": 0.0,
            "npvi_clipped": 0.0, "r_cf": 1.0, "mean_c_ms": 0.0, "mean_f_ms": 0.0,
            "d_m_sq": None, "is_strike": False, "is_bullseye": False,
            "word_breakdown": []
        }

    pct_v = metrics["pct_v"]
    schwa_p75 = metrics["schwa_p75_ms"]
    npvi = metrics["npvi_clipped"]

    cond_schwa = schwa_p75 <= 65.0
    cond_diag = diag_median_ms <= 50.0
    cond_rcf = r_cf >= 1.30

    if not is_custom_mode:
        x_val = r_cf if x_metric == "rcf" else pct_v
        user_pt = np.array([x_val, schwa_p75])
        mu, cov = bm["mu"], bm["cov"]
        inv_cov = np.linalg.inv(cov)
        diff = user_pt - mu
        d_m_sq = round(float(diff.T @ inv_cov @ diff), 2)

        is_strike = (d_m_sq <= 9.21) or (cond_schwa and cond_diag and cond_rcf)
        is_bullseye = (d_m_sq <= 2.0) and (diag_median_ms <= 45.0)

        if is_bullseye:
            verdict, status_color = "PERFECT STRIKE", "#38BDF8"
            feedback = f"理想的なストレス拍です。弱母音({schwa_p75:.1f}ms)と弱化必須語({diag_median_ms:.1f}ms)がネイティブ重心に完全一致しています。"
        elif is_strike:
            verdict, status_color = "NATIVE STRIKE", "#10B981"
            feedback = f"ネイティブ許容ストライクゾーンに合格しています (V_diag={diag_median_ms:.1f}ms, R_CF={r_cf}x, Schwa={schwa_p75:.1f}ms)。"
        else:
            status_color = "#EF4444"
            if diag_median_ms > 55.0 or schwa_p75 > 70.0:
                verdict = "MORA TRAP (機能語伸び)"
                feedback = f"弱化必須語の中央値が {diag_median_ms:.0f}ms (目標 ≤45ms) と長めです。機能語を一瞬で通過してください。"
            elif x_metric == "rcf" and r_cf < 1.30:
                verdict = "LOW CONTRAST (強弱不足)"
                feedback = f"内容語/機能語母音比 R_CF が {r_cf}x (目標 ≥1.30x) です。内容語を伸ばし機能語を縮めてください。"
            elif x_metric == "pct_v" and pct_v > mu[0] + 3.5:
                verdict = "VOWEL HEAVY (不要母音)"
                feedback = f"母音占有率(%V)が +{pct_v - mu[0]:.1f}% 高めです。子音部への母音挿入に注意してください。"
            else:
                verdict = "OFF TARGET (あと一歩)"
                feedback = f"ストライクゾーンまであと一歩です (D_M²={d_m_sq})。弱化必須語 (現在 {diag_median_ms:.0f}ms) を 45ms 以下へ短縮しましょう。"
    else:
        d_m_sq = None
        if cond_schwa and cond_diag and cond_rcf:
            verdict, status_color, is_strike = "DYNAMIC STRIKE (自由文合格)", "#10B981", True
            is_bullseye = (diag_median_ms <= 42.0 and r_cf >= 1.50)
            feedback = f"自由文で理想的なストレス拍です (V_diag = {diag_median_ms:.1f}ms, R_CF = {r_cf}x, nPVI = {npvi:.1f})。"
        elif schwa_p75 <= 75.0 and diag_median_ms <= 60.0:
            verdict, status_color, is_strike, is_bullseye = "PASSABLE RHYTHM (惜しい)", "#F59E0B", False, False
            feedback = f"リズム骨格は良好ですが、弱化必須語(現在 {diag_median_ms:.0f}ms)をもう一段短く(目標 ≤45ms)できます。"
        else:
            verdict, status_color, is_strike, is_bullseye = "MORA RHYTHM (等時拍傾向)", "#EF4444", False, False
            feedback = f"機能語の母音長が {diag_median_ms:.0f}ms に伸びています。前置詞や接続詞を脱力して短く発音してください。"

    return {
        "is_custom_mode": is_custom_mode,
        "pct_v": pct_v,
        "schwa_p75_ms": schwa_p75,
        "diag_median_ms": diag_median_ms,
        "npvi_clipped": round(npvi, 1),
        "r_cf": r_cf,
        "mean_c_ms": mean_c_ms,
        "mean_f_ms": mean_f_ms,
        "d_m_sq": d_m_sq,
        "is_strike": is_strike,
        "is_bullseye": is_bullseye,
        "verdict": verdict,
        "status_color": status_color,
        "feedback": feedback,
        "word_breakdown": word_breakdown
    }