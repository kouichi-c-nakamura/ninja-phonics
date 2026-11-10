import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import os
import sys
import re
import numpy as np
import soundfile as sf
import torch
import torchaudio
from g2p_en import G2p

# 初回のみNLTKリソースが必要な場合のフォールバック
import nltk
try:
    nltk.data.find('taggers/averaged_perceptron_tagger_eng')
except LookupError:
    nltk.download('averaged_perceptron_tagger_eng', quiet=True)
    nltk.download('cmudict', quiet=True)

# お題スクリプト
DEFAULT_SCRIPT = "A photograph of a famous photographer was taken in the station."

# ARPABET 母音セット（数字を除いたベース名）
ARPABET_VOWELS = {
    "AA", "AE", "AH", "AO", "AW", "AY", "EH", "ER", "EY",
    "IH", "IY", "OW", "OY", "UH", "UW"
}

# MMS_FA モデルのロード
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
bundle = torchaudio.pipelines.MMS_FA
model = bundle.get_model(with_star=False).to(device)
model.eval()
LABELS = bundle.get_labels(star=None)
DICTIONARY = {c: i for i, c in enumerate(LABELS)}
g2p = G2p()


def align_word_characters(audio_path: str, text: str):
    """
    音声ファイルとテキストを強制アライメントし、
    map_words_to_phonemes が要求する 'chars' フィールド付きの単語情報を返す。
    """
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    # 1. soundfile で確実に音声をロード (TorchCodec エラーを回避)
    audio_np, sr = sf.read(audio_path, dtype="float32")
    if audio_np.ndim > 1:
        audio_np = np.mean(audio_np, axis=1)
    waveform = torch.from_numpy(audio_np).unsqueeze(0)

    if sr != bundle.sample_rate:
        waveform = torchaudio.functional.resample(waveform, sr, bundle.sample_rate)

    waveform = waveform.to(device)

    # 2. 音響モデルで emission 抽出
    with torch.inference_mode():
        emission, _ = model(waveform)

    num_frames = emission.size(1)

    # 3. テキストのトークン化と CTC 長セーフガード
    raw_words = text.strip().split()
    if not raw_words:
        return []

    accum_tokens = []
    word_token_info = []

    for raw_w in raw_words:
        clean_w = re.sub(r"[^a-zA-Z']", "", raw_w).lower()
        if not clean_w:
            continue

        w_toks = []
        for ch in clean_w:
            if ch in DICTIONARY:
                w_toks.append(DICTIONARY[ch])
            elif ch == "'" and "'" in DICTIONARY:
                w_toks.append(DICTIONARY["'"])

        if not w_toks:
            continue

        repeats = sum(1 for i in range(1, len(w_toks)) if w_toks[i] == w_toks[i - 1])
        needed_frames = len(w_toks) + repeats + 2

        if len(accum_tokens) + needed_frames >= (num_frames - 12):
            break

        start_t_idx = len(accum_tokens)
        accum_tokens.extend(w_toks)
        end_t_idx = len(accum_tokens)

        word_token_info.append({
            "word": raw_w,
            "clean_word": clean_w,
            "tokens": w_toks,
            "start_idx": start_t_idx,
            "end_idx": end_t_idx
        })

    if not word_token_info or not accum_tokens:
        duration_sec = num_frames * 0.02
        raise ValueError(
            f"録音音声が短すぎます（実測 約{duration_sec:.2f}秒）。単語を発声し終えるまで録音してください。"
        )

    targets = torch.tensor([accum_tokens], dtype=torch.int32, device=device)

    # 4. forced_align 実行
    alignments, scores = torchaudio.functional.forced_align(emission, targets, blank=0)
    frame_indices = alignments[0].cpu().tolist()

    # 5. 各文字（トークン）の出現開始・終了フレームを抽出
    token_spans = []
    curr_tok = None
    curr_start = None

    for f_idx, tok_id in enumerate(frame_indices):
        if tok_id != curr_tok:
            if curr_tok is not None and curr_tok > 0:
                token_spans.append({"token_id": curr_tok, "start_frame": curr_start, "end_frame": f_idx})
            curr_tok = tok_id
            curr_start = f_idx

    if curr_tok is not None and curr_tok > 0:
        token_spans.append({"token_id": curr_tok, "start_frame": curr_start, "end_frame": len(frame_indices)})

    frame_dur = 1.0 / bundle.sample_rate * 320  # 約0.02s

    # 6. map_words_to_phonemes 用の chars リストを組み立て
    word_alignments = []
    tok_cursor = 0

    for winfo in word_token_info:
        w_len = len(winfo["tokens"])
        w_spans = token_spans[tok_cursor : tok_cursor + w_len]
        tok_cursor += w_len

        chars_list = []
        if w_spans:
            for s in w_spans:
                c_start = s["start_frame"] * frame_dur
                c_end = s["end_frame"] * frame_dur
                chars_list.append({"start": round(float(c_start), 3), "end": round(float(c_end), 3)})
            w_start = chars_list[0]["start"]
            w_end = chars_list[-1]["end"]
        else:
            w_start = 0.0
            w_end = 0.1
            chars_list = [{"start": w_start, "end": w_end}]

        word_alignments.append({
            "word": winfo["word"],
            "clean_word": winfo["clean_word"],
            "start": w_start,
            "end": w_end,
            "duration": round(w_end - w_start, 3),
            "chars": chars_list
        })

    return word_alignments


def map_words_to_phonemes(word_alignments: list):
    """単語内の文字タイムスタンプをARPABET音素（母音V / 子音C / 強勢有無）へ比例配分マッピング"""
    phoneme_events = []

    for w_info in word_alignments:
        word = w_info["word"]
        chars = w_info.get("chars", [])
        if not chars:
            continue

        phones = [p for p in g2p(word) if re.match(r"^[A-Z]+[0-2]?$", p)]
        if not phones:
            continue

        char_boundaries = [c["start"] for c in chars] + [chars[-1]["end"]]
        total_chars = len(chars)
        total_phones = len(phones)

        for p_idx, p in enumerate(phones):
            c_pos_start = (p_idx / total_phones) * total_chars
            c_pos_end = ((p_idx + 1) / total_phones) * total_chars

            def interp_time(pos):
                idx = int(np.floor(pos))
                frac = pos - idx
                if idx >= total_chars:
                    return char_boundaries[-1]
                return char_boundaries[idx] + frac * (char_boundaries[idx + 1] - char_boundaries[idx])

            p_start = interp_time(c_pos_start)
            p_end = interp_time(c_pos_end)
            p_dur_ms = max(5.0, (p_end - p_start) * 1000.0)

            base_p = p.rstrip("012")
            stress = p[-1] if p[-1] in "012" else None
            is_vowel = base_p in ARPABET_VOWELS

            phoneme_events.append({
                "word": word,
                "phone": p,
                "base": base_p,
                "is_vowel": is_vowel,
                "stress": int(stress) if stress is not None else -1,
                "start": round(p_start, 3),
                "end": round(p_end, 3),
                "dur_ms": round(p_dur_ms, 1)
            })

    return phoneme_events


def compute_rhythm_metrics(phoneme_events: list):
    """%V, %C, nPVI-V, 強弱母音比を算出"""
    vowels = [e for e in phoneme_events if e["is_vowel"]]
    consonants = [e for e in phoneme_events if not e["is_vowel"]]

    sum_v = sum(e["dur_ms"] for e in vowels)
    sum_c = sum(e["dur_ms"] for e in consonants)
    total_speech_ms = sum_v + sum_c

    pct_v = (sum_v / total_speech_ms * 100.0) if total_speech_ms > 0 else 0.0
    pct_c = 100.0 - pct_v

    v_durs = [e["dur_ms"] for e in vowels if e["dur_ms"] > 5.0]
    m = len(v_durs)
    npvi_v = 0.0
    if m > 1:
        diffs = [
            abs(v_durs[k] - v_durs[k + 1]) / ((v_durs[k] + v_durs[k + 1]) / 2.0)
            for k in range(m - 1)
        ]
        npvi_v = (100.0 / (m - 1)) * sum(diffs)

    stressed_v = [e for e in vowels if e["stress"] in (1, 2)]
    unstressed_v = [e for e in vowels if e["stress"] == 0]

    mean_s = np.mean([e["dur_ms"] for e in stressed_v]) if stressed_v else 0.0
    mean_u = np.mean([e["dur_ms"] for e in unstressed_v]) if unstressed_v else 1.0
    s_u_ratio = mean_s / mean_u if mean_u > 0 else 0.0

    worst_schwas = sorted(unstressed_v, key=lambda x: x["dur_ms"], reverse=True)[:3]

    return {
        "pct_v": round(pct_v, 1),
        "pct_c": round(pct_c, 1),
        "npvi_v": round(npvi_v, 1),
        "mean_stressed_ms": round(float(mean_s), 1),
        "mean_unstressed_ms": round(float(mean_u), 1),
        "s_u_ratio": round(float(s_u_ratio), 2),
        "vowels": vowels,
        "worst_schwas": worst_schwas
    }