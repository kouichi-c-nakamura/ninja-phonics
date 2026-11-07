import ssl
ssl._create_default_https_context = ssl._create_unverified_context

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

# 1. お題スクリプト（強勢母音とSchwaが交互に現れるテスト文）
DEFAULT_SCRIPT = "A photograph of a famous photographer was taken in the station."

# ARPABET 母音セット（数字を除いたベース名）
ARPABET_VOWELS = {
    "AA", "AE", "AH", "AO", "AW", "AY", "EH", "ER", "EY",
    "IH", "IY", "OW", "OY", "UH", "UW"
}

# MMS_FA モデルのロード（初回のみ自動ダウンロード）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
bundle = torchaudio.pipelines.MMS_FA
model = bundle.get_model(with_star=False).to(device)
model.eval()
LABELS = bundle.get_labels(star=None)
DICTIONARY = {c: i for i, c in enumerate(LABELS)}
g2p = G2p()


def align_word_characters(wav_path: str, script_text: str):
    """音声と英文スクリプトから、単語ごとの文字レベル・タイムスタンプを抽出"""
    audio_data, sr = sf.read(wav_path)
    if audio_data.ndim > 1:
        audio_data = np.mean(audio_data, axis=1)

    waveform = torch.from_numpy(audio_data).float().unsqueeze(0)
    if sr != bundle.sample_rate:
        waveform = torchaudio.functional.resample(waveform, sr, bundle.sample_rate)

    # 単語リストの正規化（英字のみ小文字化）
    raw_words = re.findall(r"[A-Za-z]+", script_text)
    clean_words = [w.lower() for w in raw_words]

    # MMS_FA 用のトークン列作成
    transcript_chars = "".join(clean_words)
    tokens = [DICTIONARY[c] for c in transcript_chars if c in DICTIONARY]

    with torch.inference_mode():
        emission, _ = model(waveform.to(device))
        targets = torch.tensor([tokens], dtype=torch.int32, device=device)
        alignments, scores = torchaudio.functional.forced_align(emission, targets, blank=0)
        token_spans = torchaudio.functional.merge_tokens(alignments[0], scores[0])

    # フレームインデックスから秒数への変換係数
    num_frames = emission.size(1)
    audio_dur = waveform.size(1) / bundle.sample_rate
    frame_dur = audio_dur / num_frames

    # 単語ごとに文字スパンを束ねる
    word_alignments = []
    span_idx = 0
    for orig_w, clean_w in zip(raw_words, clean_words):
        n_chars = len(clean_w)
        w_spans = token_spans[span_idx : span_idx + n_chars]
        span_idx += n_chars
        if not w_spans:
            continue

        w_start = w_spans[0].start * frame_dur
        w_end = w_spans[-1].end * frame_dur
        char_boxes = [
            {
                "char": clean_w[i],
                "start": s.start * frame_dur,
                "end": s.end * frame_dur,
                "dur": (s.end - s.start) * frame_dur
            }
            for i, s in enumerate(w_spans)
        ]
        word_alignments.append({
            "word": orig_w,
            "start": w_start,
            "end": w_end,
            "chars": char_boxes
        })

    return word_alignments


def map_words_to_phonemes(word_alignments: list):
    """単語内の文字タイムスタンプをARPABET音素（母音V / 子音C / 強勢有無）へ比例配分マッピング"""
    phoneme_events = []

    for w_info in word_alignments:
        word = w_info["word"]
        chars = w_info["chars"]
        if not chars:
            continue

        # G2Pで音素列を取得 (例: photograph -> ['F', 'OW1', 'T', 'AH0', 'G', 'R', 'AE2', 'F'])
        phones = [p for p in g2p(word) if re.match(r"^[A-Z]+[0-2]?$", p)]
        if not phones:
            continue

        # 単語内の文字時間軸を音素数に応じてマッピング
        # （各文字の持続時間重みを保持したまま音素へ割り当て）
        char_Boundaries = [c["start"] for c in chars] + [chars[-1]["end"]]
        total_chars = len(chars)
        total_phones = len(phones)

        for p_idx, p in enumerate(phones):
            # 音素の相対位置から対応する文字境界を線形補間
            c_pos_start = (p_idx / total_phones) * total_chars
            c_pos_end = ((p_idx + 1) / total_phones) * total_chars

            def interp_time(pos):
                idx = int(np.floor(pos))
                frac = pos - idx
                if idx >= total_chars:
                    return char_Boundaries[-1]
                return char_Boundaries[idx] + frac * (char_Boundaries[idx + 1] - char_Boundaries[idx])

            p_start = interp_time(c_pos_start)
            p_end = interp_time(c_pos_end)
            p_dur_ms = (p_end - p_start) * 1000.0

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

    # nPVI-V (隣り合う母音の長さの変動係数)
    v_durs = [e["dur_ms"] for e in vowels if e["dur_ms"] > 5.0]
    m = len(v_durs)
    npvi_v = 0.0
    if m > 1:
        diffs = [
            abs(v_durs[k] - v_durs[k + 1]) / ((v_durs[k] + v_durs[k + 1]) / 2.0)
            for k in range(m - 1)
        ]
        npvi_v = (100.0 / (m - 1)) * sum(diffs)

    # 強勢母音 (stress 1, 2) vs 弱勢母音 (stress 0) の平均持続時間比
    stressed_v = [e for e in vowels if e["stress"] in (1, 2)]
    unstressed_v = [e for e in vowels if e["stress"] == 0]

    mean_s = np.mean([e["dur_ms"] for e in stressed_v]) if stressed_v else 0.0
    mean_u = np.mean([e["dur_ms"] for e in unstressed_v]) if unstressed_v else 1.0
    s_u_ratio = mean_s / mean_u if mean_u > 0 else 0.0

    # ワースト弱母音（本来短く潰すべき Stress=0 の母音で長すぎるもの Top 3）
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


def print_report(metrics: dict, script_text: str):
    print("\n" + "=" * 68)
    print(f"🎯 お題: {script_text}")
    print("=" * 68)

    # 1. タイムライン・ビートバー (ASCII可視化)
    print("\n【1. タイムライン・ビートバー (母音持続時間と強弱コントラスト)】")
    for v in metrics["vowels"]:
        bar_len = int(v["dur_ms"] / 8.0)
        if v["stress"] in (1, 2):
            marker = "🟧 [強]"
            bar = "█" * max(1, bar_len)
        else:
            marker = "🟦 [弱]"
            bar = "▒" * max(1, bar_len)
        print(f"  {v['word']:<13} | {v['phone']:<4} {marker} | {v['dur_ms']:>5.1f} ms | {bar}")

    # 2. 2大リズム＆C/V骨格スコア
    print("\n【2. ストライクゾーン判定メトリクス】")
    npvi_status = "✅ STRIKE (英語ストレス拍)" if metrics["npvi_v"] >= 58.0 else "❌ MORA TRAP (平坦・日本語拍)"
    pct_v_status = "✅ STRIKE (子音骨格◎)" if metrics["pct_v"] <= 46.0 else "⚠️ VOWEL HEAVY (母音過多・子音不足)"

    print(f"  ・リズム起伏 (nPVI-V)   : {metrics['npvi_v']:>5.1f}  (目標: >= 60.0) -> {npvi_status}")
    print(f"  ・母音占有率 (%V)       : {metrics['pct_v']:>5.1f}% (目標: 38~45%)  -> {pct_v_status}")
    print(f"  ・子音ホールド率 (%C)   : {metrics['pct_c']:>5.1f}% (目標: >= 55%)")
    print(f"  ・平均 強/弱母音長比    : {metrics['s_u_ratio']:>5.2f} : 1 (強 {metrics['mean_stressed_ms']}ms / 弱 {metrics['mean_unstressed_ms']}ms)")

    # 3. ワースト弱母音処方箋
    print("\n【3. 肥大化した曖昧母音 (Schwa) ワースト3 処方箋】")
    for idx, w in enumerate(metrics["worst_schwas"], 1):
        print(f"  {idx}位: 単語 '{w['word']}' の弱母音 [{w['phone']}] ({w['start']}s~) が {w['dur_ms']} ms 鳴っています。")
        print(f"       ➔ 処方箋: 母音を響かせず、息を抜いて 40~50ms 以内で次の子音へ飛び移ってください。")
    print("=" * 68 + "\n")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("使い方: python test_rhythm_align.py <音声ファイル.wav> [任意のスクリプト文字列]")
        print(f"デフォルトのお題: \"{DEFAULT_SCRIPT}\"")
        sys.exit(1)

    wav_file = sys.argv[1]
    script = sys.argv[2] if len(sys.argv) >= 3 else DEFAULT_SCRIPT

    word_aligns = align_word_characters(wav_file, script)
    phonemes = map_words_to_phonemes(word_aligns)
    res = compute_rhythm_metrics(phonemes)
    print_report(res, script)