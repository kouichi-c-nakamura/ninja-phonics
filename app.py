import streamlit as st
import streamlit.components.v1 as components
import parselmouth
import soundfile as sf
import numpy as np
import pandas as pd
import base64
import json
import io
import os
import re

# MMS_FA アライメントエンジンのインポート（E2用）
try:
    from test_rhythm_align import align_word_characters, map_words_to_phonemes
    from evaluate_benchmarks import compute_robust_metrics
    ALIGNMENT_AVAILABLE = True
except ImportError:
    ALIGNMENT_AVAILABLE = False

st.set_page_config(page_title="NinjaPhonetics Scope", layout="wide", initial_sidebar_state="collapsed")

# 画面パディングとスタイル調整（アクティブボタンのブルーティント統一）
st.markdown("""
<style>
    header[data-testid="stHeader"] {
        height: 2.5rem !important;
        background: transparent !important;
    }
    .block-container {
        padding-top: 3.2rem !important;
        padding-bottom: 0.5rem !important;
        padding-left: 1.5rem !important;
        padding-right: 1.5rem !important;
    }
    h1 {
        font-size: 1.45rem !important;
        font-weight: 700 !important;
        margin: 0 !important;
        color: #F8FAFC !important;
        line-height: 2.2rem !important;
    }
    div[data-testid="stAudioInput"] {
        background: #1E293B;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 6px 14px;
        margin-bottom: 0.4rem;
    }
    div[data-testid="stAudioInput"] button {
        transform: scale(1.1);
    }
    .stButton button {
        background-color: #1E293B !important;
        color: #E2E8F0 !important;
        border: 1px solid #475569 !important;
        font-weight: bold !important;
        padding: 6px 8px !important;
        border-radius: 6px !important;
        font-size: 0.9rem !important;
    }
    .stButton button:hover {
        background-color: #334155 !important;
        border-color: #64748B !important;
        color: #FFFFFF !important;
    }
    .stButton button[kind="primary"],
    .stButton button[data-testid="baseButton-primary"] {
        background-color: #1E3A8A !important;
        border-color: #3B82F6 !important;
        color: #93C5FD !important;
        box-shadow: 0 0 0 1px #3B82F6 !important;
    }
    .stButton button[kind="primary"]:hover,
    .stButton button[data-testid="baseButton-primary"]:hover {
        background-color: #1E40AF !important;
        border-color: #60A5FA !important;
        color: #BFDBFE !important;
    }
    div[data-testid="stTextArea"] textarea {
        background-color: #0F172A !important;
        color: #F8FAFC !important;
        border: 1px solid #334155 !important;
        font-size: 1.02rem !important;
        font-weight: 600 !important;
        border-radius: 8px !important;
        line-height: 1.5 !important;
    }
    div[data-testid="stTextArea"] textarea:focus {
        border-color: #38BDF8 !important;
        box-shadow: 0 0 0 1px #38BDF8 !important;
    }
</style>
""", unsafe_allow_html=True)

# =====================================================================
# MODE 2 定数 & 辞書定義
# =====================================================================
E2_PASSAGE_BENCHMARKS = {
    # --- 短文フォーカス・ドリル (S1〜S3) ---
    "S1_stress": {
        "category": "drill",
        "title": "短文ドリル 1: Alternating Stress & Reduction (語彙強弱拍)",
        "target_text": "A photograph of a famous photographer was taken in the station.",
        "phonetics_desc": "photograph /ˈfoʊ.tə.ɡræf/ と photographer /fəˈtɑː.ɡrə.fɚ/ の強弱母音コントラストを測定。弱母音(/ə/)の鋭い短縮が不可欠です。",
        "mu": np.array([47.01, 58.61]),
        "cov": np.array([[1.045, 3.120], [3.120, 39.810]]),
        "x_range": (42.0, 58.0),
        "y_range": (35.0, 110.0),
        "ref_audio": "benchmarks/S1_stress/UK-Native/UK-Native_01_ArcherConver_male.wav",
        "bm_csv": "benchmarks/benchmark_results_all.csv"
    },
    "S2_clusters": {
        "category": "drill",
        "title": "短文ドリル 2: Consonant Clusters & Coda Holding (子音連結保持)",
        "target_text": "Strict instructors asked complex questions about the project tests.",
        "phonetics_desc": "/str/, /mpl/, /sts/ などの連続子音に余計な母音([ɯ], [o])を挟まずに子音を保持できるかを診断。ネイティブは %V が 36% 付近まで急激に低下します。",
        "mu": np.array([35.85, 59.12]),
        "cov": np.array([[1.380, 2.850], [2.850, 48.200]]),
        "x_range": (30.0, 46.0),
        "y_range": (35.0, 120.0),
        "ref_audio": "benchmarks/S2_clusters/US-Native/US-Native_01_RogerLaidBac_male.wav",
        "bm_csv": "benchmarks/benchmark_results_all.csv"
    },
    "S3_connected": {
        "category": "drill",
        "title": "短文ドリル 3: Connected Speech & Function Word Weakening (機能語弱化)",
        "target_text": "Can you give him a cup of water from the kitchen at lunch?",
        "phonetics_desc": "can, him, a, of, from, at などの機能語を極小化し、息の流れに乗せるテスト。全単語を等時拍で読むと Schwa P75 が 80ms を突破します。",
        "mu": np.array([45.42, 45.18]),
        "cov": np.array([[1.150, 1.950], [1.950, 36.400]]),
        "x_range": (38.0, 54.0),
        "y_range": (30.0, 140.0),
        "ref_audio": "benchmarks/S3_connected/UK-Native/UK-Native_01_ArcherConver_male.wav",
        "bm_csv": "benchmarks/benchmark_results_all.csv"
    },
    # --- 総合アセスメント長文 (P1: The North Wind and the Sun 全文113語) ---
    "P1_northwind": {
        "category": "passage",
        "title": "総合アセスメント 1: The North Wind and the Sun (北風と太陽・全文113語)",
        "target_text": "The North Wind and the Sun were disputing which was the stronger, when a traveler came along wrapped in a warm cloak. They agreed that the one who first succeeded in making the traveler take his cloak off should be considered stronger than the other. Then the North Wind blew with all his might, but the more he blew, the more closely did the traveler fold his cloak around him, and at last the North Wind gave up the attempt. Then the Sun shone out warmly, and immediately the traveler took off his cloak. And so the North Wind was obliged to confess that the Sun was the stronger of the two.",
        "phonetics_desc": "国際音声学会(IPA)正準テキスト。前置詞(with, at, to, of)、接続詞(and, that, than, but)、助動詞(was, were, should)の完全な弱化とストレス拍を総合診断します。",
        "mu": np.array([39.94, 58.03]),
        "cov": np.array([[0.835, 4.218], [4.218, 24.163]]),
        "x_range": (31.0, 50.0),
        "y_range": (33.0, 115.0),
        "ref_audio": "benchmarks/P1_northwind/UK-Native/UK-Native_01_Archer.wav",
        "bm_csv": "benchmarks/P1_northwind/northwind_benchmark_results.csv"
    }
}

# 冠詞(the, a, an)を除外した、真の判別リトマス機能語リスト
DIAGNOSTIC_FUNCTION_WORDS = {
    # 前置詞 (Prepositions: 助走小粒化)
    "to", "for", "at", "of", "in", "from", "with", "into", "through", "about",
    # 接続詞 (Conjunctions: カタカナ化の最大トラップ)
    "and", "that", "than", "but", "as", "if", "or", "so",
    # 助動詞・代名詞 (Auxiliaries & Pronouns)
    "was", "were", "should", "could", "would", "can", "his", "him", "them", "us"
}

# 一般の機能語全体 (R_CF 計算およびバッジ表示用)
ALL_FUNCTION_WORDS = {
    "a", "an", "the",
    "in", "on", "at", "to", "for", "of", "from", "with", "by", "about", "as", "into", "like", "through", "after", "over", "between", "out", "against", "during", "without", "before", "under", "around", "among",
    "i", "you", "he", "she", "it", "we", "they", "me", "him", "her", "us", "them", "my", "your", "his", "its", "our", "their", "mine", "yours", "hers", "ours", "theirs", "myself", "yourself", "himself", "herself", "itself", "ourselves", "themselves",
    "this", "that", "these", "those", "who", "whom", "whose", "which", "what", "where", "when", "why", "how",
    "am", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "having", "do", "does", "did",
    "will", "would", "shall", "should", "can", "could", "may", "might", "must",
    "and", "but", "or", "nor", "so", "yet", "if", "because", "although", "though", "while", "since", "unless", "until", "whether",
    "there", "here", "some", "any", "no", "not", "too", "very", "just", "than", "off", "along"
}

VOWEL_SET = {"AA", "AE", "AH", "AO", "AW", "AY", "EH", "ER", "EY", "IH", "IY", "OW", "OY", "UH", "UW"}

# --- セッション状態の初期化 ---
if "app_mode" not in st.session_state:
    st.session_state.app_mode = "RL"
if "current_label" not in st.session_state:
    st.session_state.current_label = "お手本: Right (US)"
if "current_audio_bytes" not in st.session_state:
    init_path = os.path.join("presets", "right_us.wav")
    if os.path.exists(init_path):
        with open(init_path, "rb") as f:
            st.session_state.current_audio_bytes = f.read()
    else:
        st.session_state.current_audio_bytes = None
if "last_recorded_bytes" not in st.session_state:
    st.session_state.last_recorded_bytes = None
if "auto_play" not in st.session_state:
    st.session_state.auto_play = False

# E2 セッション状態
if "e2_category" not in st.session_state:
    st.session_state.e2_category = "drill"
if "e2_selected_passage" not in st.session_state:
    st.session_state.e2_selected_passage = "S1_stress"
if "e2_custom_text" not in st.session_state:
    st.session_state.e2_custom_text = E2_PASSAGE_BENCHMARKS["S1_stress"]["target_text"]
if "e2_last_analyzed_text" not in st.session_state:
    st.session_state.e2_last_analyzed_text = st.session_state.e2_custom_text
if "e2_audio_bytes" not in st.session_state:
    st.session_state.e2_audio_bytes = None
if "e2_eval_history" not in st.session_state:
    st.session_state.e2_eval_history = []

# --- 最上段ヘッダー & モード切替ボタン ---
head_col1, head_col2, head_col3 = st.columns([2.0, 1.2, 1.2], vertical_alignment="center")
with head_col1:
    st.title("NinjaPhonetics: Biofeedback Scope")
with head_col2:
    if st.button(
        "A1: R / L スコープ",
        type="primary" if st.session_state.app_mode == "RL" else "secondary",
        use_container_width=True
    ):
        st.session_state.app_mode = "RL"
        st.rerun()
with head_col3:
    if st.button(
        "E2: 3大プロソディ診断",
        type="primary" if st.session_state.app_mode == "RHYTHM" else "secondary",
        use_container_width=True
    ):
        st.session_state.app_mode = "RHYTHM"
        st.rerun()


# =====================================================================
# 共通ユーティリティ: 音声バイト列のデコード
# =====================================================================
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


# =====================================================================
# MODE 1: R / L ホルマントスコープ (A1)
# =====================================================================
PRESET_DIR = "presets"
PRESETS = [
    {"label": "Right (US)", "file": "right_us.wav"},
    {"label": "Right (UK)", "file": "right_uk.wav"},
    {"label": "Light (US)", "file": "light_us.wav"},
    {"label": "Light (UK)", "file": "light_uk.wav"},
    {"label": "White (UK)", "file": "white_uk.wav"},
    {"label": "ライト (JP)", "file": "raito_jp.wav"},
    {"label": "Really (US)", "file": "really_us.wav"},
    {"label": "Really (UK)", "file": "really_uk.wav"},
    {"label": "Very (US)", "file": "very_us.wav"},
    {"label": "Very (UK)", "file": "very_uk.wav"}
]

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

def build_spike2_html(data: dict, auto_play: bool = False, export_name: str = "scope") -> str:
    data["auto_play"] = auto_play
    data["export_name"] = export_name
    data_json = json.dumps(data)
    return f"""
<!DOCTYPE html>
<html>
<head>
<style>
  body {{ margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: #0B0F19; color: #E2E8F0; user-select: none; overflow: hidden; }}
  .toolbar {{ display: flex; align-items: center; gap: 10px; padding: 10px 14px; background: #0F172A; border: 1px solid #1E293B; border-radius: 8px 8px 0 0; }}
  button {{ font-family: inherit; font-size: 14px; font-weight: bold; padding: 8px 14px; border-radius: 6px; border: 1px solid #334155; cursor: pointer; background: #1E293B; color: #E2E8F0; transition: all 0.15s; white-space: nowrap; }}
  button:hover {{ background: #334155; color: #FFFFFF; }}
  button.btn-play-ready {{ background: #16A34A; color: #FFFFFF; border-color: #15803D; }}
  button.btn-play-ready:hover {{ background: #15803D; }}
  button.btn-play-active {{ background: #DC2626; color: #FFFFFF; border-color: #B91C1C; }}
  button.toggle-active {{ background: #334155; color: #F8FAFC; border-color: #64748B; }}
  select {{ font-family: inherit; font-size: 14px; padding: 6px 10px; border-radius: 6px; border: 1px solid #334155; background: #1E293B; color: #E2E8F0; font-weight: bold; }}
  .readout {{ font-family: inherit; font-size: 14px; font-weight: 500; margin-left: auto; font-variant-numeric: tabular-nums; color: #E2E8F0; white-space: nowrap; }}
  #scope-wrapper {{ position: relative; width: 100%; height: 580px; border: 1px solid #1E293B; border-top: none; background: #090D16; }}
  canvas {{ position: absolute; left: 0; top: 0; width: 100%; height: 100%; }}
  #overlay-canvas {{ mix-blend-mode: screen; }}
</style>
</head>
<body>
<div class="toolbar">
  <button id="btn-play" class="btn-play-ready">▶ Play (Space)</button>
  <button id="btn-reset">⏮ 全体選択</button>
  <button id="btn-spec" class="toggle-active">Spectrogram: ON</button>
  <label style="font-size: 14px; font-weight: 500; display: flex; align-items: center; gap: 6px; color: #94A3B8;">
    Speed:
    <select id="sel-rate"><option value="1.0">1.0x</option><option value="0.7">0.7x</option><option value="0.5">0.5x</option></select>
  </label>
  <button id="btn-dl-plot">💾 Plot (.png)</button>
  <button id="btn-dl-audio">🔊 Audio (.wav)</button>
  <div class="readout" id="time-readout">Cursor: 0.000 s</div>
</div>
<div id="scope-wrapper">
  <canvas id="bg-canvas"></canvas>
  <canvas id="overlay-canvas"></canvas>
</div>
<script>
const D = {data_json};
const wrapper = document.getElementById('scope-wrapper');
const bgCanvas = document.getElementById('bg-canvas');
const ovCanvas = document.getElementById('overlay-canvas');
const bgCtx = bgCanvas.getContext('2d');
const ovCtx = ovCanvas.getContext('2d');

const btnPlay = document.getElementById('btn-play');
const btnReset = document.getElementById('btn-reset');
const btnSpec = document.getElementById('btn-spec');
const btnDlPlot = document.getElementById('btn-dl-plot');
const btnDlAudio = document.getElementById('btn-dl-audio');
const selRate = document.getElementById('sel-rate');
const readout = document.getElementById('time-readout');

const MARGIN = {{ left: 110, right: 30, top: 20, bottom: 45 }};
let W = 1000, H = 580;
let xMin = D.x_min, xMax = D.x_max;
let selStart = xMin, selEnd = xMax;
let cursorT = (xMin + xMax) / 2;
let showSpec = true;

let specCanvas = null;
if (D.spec_w > 0 && D.spec_h > 0 && D.spec_b64) {{
  specCanvas = document.createElement('canvas');
  specCanvas.width = D.spec_w; specCanvas.height = D.spec_h;
  const sCtx = specCanvas.getContext('2d');
  const imgData = sCtx.createImageData(D.spec_w, D.spec_h);
  const raw = atob(D.spec_b64);
  for (let i = 0; i < raw.length; i++) {{
    const val = raw.charCodeAt(i);
    const p = i * 4;
    const gray = Math.min(255, 12 + val * 0.85);
    imgData.data[p] = gray; imgData.data[p+1] = gray; imgData.data[p+2] = gray; imgData.data[p+3] = 255;
  }}
  sCtx.putImageData(imgData, 0, 0);
}}

let audioEl = new Audio();
let audioBlobUrl = null;
let isPlaying = false;
let animId = null;

function initAudio() {{
  const binary = atob(D.wav_b64);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
  const blob = new Blob([bytes], {{ type: "audio/wav" }});
  audioBlobUrl = URL.createObjectURL(blob);
  audioEl.src = audioBlobUrl;
  audioEl.preservesPitch = true; audioEl.mozPreservesPitch = true; audioEl.webkitPreservesPitch = true;
}}
initAudio();

function tToX(t) {{ return MARGIN.left + ((t - xMin) / (xMax - xMin)) * (W - MARGIN.left - MARGIN.right); }}
function xToT(x) {{ return Math.max(xMin, Math.min(xMax, xMin + ((x - MARGIN.left) / (W - MARGIN.left - MARGIN.right)) * (xMax - xMin))); }}

function getPanels() {{
  const totalH = H - MARGIN.top - MARGIN.bottom;
  const gap = 24, h1 = 120;
  const remainH = totalH - h1 - (gap * 2);
  const h2 = remainH * 0.50, h3 = remainH * 0.50;
  return [
    {{ top: MARGIN.top, bottom: MARGIN.top + h1, height: h1 }},
    {{ top: MARGIN.top + h1 + gap, bottom: MARGIN.top + h1 + gap + h2, height: h2 }},
    {{ top: MARGIN.top + h1 + gap + h2 + gap, bottom: MARGIN.top + h1 + gap + h2 + gap + h3, height: h3 }}
  ];
}}
function valToY(val, vMin, vMax, panel) {{ return panel.bottom - ((val - vMin) / (vMax - vMin)) * panel.height; }}

function drawBackground() {{
  const dpr = window.devicePixelRatio || 1;
  W = wrapper.clientWidth; H = wrapper.clientHeight;
  bgCanvas.width = W * dpr; bgCanvas.height = H * dpr;
  ovCanvas.width = W * dpr; ovCanvas.height = H * dpr;
  bgCtx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ovCtx.setTransform(dpr, 0, 0, dpr, 0, 0);

  bgCtx.fillStyle = "#090D16"; bgCtx.fillRect(0, 0, W, H);
  const panels = getPanels();
  const plotW = W - MARGIN.left - MARGIN.right;
  const p1 = panels[0], p2 = panels[1], p3 = panels[2];

  if (showSpec && specCanvas) {{
    bgCtx.save();
    bgCtx.beginPath(); bgCtx.rect(MARGIN.left, p2.top, plotW, p2.height); bgCtx.clip();
    bgCtx.drawImage(specCanvas, MARGIN.left, p2.top, plotW, p2.height);
    bgCtx.restore();
  }}

  const stepT = (xMax - xMin) <= 0.6 ? 0.05 : ((xMax - xMin) <= 1.5 ? 0.1 : 0.2);
  const firstTick = Math.ceil(xMin / stepT) * stepT;

  panels.forEach((p) => {{
    for (let t = firstTick; t <= xMax + 1e-6; t += stepT) {{
      const x = tToX(t);
      bgCtx.strokeStyle = "rgba(148, 163, 184, 0.14)"; bgCtx.lineWidth = 1.0;
      bgCtx.beginPath(); bgCtx.moveTo(x, p.top); bgCtx.lineTo(x, p.bottom); bgCtx.stroke();
      bgCtx.fillStyle = "#94A3B8"; bgCtx.font = "12px sans-serif"; bgCtx.textAlign = "center";
      bgCtx.fillText(t.toFixed(2), x, p.bottom + 14);
    }}
  }});

  const maxAmp = Math.max(...D.wave_y.map(Math.abs), 0.05);
  bgCtx.save();
  bgCtx.beginPath(); bgCtx.rect(MARGIN.left, p1.top, plotW, p1.height); bgCtx.clip();
  bgCtx.strokeStyle = "#94A3B8"; bgCtx.lineWidth = 1.5; bgCtx.beginPath();
  for (let i = 0; i < D.wave_t.length; i++) {{
    const x = tToX(D.wave_t[i]); const y = valToY(D.wave_y[i], -maxAmp * 1.1, maxAmp * 1.1, p1);
    if (i === 0) bgCtx.moveTo(x, y); else bgCtx.lineTo(x, y);
  }}
  bgCtx.stroke(); bgCtx.restore();

  const fMin = 500, fMax = 4000;
  [1000, 2000, 3000, 4000].forEach(hz => {{
    const y = valToY(hz, fMin, fMax, p2);
    bgCtx.strokeStyle = "rgba(148, 163, 184, 0.14)"; bgCtx.beginPath(); bgCtx.moveTo(MARGIN.left, y); bgCtx.lineTo(W - MARGIN.right, y); bgCtx.stroke();
    bgCtx.fillStyle = "#94A3B8"; bgCtx.font = "12px sans-serif"; bgCtx.textAlign = "right"; bgCtx.fillText(hz.toString(), MARGIN.left - 8, y + 4);
  }});

  bgCtx.save();
  bgCtx.beginPath(); bgCtx.rect(MARGIN.left, p2.top, plotW, p2.height); bgCtx.clip();
  drawSeries(D.times, D.f3, fMin, fMax, p2, "#38BDF8", 3.0);
  drawSeries(D.times, D.f2, fMin, fMax, p2, "#10B981", 3.0);
  bgCtx.restore();

  const dMin = 0, dMax = 2500;
  const yTarget = valToY(1000, dMin, dMax, p3);
  const yZero = valToY(0, dMin, dMax, p3);
  bgCtx.fillStyle = "rgba(239, 68, 68, 0.08)"; bgCtx.fillRect(MARGIN.left, yTarget, plotW, yZero - yTarget);
  bgCtx.strokeStyle = "rgba(239, 68, 68, 0.65)"; bgCtx.lineWidth = 1.5; bgCtx.setLineDash([6, 5]);
  bgCtx.beginPath(); bgCtx.moveTo(MARGIN.left, yTarget); bgCtx.lineTo(W - MARGIN.right, yTarget); bgCtx.stroke(); bgCtx.setLineDash([]);

  bgCtx.save();
  bgCtx.beginPath(); bgCtx.rect(MARGIN.left, p3.top, plotW, p3.height); bgCtx.clip();
  drawSeries(D.times, D.delta, dMin, dMax, p3, "#F59E0B", 3.2);
  bgCtx.restore();

  panels.forEach(p => {{
    bgCtx.strokeStyle = "#334155"; bgCtx.lineWidth = 1.5;
    bgCtx.beginPath(); bgCtx.moveTo(MARGIN.left, p.top); bgCtx.lineTo(MARGIN.left, p.bottom); bgCtx.lineTo(W - MARGIN.right, p.bottom); bgCtx.stroke();
  }});
}}

function drawSeries(times, values, vMin, vMax, panel, color, width) {{
  bgCtx.strokeStyle = color; bgCtx.lineWidth = width; bgCtx.lineJoin = "round"; bgCtx.beginPath();
  let penDown = false;
  for (let i = 0; i < times.length; i++) {{
    const val = values[i];
    if (val === null || val === undefined) {{ penDown = false; continue; }}
    const x = tToX(times[i]); const y = valToY(val, vMin, vMax, panel);
    if (!penDown) {{ bgCtx.moveTo(x, y); penDown = true; }} else {{ bgCtx.lineTo(x, y); }}
  }}
  bgCtx.stroke();
}}

function drawOverlay() {{
  ovCtx.clearRect(0, 0, W, H);
  const panels = getPanels();
  const topY = panels[0].top, botY = panels[2].bottom;
  if (Math.abs(selEnd - selStart) > 0.005) {{
    const sx = tToX(Math.min(selStart, selEnd)), ex = tToX(Math.max(selStart, selEnd));
    ovCtx.fillStyle = "rgba(255, 255, 255, 0.08)"; ovCtx.fillRect(sx, topY, ex - sx, botY - topY);
    ovCtx.strokeStyle = "rgba(255, 255, 255, 0.4)"; ovCtx.lineWidth = 1.0;
    ovCtx.beginPath(); ovCtx.moveTo(sx, topY); ovCtx.lineTo(sx, botY); ovCtx.moveTo(ex, topY); ovCtx.lineTo(ex, botY); ovCtx.stroke();
  }}
  const cx = tToX(cursorT);
  ovCtx.strokeStyle = "#EF4444"; ovCtx.lineWidth = 2.0;
  ovCtx.beginPath(); ovCtx.moveTo(cx, topY); ovCtx.lineTo(cx, botY); ovCtx.stroke();

  let idx = D.times.findIndex(t => t >= cursorT);
  let html = `<span style="color:#94A3B8;">Cursor:</span> ${{cursorT.toFixed(3)}} s`;
  if (idx !== -1 && D.delta[idx] !== null) {{
    html += `  |  <span style="color:#38BDF8;">F3:</span> ${{D.f3[idx]}} Hz   <span style="color:#10B981;">F2:</span> ${{D.f2[idx]}} Hz   <span style="color:#F59E0B;">Δ:</span> ${{D.delta[idx]}} Hz`;
  }}
  readout.innerHTML = html;
}}

btnSpec.addEventListener("click", () => {{
  showSpec = !showSpec;
  btnSpec.textContent = showSpec ? "Spectrogram: ON" : "Spectrogram: OFF";
  btnSpec.classList.toggle("toggle-active", showSpec);
  drawBackground();
}});

function startPlayback() {{
  if (!audioEl.src) return; stopPlayback();
  audioEl.playbackRate = parseFloat(selRate.value);
  audioEl.preservesPitch = true; audioEl.mozPreservesPitch = true; audioEl.webkitPreservesPitch = true;
  const s = Math.min(selStart, selEnd), e = Math.max(selStart, selEnd);
  audioEl.currentTime = (cursorT >= s && cursorT < e - 0.02) ? cursorT : s;
  isPlaying = true; btnPlay.textContent = "⏹ Stop (Space)"; btnPlay.className = "btn-play-active";
  audioEl.play().catch(() => stopPlayback());

  function tick() {{
    if (!isPlaying) return;
    cursorT = audioEl.currentTime;
    if (cursorT >= e || audioEl.ended) {{
      audioEl.pause(); isPlaying = false; cursorT = e;
      btnPlay.textContent = "▶ Play (Space)"; btnPlay.className = "btn-play-ready";
      drawOverlay(); return;
    }}
    drawOverlay(); animId = requestAnimationFrame(tick);
  }}
  animId = requestAnimationFrame(tick);
}}
function stopPlayback() {{
  audioEl.pause(); if (animId) cancelAnimationFrame(animId);
  isPlaying = false; btnPlay.textContent = "▶ Play (Space)"; btnPlay.className = "btn-play-ready";
}}
selRate.addEventListener("change", () => {{ audioEl.playbackRate = parseFloat(selRate.value); audioEl.preservesPitch = true; }});
btnPlay.addEventListener("click", () => isPlaying ? stopPlayback() : startPlayback());
btnReset.addEventListener("click", () => {{ stopPlayback(); selStart = xMin; selEnd = xMax; cursorT = (xMin + xMax) / 2; drawOverlay(); }});
btnDlPlot.addEventListener("click", () => {{
  const expCanvas = document.createElement("canvas");
  expCanvas.width = bgCanvas.width; expCanvas.height = bgCanvas.height;
  const eCtx = expCanvas.getContext("2d");
  eCtx.drawImage(bgCanvas, 0, 0); eCtx.drawImage(ovCanvas, 0, 0);
  const link = document.createElement("a");
  link.download = `NinjaPhonetics_${{D.export_name}}.png`; link.href = expCanvas.toDataURL("image/png"); link.click();
}});
btnDlAudio.addEventListener("click", () => {{
  if (!audioBlobUrl) return;
  const link = document.createElement("a"); link.download = `NinjaPhonetics_${{D.export_name}}.wav`; link.href = audioBlobUrl; link.click();
}});
window.addEventListener("keydown", (e) => {{
  if (e.code === "Space" && e.target === document.body) {{ e.preventDefault(); isPlaying ? stopPlayback() : startPlayback(); }}
}});

let dragMode = "none";
ovCanvas.addEventListener("mousemove", (e) => {{
  const rect = ovCanvas.getBoundingClientRect(); const mouseX = e.clientX - rect.left;
  if (dragMode === "none") ovCanvas.style.cursor = Math.abs(mouseX - tToX(cursorT)) <= 12 ? "ew-resize" : "crosshair";
  else if (dragMode === "cursor") {{ cursorT = xToT(mouseX); drawOverlay(); }}
  else if (dragMode === "select") {{ selEnd = xToT(mouseX); cursorT = selEnd; drawOverlay(); }}
}});
ovCanvas.addEventListener("mousedown", (e) => {{
  stopPlayback(); const rect = ovCanvas.getBoundingClientRect(); const mouseX = e.clientX - rect.left;
  if (Math.abs(mouseX - tToX(cursorT)) <= 12) dragMode = "cursor";
  else {{ dragMode = "select"; selStart = xToT(mouseX); selEnd = selStart; cursorT = selStart; drawOverlay(); }}
}});
window.addEventListener("mouseup", () => {{
  if (dragMode === "select" && Math.abs(selEnd - selStart) < 0.015) {{ selStart = xMin; selEnd = xMax; }}
  dragMode = "none"; drawOverlay();
}});
window.addEventListener("resize", () => {{ drawBackground(); drawOverlay(); }});
drawBackground(); drawOverlay();
if (D.auto_play) setTimeout(() => {{ cursorT = xMin; startPlayback(); }}, 600);
</script>
</body>
</html>
"""

# =====================================================================
# MODE 2: E2 プロソディ診断エンジン (ドリル & 総合アセスメント)
# =====================================================================
@st.cache_data
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
        p_v = float(row["pct_v"])
        s_u = float(row.get("schwa_p75_ms", row.get("schwa_p75", 60.0)))
        pts.append({"pct_v": p_v, "schwa_p75_ms": s_u, "color": col, "cohort": c_name})
    return pts

def compute_content_function_metrics(w_aligns, phones):
    word_breakdown = []
    content_vowel_durs = []
    func_vowel_durs = []
    diagnostic_func_vowel_durs = []

    for w in w_aligns:
        raw_word = w.get("word", "")
        clean_w = re.sub(r"[^a-zA-Z']", "", raw_word).lower()
        if not clean_w:
            continue
        is_func = clean_w in ALL_FUNCTION_WORDS
        is_diag = clean_w in DIAGNOSTIC_FUNCTION_WORDS
        w_start = w.get("start", 0.0)
        w_end = w.get("end", 0.0)
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
            "word": raw_word, "is_func": is_func, "is_diag": is_diag, "dur_ms": round(w_dur_ms, 1),
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
    norm_custom = " ".join(custom_text.strip().split())
    norm_preset = " ".join(bm["target_text"].strip().split())
    is_custom_mode = (norm_custom.lower() != norm_preset.lower())

    w_aligns = align_word_characters(audio_path, norm_custom)
    phones = map_words_to_phonemes(w_aligns)
    metrics = compute_robust_metrics(phones)
    r_cf, mean_c_ms, mean_f_ms, diag_median_ms, word_breakdown = compute_content_function_metrics(w_aligns, phones)

    pct_v = metrics["pct_v"]
    schwa_p75 = metrics["schwa_p75_ms"]
    npvi = metrics["npvi_clipped"]

    # 4次元マルチパラメータ判定
    cond_schwa = schwa_p75 <= 65.0
    cond_diag = diag_median_ms <= 48.0
    cond_rcf = r_cf >= 1.30

    if not is_custom_mode:
        user_pt = np.array([pct_v, schwa_p75])
        mu = bm["mu"]
        cov = bm["cov"]
        inv_cov = np.linalg.inv(cov)
        diff = user_pt - mu
        d_m_sq = round(float(diff.T @ inv_cov @ diff), 2)

        is_strike = (d_m_sq <= 5.991) and cond_diag
        is_bullseye = (d_m_sq <= (5.991 / 20.0)) and (diag_median_ms <= 42.0)

        if is_bullseye:
            verdict = "PERFECT STRIKE"
            status_color = "#38BDF8"
            feedback = f"理想的なストレス拍！弱母音({schwa_p75:.1f}ms)と重要機能語({diag_median_ms:.1f}ms)が完璧に小粒化されています。"
        elif is_strike:
            verdict = "NATIVE STRIKE"
            status_color = "#10B981"
            feedback = f"ネイティブ許容ゾーンに合格！(V_diag={diag_median_ms:.1f}ms, R_CF={r_cf}x, nPVI={npvi:.1f})。"
        else:
            status_color = "#EF4444"
            if diag_median_ms > 65.0 or schwa_p75 > 75.0:
                verdict = "MORA TRAP (等時拍)"
                feedback = f"前置詞・接続詞・助動詞が {diag_median_ms:.0f}ms と長すぎます。意味を担わない機能語を一瞬で通過してください。"
            elif pct_v > mu[0] + 3.5:
                verdict = "VOWEL HEAVY (子音崩れ)"
                feedback = f"母音占有率(%V)が +{pct_v - mu[0]:.1f}% 高すぎます。子音部への不要な母音挿入に注意してください。"
            else:
                verdict = "OFF TARGET"
                feedback = "ストライクゾーンから外れています。内容語をしっかり伸ばし、機能語を極小化してください。"
    else:
        d_m_sq = None
        if cond_schwa and cond_diag and cond_rcf:
            verdict = "DYNAMIC STRIKE (自由文合格)"
            status_color = "#10B981"
            is_strike = True
            is_bullseye = (diag_median_ms <= 42.0 and r_cf >= 1.60)
            feedback = f"自由文で理想的なストレス拍！(V_diag = {diag_median_ms:.1f}ms, R_CF = {r_cf}x, nPVI = {npvi:.1f})"
        elif schwa_p75 <= 75.0 and diag_median_ms <= 60.0:
            verdict = "PASSABLE RHYTHM (惜しい)"
            status_color = "#F59E0B"
            is_strike = False
            is_bullseye = False
            feedback = f"リズム骨格はありますが、機能語(現在 {diag_median_ms:.0f}ms)をもう一段短く(目標 ≤45ms)できます。"
        else:
            verdict = "MORA RHYTHM (等時拍傾向)"
            status_color = "#EF4444"
            is_strike = False
            is_bullseye = False
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

def build_e2_radar_html(passage_key: str, eval_history: list = None, is_custom_mode: bool = False) -> str:
    bm = E2_PASSAGE_BENCHMARKS[passage_key]
    mu = bm["mu"]
    cov = bm["cov"]

    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    
    n_std_95 = 2.447
    v1_95 = vecs[:, 0] * (n_std_95 * np.sqrt(max(vals[0], 0)))
    v2_95 = vecs[:, 1] * (n_std_95 * np.sqrt(max(vals[1], 0)))

    n_std_ci = 2.447 / np.sqrt(20)
    v1_ci = vecs[:, 0] * (n_std_ci * np.sqrt(max(vals[0], 0)))
    v2_ci = vecs[:, 1] * (n_std_ci * np.sqrt(max(vals[1], 0)))

    bench_points = load_benchmark_points(bm["bm_csv"], passage_key) if not is_custom_mode else []
    history_list = eval_history if eval_history is not None else []

    canvas_data = {
        "passage_key": passage_key,
        "is_custom_mode": is_custom_mode,
        "mu_x": float(mu[0]), "mu_y": float(mu[1]),
        "v1_95": [float(v1_95[0]), float(v1_95[1])],
        "v2_95": [float(v2_95[0]), float(v2_95[1])],
        "v1_ci": [float(v1_ci[0]), float(v1_ci[1])],
        "v2_ci": [float(v2_ci[0]), float(v2_ci[1])],
        "x_min": 30.0 if is_custom_mode else bm["x_range"][0],
        "x_max": 58.0 if is_custom_mode else bm["x_range"][1],
        "y_min": 30.0 if is_custom_mode else bm["y_range"][0],
        "y_max": 130.0 if is_custom_mode else bm["y_range"][1],
        "history": history_list,
        "benchmark": bench_points
    }

    mode_badge = "✏️ CUSTOM FREE-TEXT MODE" if is_custom_mode else f"🎯 2D Strike Zone (Trials: {len(history_list)})"

    return f"""
<!DOCTYPE html>
<html>
<head>
<style>
  body {{ margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: #0B0F19; color: #E2E8F0; user-select: none; overflow: hidden; }}
  .toolbar {{ display: flex; align-items: center; justify-content: space-between; padding: 6px 12px; background: #0F172A; border: 1px solid #1E293B; border-radius: 8px 8px 0 0; }}
  .toolbar-title {{ font-size: 13px; font-weight: 600; color: #94A3B8; display: flex; align-items: center; gap: 8px; }}
  .btn-group {{ display: flex; align-items: center; gap: 8px; }}
  button {{ font-family: inherit; font-size: 12px; font-weight: bold; padding: 5px 12px; border-radius: 6px; border: 1px solid #334155; cursor: pointer; background: #1E293B; color: #E2E8F0; transition: all 0.15s; }}
  button:hover {{ background: #334155; color: #FFFFFF; }}
  button.toggle-active {{ background: #1E3A8A; color: #93C5FD; border-color: #3B82F6; }}
  #scope-wrapper {{ position: relative; width: 100%; height: 410px; background: #090D16; border: 1px solid #1E293B; border-top: none; border-radius: 0 0 8px 8px; }}
  canvas {{ position: absolute; left: 0; top: 0; width: 100%; height: 100%; }}
</style>
</head>
<body>
<div class="toolbar">
  <div class="toolbar-title"><span>{mode_badge}</span></div>
  <div class="btn-group">
    <button id="btn-toggle-bm" class="toggle-active" style="display: {'none' if is_custom_mode else 'inline-block'};">👥 Benchmark: ON</button>
    <button id="btn-dl-radar">💾 Radar (.png)</button>
  </div>
</div>
<div id="scope-wrapper">
  <canvas id="radar-canvas"></canvas>
</div>
<script>
const D = {json.dumps(canvas_data)};
const canvas = document.getElementById('radar-canvas');
const ctx = canvas.getContext('2d');
const wrapper = document.getElementById('scope-wrapper');
const btnDl = document.getElementById('btn-dl-radar');
const btnToggleBm = document.getElementById('btn-toggle-bm');
let showBenchmark = true;

const MARGIN = {{ left: 80, right: 30, top: 20, bottom: 45 }};
let W = 800, H = 410;

function xToPx(x) {{ return MARGIN.left + ((x - D.x_min) / (D.x_max - D.x_min)) * (W - MARGIN.left - MARGIN.right); }}
function yToPx(y) {{ return (H - MARGIN.bottom) - ((y - D.y_min) / (D.y_max - D.y_min)) * (H - MARGIN.top - MARGIN.bottom); }}

function createEllipsePath(ctx, mu_x, mu_y, v1, v2) {{
  ctx.beginPath();
  const N_PTS = 64;
  for (let i = 0; i <= N_PTS; i++) {{
    const theta = (i / N_PTS) * 2 * Math.PI;
    const dataX = mu_x + Math.cos(theta) * v1[0] + Math.sin(theta) * v2[0];
    const dataY = mu_y + Math.cos(theta) * v1[1] + Math.sin(theta) * v2[1];
    const px = xToPx(dataX), py = yToPx(dataY);
    if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
  }}
  ctx.closePath();
}}

function draw() {{
  const dpr = window.devicePixelRatio || 1;
  W = wrapper.clientWidth; H = wrapper.clientHeight;
  canvas.width = W * dpr; canvas.height = H * dpr;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

  ctx.fillStyle = "#090D16"; ctx.fillRect(0, 0, W, H);
  const plotW = W - MARGIN.left - MARGIN.right;
  const plotH = H - MARGIN.top - MARGIN.bottom;

  // グリッド線
  ctx.strokeStyle = "rgba(148, 163, 184, 0.10)"; ctx.lineWidth = 1.0;
  const stepX = (D.x_max - D.x_min) > 20 ? 4 : 2;
  for (let x = Math.ceil(D.x_min / stepX) * stepX; x <= D.x_max; x += stepX) {{
    const px = xToPx(x);
    ctx.beginPath(); ctx.moveTo(px, MARGIN.top); ctx.lineTo(px, H - MARGIN.bottom); ctx.stroke();
    ctx.fillStyle = "#64748B"; ctx.font = "12px sans-serif"; ctx.textAlign = "center";
    ctx.fillText(x + "%", px, H - MARGIN.bottom + 18);
  }}
  for (let y = Math.ceil(D.y_min / 10) * 10; y <= D.y_max; y += 10) {{
    const py = yToPx(y);
    ctx.beginPath(); ctx.moveTo(MARGIN.left, py); ctx.lineTo(W - MARGIN.right, py); ctx.stroke();
    ctx.fillStyle = "#64748B"; ctx.font = "12px sans-serif"; ctx.textAlign = "right"; ctx.textBaseline = "middle";
    ctx.fillText(y + "ms", MARGIN.left - 8, py);
  }}

  ctx.strokeStyle = "#334155"; ctx.lineWidth = 1.5;
  ctx.strokeRect(MARGIN.left, MARGIN.top, plotW, plotH);

  ctx.fillStyle = "#94A3B8"; ctx.font = "bold 13px sans-serif"; ctx.textAlign = "center";
  ctx.fillText("Vowel Occupancy: %V (%)", MARGIN.left + plotW / 2, H - 6);
  ctx.save(); ctx.translate(22, MARGIN.top + plotH / 2); ctx.rotate(-Math.PI / 2);
  ctx.fillText("Schwa P75 Duration (ms)", 0, 0); ctx.restore();

  ctx.save();
  ctx.beginPath(); ctx.rect(MARGIN.left, MARGIN.top, plotW, plotH); ctx.clip();

  if (!D.is_custom_mode) {{
    // 95% Strike Zone
    createEllipsePath(ctx, D.mu_x, D.mu_y, D.v1_95, D.v2_95);
    ctx.fillStyle = "rgba(16, 185, 129, 0.15)"; ctx.fill();
    ctx.strokeStyle = "#10B981"; ctx.lineWidth = 2.0; ctx.setLineDash([5, 4]); ctx.stroke();

    // Centroid
    createEllipsePath(ctx, D.mu_x, D.mu_y, D.v1_ci, D.v2_ci);
    ctx.fillStyle = "rgba(56, 189, 248, 0.35)"; ctx.fill();
    ctx.strokeStyle = "#38BDF8"; ctx.lineWidth = 1.2; ctx.setLineDash([]); ctx.stroke();

    if (showBenchmark && D.benchmark && D.benchmark.length > 0) {{
      D.benchmark.forEach(b => {{
        const bx = xToPx(b.pct_v), by = yToPx(b.schwa_p75_ms);
        ctx.save(); ctx.globalAlpha = 0.40; ctx.fillStyle = b.color;
        ctx.beginPath(); ctx.arc(bx, by, 4.5, 0, 2 * Math.PI); ctx.fill(); ctx.restore();
      }});
      const legX = W - MARGIN.right - 145, legY = MARGIN.top + 8;
      ctx.fillStyle = "rgba(9, 13, 22, 0.82)"; ctx.fillRect(legX, legY, 135, 52);
      ctx.strokeStyle = "#1E293B"; ctx.strokeRect(legX, legY, 135, 52);
      ctx.font = "10.5px sans-serif"; ctx.textAlign = "left"; ctx.textBaseline = "middle";
      ctx.fillStyle = "#38BDF8"; ctx.beginPath(); ctx.arc(legX + 12, legY + 12, 3.5, 0, 2 * Math.PI); ctx.fill();
      ctx.fillStyle = "#94A3B8"; ctx.fillText("Native (US/UK)", legX + 22, legY + 12);
      ctx.fillStyle = "#F59E0B"; ctx.beginPath(); ctx.arc(legX + 12, legY + 26, 3.5, 0, 2 * Math.PI); ctx.fill();
      ctx.fillStyle = "#94A3B8"; ctx.fillText("JP-Accented", legX + 22, legY + 26);
      ctx.fillStyle = "#EF4444"; ctx.beginPath(); ctx.arc(legX + 12, legY + 40, 3.5, 0, 2 * Math.PI); ctx.fill();
      ctx.fillStyle = "#94A3B8"; ctx.fillText("JP-Mora-Forced", legX + 22, legY + 40);
    }}
  }} else {{
    const zx1 = xToPx(35.0), zx2 = xToPx(49.0);
    const zy1 = yToPx(65.0), zy2 = yToPx(32.0);
    ctx.fillStyle = "rgba(16, 185, 129, 0.14)"; ctx.fillRect(zx1, zy1, zx2 - zx1, zy2 - zy1);
    ctx.strokeStyle = "#10B981"; ctx.lineWidth = 2.0; ctx.setLineDash([6, 4]); ctx.strokeRect(zx1, zy1, zx2 - zx1, zy2 - zy1); ctx.setLineDash([]);
    ctx.fillStyle = "#10B981"; ctx.font = "bold 12px sans-serif"; ctx.textAlign = "left";
    ctx.fillText("UNIVERSAL STRESS-TIMED ZONE (Schwa ≤ 65ms)", zx1 + 8, zy1 + 16);
  }}

  // ユーザー試行履歴
  const total = D.history.length;
  if (total > 0) {{
    if (total > 1) {{
      ctx.strokeStyle = "rgba(248, 250, 252, 0.30)"; ctx.lineWidth = 1.5; ctx.setLineDash([3, 3]); ctx.beginPath();
      for (let i = 0; i < total; i++) {{
        const px = xToPx(D.history[i].pct_v), py = yToPx(D.history[i].schwa_p75_ms);
        if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
      }}
      ctx.stroke(); ctx.setLineDash([]);
    }}

    D.history.forEach((pt, idx) => {{
      const isLatest = (idx === total - 1);
      const px = xToPx(pt.pct_v), py = yToPx(pt.schwa_p75_ms);
      const alpha = total === 1 ? 1.0 : 0.35 + (idx / (total - 1)) * 0.65;
      ctx.save(); ctx.globalAlpha = alpha;
      if (isLatest) {{
        ctx.fillStyle = pt.status_color + "44"; ctx.beginPath(); ctx.arc(px, py, 18, 0, 2 * Math.PI); ctx.fill();
        ctx.fillStyle = pt.status_color; ctx.strokeStyle = "#FFFFFF"; ctx.lineWidth = 2.5; ctx.beginPath(); ctx.arc(px, py, 8, 0, 2 * Math.PI); ctx.fill(); ctx.stroke();
        ctx.fillStyle = "#F8FAFC"; ctx.font = "bold 12px sans-serif"; ctx.textAlign = px > W - 140 ? "right" : "left";
        ctx.fillText(`#${{idx+1}} (V_diag:${{pt.diag_median_ms}}ms): ${{pt.pct_v}}%, ${{pt.schwa_p75_ms}}ms`, px + (px > W - 140 ? -12 : 12), py - 8);
      }} else {{
        ctx.fillStyle = pt.status_color; ctx.strokeStyle = "rgba(11, 15, 25, 0.9)"; ctx.lineWidth = 1.5; ctx.beginPath(); ctx.arc(px, py, 6, 0, 2 * Math.PI); ctx.fill(); ctx.stroke();
        ctx.fillStyle = "#E2E8F0"; ctx.font = "10px sans-serif"; ctx.textAlign = "center"; ctx.fillText(`#${{idx+1}}`, px, py - 8);
      }}
      ctx.restore();
    }});
  }}
  ctx.restore();
}}

if (btnToggleBm) {{
  btnToggleBm.addEventListener("click", () => {{
    showBenchmark = !showBenchmark;
    btnToggleBm.textContent = showBenchmark ? "👥 Benchmark: ON" : "👥 Benchmark: OFF";
    btnToggleBm.classList.toggle("toggle-active", showBenchmark);
    draw();
  }});
}}
btnDl.addEventListener("click", () => {{
  const link = document.createElement("a");
  link.download = `NinjaPhonetics_E2_${{D.passage_key}}_radar.png`; link.href = canvas.toDataURL("image/png"); link.click();
}});
window.addEventListener("resize", draw);
draw();
</script>
</body>
</html>
"""

# =====================================================================
# メイン画面描画分岐
# =====================================================================
if st.session_state.app_mode == "RL":
    cols = st.columns(len(PRESETS))
    preset_clicked = False
    for idx, col in enumerate(cols):
        p_info = PRESETS[idx]
        is_active = (st.session_state.current_label == f"お手本: {p_info['label']}")
        if col.button(p_info["label"], type="primary" if is_active else "secondary", use_container_width=True):
            file_path = os.path.join(PRESET_DIR, p_info["file"])
            if os.path.exists(file_path):
                with open(file_path, "rb") as f:
                    st.session_state.current_audio_bytes = f.read()
                    st.session_state.current_label = f"お手本: {p_info['label']}"
                    st.session_state.auto_play = True
                preset_clicked = True
                st.rerun()

    audio_input = st.audio_input("または自分の声を録音して解析 (R / L)")
    if not preset_clicked and audio_input is not None:
        new_bytes = audio_input.read()
        if new_bytes != st.session_state.last_recorded_bytes:
            st.session_state.last_recorded_bytes = new_bytes
            st.session_state.current_audio_bytes = new_bytes
            st.session_state.current_label = "あなたの発音録音"
            st.session_state.auto_play = False

    if st.session_state.current_audio_bytes is not None:
        if st.session_state.current_label:
            st.markdown(
                f"<div style='color: #94A3B8; font-size: 1.85rem; font-weight: 700; margin-top: 14px; margin-bottom: 12px; line-height: 1.25;'>"
                f"● 表示中: <span style='color: #F8FAFC;'>{st.session_state.current_label}</span>"
                f"</div>",
                unsafe_allow_html=True
            )
        clean_export_name = re.sub(r'[^a-zA-Z0-9_\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF]', '_', st.session_state.current_label)
        clean_export_name = re.sub(r'_+', '_', clean_export_name).strip('_')

        scope_data = extract_scope_data_from_bytes(st.session_state.current_audio_bytes)
        html_code = build_spike2_html(scope_data, auto_play=st.session_state.auto_play, export_name=clean_export_name or "sample")
        st.session_state.auto_play = False
        components.html(html_code, height=640, scrolling=False)

else:
    # =====================================================================
    # --- MODE 2: E2 3大プロソディ診断 (2階層: ドリル vs 総合アセスメント) ---
    # =====================================================================
    cat_col1, cat_col2 = st.columns(2)
    with cat_col1:
        if st.button(
            "🥊 フォーカス・ドリル (短文・反復修正)",
            type="primary" if st.session_state.e2_category == "drill" else "secondary",
            use_container_width=True
        ):
            if st.session_state.e2_category != "drill":
                st.session_state.e2_category = "drill"
                st.session_state.e2_selected_passage = "S1_stress"
                target_txt = E2_PASSAGE_BENCHMARKS["S1_stress"]["target_text"]
                st.session_state.e2_custom_text = target_txt
                st.session_state.e2_last_analyzed_text = target_txt
                st.session_state["e2_text_input_widget"] = target_txt  # ウィジェット値を強制同期
                st.session_state.e2_eval_history = []
                st.session_state.e2_audio_bytes = None
                st.rerun()
    with cat_col2:
        if st.button(
            "🏆 総合アセスメント (長文・標準規格)",
            type="primary" if st.session_state.e2_category == "passage" else "secondary",
            use_container_width=True
        ):
            if st.session_state.e2_category != "passage":
                st.session_state.e2_category = "passage"
                st.session_state.e2_selected_passage = "P1_northwind"
                target_txt = E2_PASSAGE_BENCHMARKS["P1_northwind"]["target_text"]
                st.session_state.e2_custom_text = target_txt
                st.session_state.e2_last_analyzed_text = target_txt
                st.session_state["e2_text_input_widget"] = target_txt  # ウィジェット値を強制同期
                st.session_state.e2_eval_history = []
                st.session_state.e2_audio_bytes = None
                st.rerun()

    # カテゴリごとの課題選択ボタン
    if st.session_state.e2_category == "drill":
        d_col1, d_col2, d_col3 = st.columns(3)
        with d_col1:
            if st.button("ドリル 1: 語彙強弱 (S1)", type="primary" if st.session_state.e2_selected_passage == "S1_stress" else "secondary", use_container_width=True):
                st.session_state.e2_selected_passage = "S1_stress"
                target_txt = E2_PASSAGE_BENCHMARKS["S1_stress"]["target_text"]
                st.session_state.e2_custom_text = target_txt
                st.session_state.e2_last_analyzed_text = target_txt
                st.session_state["e2_text_input_widget"] = target_txt
                st.session_state.e2_eval_history = []
                st.rerun()
        with d_col2:
            if st.button("ドリル 2: 子音連結 (S2)", type="primary" if st.session_state.e2_selected_passage == "S2_clusters" else "secondary", use_container_width=True):
                st.session_state.e2_selected_passage = "S2_clusters"
                target_txt = E2_PASSAGE_BENCHMARKS["S2_clusters"]["target_text"]
                st.session_state.e2_custom_text = target_txt
                st.session_state.e2_last_analyzed_text = target_txt
                st.session_state["e2_text_input_widget"] = target_txt
                st.session_state.e2_eval_history = []
                st.rerun()
        with d_col3:
            if st.button("ドリル 3: 機能語弱化 (S3)", type="primary" if st.session_state.e2_selected_passage == "S3_connected" else "secondary", use_container_width=True):
                st.session_state.e2_selected_passage = "S3_connected"
                target_txt = E2_PASSAGE_BENCHMARKS["S3_connected"]["target_text"]
                st.session_state.e2_custom_text = target_txt
                st.session_state.e2_last_analyzed_text = target_txt
                st.session_state["e2_text_input_widget"] = target_txt
                st.session_state.e2_eval_history = []
                st.rerun()
    else:
        p_col1, p_col2, p_col3 = st.columns(3)
        with p_col1:
            if st.button("標準 1: 北風と太陽 (P1)", type="primary" if st.session_state.e2_selected_passage == "P1_northwind" else "secondary", use_container_width=True):
                st.session_state.e2_selected_passage = "P1_northwind"
                target_txt = E2_PASSAGE_BENCHMARKS["P1_northwind"]["target_text"]
                st.session_state.e2_custom_text = target_txt
                st.session_state.e2_last_analyzed_text = target_txt
                st.session_state["e2_text_input_widget"] = target_txt  # 強制同期
                st.session_state.e2_eval_history = []
                st.rerun()
        with p_col2:
            st.button("標準 2: 虹のパッセージ (P2)", disabled=True, use_container_width=True, help="来月クレジット更新後に開放予定")
        with p_col3:
            st.button("標準 3: 祖父のパッセージ (P3)", disabled=True, use_container_width=True, help="来月クレジット更新後に開放予定")

    active_bm = E2_PASSAGE_BENCHMARKS[st.session_state.e2_selected_passage]

    # カスタム英文入力テキストエリア
    txt_col1, txt_col2 = st.columns([4.5, 1.0], vertical_alignment="bottom")
    with txt_col1:
        text_height = 115 if st.session_state.e2_category == "passage" else 65
        user_target_text = st.text_area(
            "📝 ターゲット英文（自由に書き換え可能）",
            value=st.session_state.e2_custom_text,
            height=text_height,
            key="e2_text_input_widget"
        )
    with txt_col2:
        if st.button("↺ 例文に戻す", use_container_width=True):
            target_txt = active_bm["target_text"]
            st.session_state.e2_custom_text = target_txt
            st.session_state.e2_last_analyzed_text = target_txt
            st.session_state["e2_text_input_widget"] = target_txt
            st.session_state.e2_eval_history = []
            st.session_state.e2_audio_bytes = None
            st.rerun()

    if user_target_text.strip() != st.session_state.e2_last_analyzed_text.strip():
        st.session_state.e2_custom_text = user_target_text.strip()
        st.session_state.e2_last_analyzed_text = user_target_text.strip()
        st.session_state.e2_eval_history = []
        st.session_state.e2_audio_bytes = None

    is_custom_now = (
        " ".join(st.session_state.e2_custom_text.lower().split())
        != " ".join(active_bm["target_text"].lower().split())
    )

    # 単語バッジ表示 (冠詞抜きリトマス語は特別なオレンジバッジでハイライト)
    words_preview = st.session_state.e2_custom_text.split()
    preview_badges = []
    for w in words_preview:
        clean_w = re.sub(r"[^a-zA-Z']", "", w).lower()
        if clean_w in DIAGNOSTIC_FUNCTION_WORDS:
            # 最重要リトマス機能語 (前置詞・接続詞・助動詞)
            preview_badges.append(
                f"<span style='background:#78350F; color:#FDE68A; border:1px solid #D97706; padding:2px 6px; border-radius:4px; font-size:0.9rem; font-weight:700; margin-right:4px;'>{w} <small style='font-size:0.68rem; color:#F59E0B;'>[弱化必須]</small></span>"
            )
        elif clean_w in ALL_FUNCTION_WORDS:
            # 冠詞など一般の機能語
            preview_badges.append(
                f"<span style='background:#1E293B; color:#94A3B8; border:1px solid #334155; padding:2px 6px; border-radius:4px; font-size:0.9rem; margin-right:4px;'>{w}</span>"
            )
        else:
            # 内容語
            preview_badges.append(
                f"<span style='background:#1E3A8A; color:#F8FAFC; border:1px solid #3B82F6; padding:2px 7px; border-radius:4px; font-size:0.95rem; font-weight:700; margin-right:4px;'>{w} <small style='font-size:0.68rem; color:#93C5FD;'>[内容]</small></span>"
            )

    mode_header = "✏️ CUSTOM FREE-TEXT MODE" if is_custom_now else active_bm["title"]
    mode_desc = "青色の【内容語】にしっかり時間をかけ、茶色の【弱化必須語 (to, at, and, was 等)】を極限まで短く（≤ 45ms）圧縮して発音してください。" if is_custom_now else active_bm["phonetics_desc"]

    st.markdown(f"""
    <div style="background: #0F172A; border: 1px solid #1E293B; border-left: 4px solid {'#F59E0B' if is_custom_now else '#38BDF8'}; border-radius: 8px; padding: 12px 18px; margin-bottom: 12px;">
      <div style="font-size: 0.82rem; font-weight: 700; color: {'#F59E0B' if is_custom_now else '#38BDF8'}; text-transform: uppercase; margin-bottom: 6px;">{mode_header}</div>
      <div style="line-height: 2.1; margin-bottom: 6px; max-height: {'135px' if st.session_state.e2_category == 'passage' else '70px'}; overflow-y: auto;">{''.join(preview_badges)}</div>
      <div style="font-size: 0.86rem; color: #94A3B8;">{mode_desc}</div>
    </div>
    """, unsafe_allow_html=True)

    card_col1, card_col2 = st.columns([1.5, 1.0])

    with card_col1:
        e2_input = st.audio_input("この英文を録音してプロソディ診断")
        if e2_input is not None:
            new_e2_bytes = e2_input.read()
            if new_e2_bytes != st.session_state.e2_audio_bytes:
                st.session_state.e2_audio_bytes = new_e2_bytes
                tmp_wav = "tmp_e2_eval.wav"
                with open(tmp_wav, "wb") as f:
                    f.write(new_e2_bytes)
                eval_res = evaluate_user_rhythm(
                    tmp_wav,
                    st.session_state.e2_selected_passage,
                    st.session_state.e2_custom_text
                )
                if os.path.exists(tmp_wav):
                    os.remove(tmp_wav)
                if "error" not in eval_res:
                    st.session_state.e2_eval_history.append(eval_res)

        radar_html = build_e2_radar_html(
            st.session_state.e2_selected_passage,
            st.session_state.e2_eval_history,
            is_custom_mode=is_custom_now
        )
        components.html(radar_html, height=460, scrolling=False)

    with card_col2:
        st.markdown("<div style='font-size: 1.05rem; font-weight: bold; color: #F8FAFC; margin-bottom: 8px;'>🎯 多次元プロソディ診断結果</div>", unsafe_allow_html=True)

        if st.session_state.e2_eval_history:
            latest_ev = st.session_state.e2_eval_history[-1]
            trial_no = len(st.session_state.e2_eval_history)
            dm_str = f"{latest_ev['d_m_sq']} (≤5.99 合格)" if latest_ev["d_m_sq"] is not None else "自由文判定"

            # 多次元ゲージのステータス色
            c_diag = "#10B981" if latest_ev["diag_median_ms"] <= 48.0 else ("#F59E0B" if latest_ev["diag_median_ms"] <= 65.0 else "#EF4444")
            c_schwa = "#10B981" if latest_ev["schwa_p75_ms"] <= 65.0 else "#EF4444"
            c_rcf = "#10B981" if latest_ev["r_cf"] >= 1.30 else "#EF4444"
            c_npvi = "#10B981" if 40.0 <= latest_ev["npvi_clipped"] <= 62.0 else "#F59E0B"

            st.markdown(f"""
            <div style="background: #1E293B; border: 1px solid #334155; border-radius: 8px; padding: 16px; margin-bottom: 12px;">
              <div style="display: flex; justify-content: space-between; align-items: baseline;">
                <span style="font-size: 0.85rem; color: #94A3B8; font-weight: 600;">VERDICT (Trial #{trial_no})</span>
                <span style="font-size: 0.8rem; color: #64748B;">蓄積データ: {trial_no} 件</span>
              </div>
              <div style="font-size: 1.5rem; font-weight: 800; color: {latest_ev['status_color']}; margin-bottom: 10px;">{latest_ev['verdict']}</div>
              
              <div style="display: flex; justify-content: space-between; border-bottom: 1px solid #334155; padding-bottom: 5px; margin-bottom: 5px;">
                <span style="color: #FDE68A; font-size: 0.9rem; font-weight: 700;">★ 診断機能語中央値 ($V_{{diag}}$):</span>
                <span style="font-weight: 800; color: {c_diag};">{latest_ev['diag_median_ms']} ms <small style='font-size:0.75rem; color:#94A3B8;'>(目標 ≤45ms)</small></span>
              </div>
              <div style="display: flex; justify-content: space-between; border-bottom: 1px solid #334155; padding-bottom: 5px; margin-bottom: 5px;">
                <span style="color: #94A3B8; font-size: 0.9rem;">弱母音持続時間 (Schwa P75):</span>
                <span style="font-weight: bold; color: {c_schwa};">{latest_ev['schwa_p75_ms']:.1f} ms <small style='font-size:0.75rem; color:#94A3B8;'>(目標 ≤65ms)</small></span>
              </div>
              <div style="display: flex; justify-content: space-between; border-bottom: 1px solid #334155; padding-bottom: 5px; margin-bottom: 5px;">
                <span style="color: #94A3B8; font-size: 0.9rem;">機能語圧縮比 ($R_{{CF}}$):</span>
                <span style="font-weight: bold; color: {c_rcf};">{latest_ev['r_cf']}x <small style='font-size:0.75rem; color:#94A3B8;'>(内容 {latest_ev['mean_c_ms']}ms / 機能 {latest_ev['mean_f_ms']}ms)</small></span>
              </div>
              <div style="display: flex; justify-content: space-between; border-bottom: 1px solid #334155; padding-bottom: 5px; margin-bottom: 5px;">
                <span style="color: #94A3B8; font-size: 0.9rem;">リズム伸縮指数 (nPVI-V):</span>
                <span style="font-weight: bold; color: {c_npvi};">{latest_ev['npvi_clipped']} <small style='font-size:0.75rem; color:#94A3B8;'>(適正 40〜60)</small></span>
              </div>
              <div style="display: flex; justify-content: space-between; margin-bottom: 12px;">
                <span style="color: #94A3B8; font-size: 0.9rem;">2D Mahalanobis ($D_M^2$):</span>
                <span style="font-weight: bold; color: #F8FAFC;">{dm_str}</span>
              </div>
              
              <div style="background: #0F172A; border-radius: 6px; padding: 10px; font-size: 0.88rem; color: #E2E8F0; line-height: 1.45;">
                💡 <b>アドバイス:</b><br>{latest_ev['feedback']}
              </div>
            </div>
            """, unsafe_allow_html=True)

            if latest_ev.get("word_breakdown"):
                st.markdown("<div style='font-size: 0.9rem; font-weight: bold; color: #CBD5E1; margin-top: 8px; margin-bottom: 6px;'>📊 単語別の母音持続時間 (黄色＝診断用機能語 / 赤＝引き伸ばし)</div>", unsafe_allow_html=True)
                bars_html = "<div style='background:#0F172A; border:1px solid #1E293B; border-radius:8px; padding:10px; max-height:190px; overflow-y:auto;'>"
                for wb in latest_ev["word_breakdown"]:
                    v_ms = wb["vowel_ms"]
                    width_pct = min(100, int((v_ms / 180.0) * 100))
                    if wb["is_diag"]:
                        bar_col = "#EF4444" if v_ms > 60.0 else "#F59E0B"
                        tag_lbl = "弱化語"
                    elif wb["is_func"]:
                        bar_col = "#64748B"
                        tag_lbl = "冠詞等"
                    else:
                        bar_col = "#3B82F6"
                        tag_lbl = "内容語"
                    bars_html += f"""
                    <div style="display:flex; align-items:center; gap:8px; margin-bottom:5px; font-size:0.82rem;">
                      <div style="width:105px; text-align:right; font-weight:600; color:#E2E8F0; overflow:hidden; text-overflow:ellipsis; white-space:nowrap;">{wb['word']} <span style="font-size:0.7rem; color:#64748B;">({tag_lbl})</span></div>
                      <div style="flex:1; background:#1E293B; height:10px; border-radius:4px; overflow:hidden;">
                        <div style="width:{width_pct}%; background:{bar_col}; height:100%;"></div>
                      </div>
                      <div style="width:52px; font-variant-numeric:tabular-nums; color:#94A3B8;">{v_ms:.0f}ms</div>
                    </div>
                    """
                bars_html += "</div>"
                st.markdown(bars_html, unsafe_allow_html=True)

            if st.button("この課題の履歴をクリア", use_container_width=True):
                st.session_state.e2_eval_history = []
                st.session_state.e2_audio_bytes = None
                st.rerun()

        else:
            st.markdown("""
            <div style="background: #1E293B; border: 1px dashed #475569; border-radius: 8px; padding: 24px; text-align: center; color: #94A3B8;">
              左側のマイクボタンを押して、上の英文を読み上げてください。<br><br>
              多次元解析（%V、Schwa P75、診断機能語中央値、R_CF）により、客観的かつポーズの影響を受けない高精度なストレス拍判定が行われます。
            </div>
            """, unsafe_allow_html=True)

        if not is_custom_now and os.path.exists(active_bm["ref_audio"]):
            st.markdown("<div style='font-size: 0.95rem; font-weight: bold; color: #94A3B8; margin-top: 14px; margin-bottom: 4px;'>🔊 ネイティブお手本音声</div>", unsafe_allow_html=True)
            with open(active_bm["ref_audio"], "rb") as rf:
                st.audio(rf.read(), format="audio/wav")