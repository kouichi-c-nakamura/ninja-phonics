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
import glob
import re
from pathlib import Path

# MMS_FA アライメントエンジンのインポート
try:
    from test_rhythm_align import align_word_characters, map_words_to_phonemes
    from evaluate_benchmarks import compute_robust_metrics
    ALIGNMENT_AVAILABLE = True
except ImportError:
    ALIGNMENT_AVAILABLE = False

st.set_page_config(page_title="NinjaPhonetics: Biofeedback Scope", layout="wide", initial_sidebar_state="collapsed")

# =====================================================================
# 手裏剣アイコンのロード (インデントなしで確実にHTMLとして解釈させる)
# =====================================================================
def get_shuriken_icon_html(size_px: int = 28) -> str:
    for fname in ["pink shuriken 32.png", "pink shuriken 640.jpg", "assets/pink shuriken 32.png"]:
        if os.path.exists(fname):
            with open(fname, "rb") as f:
                b64 = base64.b64encode(f.read()).decode("utf-8")
            mime = "image/png" if fname.endswith(".png") else "image/jpeg"
            return (
                f'<img src="data:{mime};base64,{b64}" width="{size_px}" height="{size_px}" '
                f'style="filter: hue-rotate(190deg) saturate(0.28) brightness(1.65) contrast(1.15) '
                f'drop-shadow(0 0 6px rgba(186, 230, 253, 0.55)); vertical-align: middle; border-radius: 4px;" />'
            )
    return (
        f'<svg width="{size_px}" height="{size_px}" viewBox="0 0 64 64" style="filter: drop-shadow(0 0 5px rgba(186, 230, 253, 0.5)); vertical-align: middle;">'
        f'<defs><linearGradient id="iceBlade" x1="0%" y1="0%" x2="100%" y2="100%">'
        f'<stop offset="0%" stop-color="#F8FAFC"/><stop offset="55%" stop-color="#BAE6FD"/><stop offset="100%" stop-color="#38BDF8"/>'
        f'</linearGradient></defs>'
        f'<path d="M32 2 C36 18 46 24 62 32 C46 36 40 46 32 62 C28 46 18 40 2 32 C18 28 24 18 32 2 Z" fill="url(#iceBlade)" stroke="#E0F2FE" stroke-width="1.5"/>'
        f'<circle cx="32" cy="32" r="7" fill="#090D16" stroke="#BAE6FD" stroke-width="2"/>'
        f'</svg>'
    )

SHURIKEN_ICON_MAIN = get_shuriken_icon_html(28)
SHURIKEN_ICON_SM = get_shuriken_icon_html(16)

# =====================================================================
# CSSスタイル (インデントなしで記述)
# =====================================================================
st.markdown("""<style>
:root {
    --ice-white: #F0F9FF;
    --ice-cyan: #BAE6FD;
    --accent-blue: #38BDF8;
    --bg-dark: #090D16;
    --bg-card: #0F172A;
    --border-muted: #1E293B;
}
header[data-testid="stHeader"] {
    height: 1.8rem !important;
    background: transparent !important;
}
.block-container {
    padding-top: 1.2rem !important;
    padding-bottom: 0.8rem !important;
    padding-left: 1.5rem !important;
    padding-right: 1.5rem !important;
}
.brand-bar {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 6px 0 10px 0;
    margin-bottom: 8px;
    border-bottom: 1px solid #1E293B;
}
.brand-title {
    font-size: 1.4rem;
    font-weight: 800;
    color: #F8FAFC;
    letter-spacing: -0.02em;
    margin: 0;
}
.brand-sub {
    font-size: 0.72rem;
    font-weight: 700;
    color: #BAE6FD;
    background: rgba(56, 189, 248, 0.12);
    border: 1px solid rgba(186, 230, 253, 0.35);
    padding: 2px 8px;
    border-radius: 4px;
    letter-spacing: 0.05em;
}
.stButton button {
    background-color: #0F172A !important;
    color: #94A3B8 !important;
    border: 1px solid #1E293B !important;
    border-bottom: 2px solid #1E293B !important;
    font-weight: 600 !important;
    padding: 6px 10px !important;
    border-radius: 6px 6px 2px 2px !important;
    font-size: 0.86rem !important;
    transition: all 0.15s ease !important;
}
.stButton button:hover {
    background-color: #1E293B !important;
    color: #F0F9FF !important;
    border-color: #334155 !important;
}
.stButton button[kind="primary"],
.stButton button[data-testid="baseButton-primary"] {
    background-color: #172554 !important;
    color: #F0F9FF !important;
    border-color: #1E3A8A !important;
    border-bottom: 2.5px solid #BAE6FD !important;
    font-weight: 700 !important;
    box-shadow: 0 2px 8px rgba(56, 189, 248, 0.15) !important;
}
div[data-testid="stAudioInput"] {
    background: #0F172A;
    border: 1px solid #1E293B;
    border-radius: 8px;
    padding: 4px 12px;
    margin-bottom: 0.4rem;
}
div[data-testid="stTextArea"] textarea {
    background-color: #090D16 !important;
    color: #F8FAFC !important;
    border: 1px solid #1E293B !important;
    font-size: 0.95rem !important;
    font-weight: 500 !important;
    border-radius: 6px !important;
    line-height: 1.5 !important;
}
.sec-header {
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 0.92rem;
    font-weight: 700;
    color: #F0F9FF;
    margin-bottom: 8px;
}
</style>""", unsafe_allow_html=True)

# =====================================================================
# ベンチマーク定義
# =====================================================================
E2_PASSAGE_BENCHMARKS = {
    "S1_stress": {
        "category": "drill",
        "x_metric": "pct_v",
        "title": "短文ドリル 1: 語彙強弱拍 (Alternating Stress & Reduction)",
        "target_text": "A photograph of a famous photographer was taken in the station.",
        "phonetics_desc": "photograph と photographer の強弱母音コントラストを測定。弱母音(/ə/)の鋭い短縮が不可欠です。",
        "mu": np.array([47.01, 58.61]),
        "cov": np.array([[1.45, 3.12], [3.12, 45.0]]),
        "x_range": (41.0, 58.0),
        "y_range": (35.0, 110.0),
        "ref_audio": "benchmarks/S1_stress/UK-Native/UK-Native_01_ArcherConver_male.wav",
        "bm_csv": "benchmarks/benchmark_results_all.csv"
    },
    "S2_clusters": {
        "category": "drill",
        "x_metric": "pct_v",
        "title": "短文ドリル 2: 子音連結保持 (Consonant Clusters)",
        "target_text": "Strict instructors asked complex questions about the project tests.",
        "phonetics_desc": "連続子音に不要な母音を挟まずに子音を保持できるかを診断します。",
        "mu": np.array([35.85, 59.12]),
        "cov": np.array([[1.80, 2.85], [2.85, 52.0]]),
        "x_range": (30.0, 46.0),
        "y_range": (35.0, 120.0),
        "ref_audio": "benchmarks/S2_clusters/US-Native/US-Native_01_RogerLaidBac_male.wav",
        "bm_csv": "benchmarks/benchmark_results_all.csv"
    },
    "S3_connected": {
        "category": "drill",
        "x_metric": "pct_v",
        "title": "短文ドリル 3: 機能語弱化 (Connected Speech Weakening)",
        "target_text": "Can you give him a cup of water from the kitchen at lunch?",
        "phonetics_desc": "機能語を極小化し、息の流れに乗せるテストです。",
        "mu": np.array([45.42, 45.18]),
        "cov": np.array([[1.60, 1.95], [1.95, 42.0]]),
        "x_range": (38.0, 54.0),
        "y_range": (30.0, 140.0),
        "ref_audio": "benchmarks/S3_connected/UK-Native/UK-Native_01_ArcherConver_male.wav",
        "bm_csv": "benchmarks/benchmark_results_all.csv"
    },
    "P1_northwind": {
        "category": "passage",
        "x_metric": "rcf",
        "title": "総合アセスメント 1: 北風と太陽 (The North Wind and the Sun 全文113語)",
        "target_text": "The North Wind and the Sun were disputing which was the stronger, when a traveler came along wrapped in a warm cloak. They agreed that the one who first succeeded in making the traveler take his cloak off should be considered stronger than the other. Then the North Wind blew with all his might, but the more he blew, the more closely did the traveler fold his cloak around him, and at last the North Wind gave up the attempt. Then the Sun shone out warmly, and immediately the traveler took off his cloak. And so the North Wind was obliged to confess that the Sun was the stronger of the two.",
        "phonetics_desc": "ポーズ非感受性の2D軸 (内容語/機能語母音比 R_CF × 弱母音持続時間 Schwa P75) により、客観的なストレス拍を評価します。",
        "mu": np.array([1.45, 52.26]),
        "cov": np.array([[0.0068, -0.165], [-0.165, 16.50]]),
        "x_range": (0.7, 1.85),
        "y_range": (35.0, 105.0),
        "ref_audio": "benchmarks/P1_northwind/UK-Native/UK-Native_01_Archer.wav",
        "bm_csv": "benchmarks/P1_northwind/northwind_benchmark_results.csv"
    }
}

DIAGNOSTIC_FUNCTION_WORDS = {
    "to", "for", "at", "of", "in", "from", "with", "into", "through", "about",
    "and", "that", "than", "but", "as", "if", "or", "so",
    "was", "were", "should", "could", "would", "can", "his", "him", "them", "us"
}

ALL_FUNCTION_WORDS = {
    "a", "an", "the", "in", "on", "at", "to", "for", "of", "from", "with", "by", "about", "as",
    "into", "like", "through", "after", "over", "between", "out", "against", "during", "without",
    "before", "under", "around", "among", "i", "you", "he", "she", "it", "we", "they", "me", "him",
    "her", "us", "them", "my", "your", "his", "its", "our", "their", "this", "that", "these",
    "those", "who", "whom", "whose", "which", "what", "where", "when", "why", "how", "am", "is",
    "are", "was", "were", "be", "been", "being", "have", "has", "had", "will", "would", "shall",
    "should", "can", "could", "may", "might", "must", "and", "but", "or", "so", "than", "off", "along"
}

VOWEL_SET = {"AA", "AE", "AH", "AO", "AW", "AY", "EH", "ER", "EY", "IH", "IY", "OW", "OY", "UH", "UW"}

# --- セッション状態 ---
if "app_mode" not in st.session_state:
    st.session_state.app_mode = "RHYTHM"
if "current_label" not in st.session_state:
    st.session_state.current_label = "お手本: Right (US)"
if "current_audio_bytes" not in st.session_state:
    init_path = os.path.join("presets", "right_us.wav")
    st.session_state.current_audio_bytes = open(init_path, "rb").read() if os.path.exists(init_path) else None
if "last_recorded_bytes" not in st.session_state:
    st.session_state.last_recorded_bytes = None
if "auto_play" not in st.session_state:
    st.session_state.auto_play = False

if "e2_category" not in st.session_state:
    st.session_state.e2_category = "passage"
if "e2_selected_passage" not in st.session_state:
    st.session_state.e2_selected_passage = "P1_northwind"
if "e2_custom_text" not in st.session_state:
    st.session_state.e2_custom_text = E2_PASSAGE_BENCHMARKS["P1_northwind"]["target_text"]
if "e2_last_analyzed_text" not in st.session_state:
    st.session_state.e2_last_analyzed_text = st.session_state.e2_custom_text
if "e2_audio_bytes" not in st.session_state:
    st.session_state.e2_audio_bytes = None
if "e2_eval_history" not in st.session_state:
    st.session_state.e2_eval_history = []
if "e2_plot_view" not in st.session_state:
    st.session_state.e2_plot_view = "scatter2d"

# =====================================================================
# 最上段ブランドヘッダー (インデントなしでレンダリング)
# =====================================================================
brand_header_html = (
    f'<div class="brand-bar">'
    f'{SHURIKEN_ICON_MAIN}'
    f'<span class="brand-title">NinjaPhonetics: Biofeedback Scope</span>'
    f'<span class="brand-sub">ACOUSTIC & PROSODY DIAGNOSTICS</span>'
    f'</div>'
)
st.markdown(brand_header_html, unsafe_allow_html=True)

tab_m1, tab_m2, _ = st.columns([1.5, 1.5, 3.0])
with tab_m1:
    if st.button("A1: 音素ホルマント・スコープ (R / L)", type="primary" if st.session_state.app_mode == "RL" else "secondary", use_container_width=True):
        st.session_state.app_mode = "RL"
        st.rerun()
with tab_m2:
    if st.button("E2: プロソディ＆ストレス拍診断", type="primary" if st.session_state.app_mode == "RHYTHM" else "secondary", use_container_width=True):
        st.session_state.app_mode = "RHYTHM"
        st.rerun()

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
# MODE 1: A1 スコープ関連
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
    return f"""<!DOCTYPE html><html><head><style>
body {{ margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: #0B0F19; color: #E2E8F0; user-select: none; overflow: hidden; }}
.toolbar {{ display: flex; align-items: center; gap: 10px; padding: 10px 14px; background: #0F172A; border: 1px solid #1E293B; border-radius: 8px 8px 0 0; }}
button {{ font-family: inherit; font-size: 13px; font-weight: bold; padding: 7px 13px; border-radius: 6px; border: 1px solid #334155; cursor: pointer; background: #1E293B; color: #E2E8F0; transition: all 0.15s; white-space: nowrap; }}
button:hover {{ background: #334155; color: #FFFFFF; }}
button.btn-play-ready {{ background: #16A34A; color: #FFFFFF; border-color: #15803D; }}
button.btn-play-Active {{ background: #DC2626; color: #FFFFFF; border-color: #B91C1C; }}
button.toggle-active {{ background: #334155; color: #F8FAFC; border-color: #64748B; }}
select {{ font-family: inherit; font-size: 13px; padding: 5px 10px; border-radius: 6px; border: 1px solid #334155; background: #1E293B; color: #E2E8F0; font-weight: bold; }}
.readout {{ font-family: inherit; font-size: 13px; font-weight: 500; margin-left: auto; font-variant-numeric: tabular-nums; color: #E2E8F0; white-space: nowrap; }}
#scope-wrapper {{ position: relative; width: 100%; height: 560px; border: 1px solid #1E293B; border-top: none; background: #090D16; }}
canvas {{ position: absolute; left: 0; top: 0; width: 100%; height: 100%; }}
#overlay-canvas {{ mix-blend-mode: screen; }}
</style></head><body>
<div class="toolbar">
  <button id="btn-play" class="btn-play-ready">Play (Space)</button>
  <button id="btn-reset">全体選択</button>
  <button id="btn-spec" class="toggle-active">Spectrogram: ON</button>
  <label style="font-size: 13px; font-weight: 500; display: flex; align-items: center; gap: 6px; color: #94A3B8;">
    Speed:
    <select id="sel-rate"><option value="1.0">1.0x</option><option value="0.7">0.7x</option><option value="0.5">0.5x</option></select>
  </label>
  <button id="btn-dl-plot">Plot (.png)</button>
  <button id="btn-dl-audio">Audio (.wav)</button>
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
let W = 1000, H = 560;
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
  audioEl.preservesPitch = true;
}}
initAudio();
function tToX(t) {{ return MARGIN.left + ((t - xMin) / (xMax - xMin)) * (W - MARGIN.left - MARGIN.right); }}
function xToT(x) {{ return Math.max(xMin, Math.min(xMax, xMin + ((x - MARGIN.left) / (W - MARGIN.left - MARGIN.right)) * (xMax - xMin))); }}
function getPanels() {{
  const totalH = H - MARGIN.top - MARGIN.bottom;
  const gap = 24, h1 = 115;
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
  audioEl.preservesPitch = true;
  const s = Math.min(selStart, selEnd), e = Math.max(selStart, selEnd);
  audioEl.currentTime = (cursorT >= s && cursorT < e - 0.02) ? cursorT : s;
  isPlaying = true; btnPlay.textContent = "Stop (Space)"; btnPlay.className = "btn-play-Active";
  audioEl.play().catch(() => stopPlayback());
  function tick() {{
    if (!isPlaying) return;
    cursorT = audioEl.currentTime;
    if (cursorT >= e || audioEl.ended) {{
      audioEl.pause(); isPlaying = false; cursorT = e;
      btnPlay.textContent = "Play (Space)"; btnPlay.className = "btn-play-ready";
      drawOverlay(); return;
    }}
    drawOverlay(); animId = requestAnimationFrame(tick);
  }}
  animId = requestAnimationFrame(tick);
}}
function stopPlayback() {{
  audioEl.pause(); if (animId) cancelAnimationFrame(animId);
  isPlaying = false; btnPlay.textContent = "Play (Space)"; btnPlay.className = "btn-play-ready";
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
</script></body></html>"""

# =====================================================================
# MODE 2: E2 ベンチマーク読み込み (高速キャッシュ利用)
# =====================================================================
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
    """キャッシュCSVのみ読み込み、重い再解析を回避"""
    cache_csv = f"benchmarks/{passage_key}/word_raw_cache.csv"
    if not os.path.exists(cache_csv):
        return {}
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
        w_start = float(w.get("start", 0.0))
        w_end = float(w.get("end", 0.0))
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

    w_aligns = align_word_characters(audio_path, norm_custom)
    phones = map_words_to_phonemes(w_aligns)
    metrics = compute_robust_metrics(phones)
    r_cf, mean_c_ms, mean_f_ms, diag_median_ms, word_breakdown = compute_content_function_metrics(w_aligns, phones)

    pct_v = metrics["pct_v"]
    schwa_p75 = metrics["schwa_p75_ms"]
    npvi = metrics["npvi_clipped"]

    cond_schwa = schwa_p75 <= 65.0
    cond_diag = diag_median_ms <= 50.0
    cond_rcf = r_cf >= 1.30

    if not is_custom_mode:
        x_val = r_cf if x_metric == "rcf" else pct_v
        user_pt = np.array([x_val, schwa_p75])
        mu = bm["mu"]
        cov = bm["cov"]
        inv_cov = np.linalg.inv(cov)
        diff = user_pt - mu
        d_m_sq = round(float(diff.T @ inv_cov @ diff), 2)

        is_strike = (d_m_sq <= 9.21) or (cond_schwa and cond_diag and cond_rcf)
        is_bullseye = (d_m_sq <= 2.0) and (diag_median_ms <= 45.0)

        if is_bullseye:
            verdict = "PERFECT STRIKE"
            status_color = "#38BDF8"
            feedback = f"理想的なストレス拍です。弱母音({schwa_p75:.1f}ms)と弱化必須語({diag_median_ms:.1f}ms)がネイティブ重心に完全一致しています。"
        elif is_strike:
            verdict = "NATIVE STRIKE"
            status_color = "#10B981"
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
            verdict = "DYNAMIC STRIKE (自由文合格)"
            status_color = "#10B981"
            is_strike = True
            is_bullseye = (diag_median_ms <= 42.0 and r_cf >= 1.50)
            feedback = f"自由文で理想的なストレス拍です (V_diag = {diag_median_ms:.1f}ms, R_CF = {r_cf}x, nPVI = {npvi:.1f})。"
        elif schwa_p75 <= 75.0 and diag_median_ms <= 60.0:
            verdict = "PASSABLE RHYTHM (惜しい)"
            status_color = "#F59E0B"
            is_strike = False
            is_bullseye = False
            feedback = f"リズム骨格は良好ですが、弱化必須語(現在 {diag_median_ms:.0f}ms)をもう一段短く(目標 ≤45ms)できます。"
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

# =====================================================================
# Canvas描画生成 (HTML)
# =====================================================================
def build_e2_multiview_canvas_html(passage_key: str, eval_history: list = None, is_custom_mode: bool = False, view_mode: str = "scatter2d", word_bench_dist: dict = None) -> str:
    bm = E2_PASSAGE_BENCHMARKS[passage_key]
    x_metric = bm.get("x_metric", "pct_v")
    mu = bm["mu"]
    cov = bm["cov"]

    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]

    n_std_zone = 3.035
    v1_95 = vecs[:, 0] * (n_std_zone * np.sqrt(max(vals[0], 0)))
    v2_95 = vecs[:, 1] * (n_std_zone * np.sqrt(max(vals[1], 0)))

    n_std_ci = 2.447 / np.sqrt(12)
    v1_ci = vecs[:, 0] * (n_std_ci * np.sqrt(max(vals[0], 0)))
    v2_ci = vecs[:, 1] * (n_std_ci * np.sqrt(max(vals[1], 0)))

    bench_points = load_benchmark_points(bm["bm_csv"], passage_key) if not is_custom_mode else []
    history_list = eval_history if eval_history is not None else []

    if is_custom_mode:
        x_min, x_max = (0.7, 1.9) if x_metric == "rcf" else (30.0, 58.0)
        y_min, y_max = 30.0, 115.0
    else:
        x_min, x_max = bm["x_range"][0], bm["x_range"][1]
        y_min, y_max = bm["y_range"][0], bm["y_range"][1]

    user_word_raw = {}
    if history_list:
        for ev in history_list:
            for wb in ev.get("word_breakdown", []):
                if wb.get("is_diag"):
                    cw = wb["clean_word"]
                    user_word_raw.setdefault(cw, []).append(wb["vowel_ms"])

    all_diag_words_present = sorted(list(set(list((word_bench_dist or {}).keys()) + list(user_word_raw.keys()))))

    canvas_data = {
        "passage_key": passage_key,
        "view_mode": view_mode,
        "x_metric": x_metric,
        "is_custom_mode": is_custom_mode,
        "mu_x": float(mu[0]), "mu_y": float(mu[1]),
        "v1_95": [float(v1_95[0]), float(v1_95[1])],
        "v2_95": [float(v2_95[0]), float(v2_95[1])],
        "v1_ci": [float(v1_ci[0]), float(v1_ci[1])],
        "v2_ci": [float(v2_ci[0]), float(v2_ci[1])],
        "x_min": x_min, "x_max": x_max,
        "y_min": y_min, "y_max": y_max,
        "history": history_list,
        "benchmark": bench_points,
        "words_alpha": all_diag_words_present,
        "word_bench": word_bench_dist or {},
        "user_word_raw": user_word_raw
    }

    view_titles = {
        "scatter2d": f"2D ストライクゾーン (試行回数: {len(history_list)})",
        "word_swarm": "弱化必須語 生データ分布 (Swarm + Box | アルファベット順)",
        "metrics_1d": "各プロソディ指標の生データ分布比較 (Swarm ストリップ)"
    }
    header_title = view_titles.get(view_mode, "2D ストライクゾーン")

    return f"""<!DOCTYPE html><html><head><style>
body {{ margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: #0B0F19; color: #E2E8F0; user-select: none; overflow: hidden; }}
.toolbar {{ display: flex; align-items: center; justify-content: space-between; padding: 6px 12px; background: #0F172A; border: 1px solid #1E293B; border-radius: 8px 8px 0 0; }}
.toolbar-title {{ font-size: 13px; font-weight: 700; color: #BAE6FD; display: flex; align-items: center; gap: 8px; }}
.btn-group {{ display: flex; align-items: center; gap: 8px; }}
button {{ font-family: inherit; font-size: 12px; font-weight: bold; padding: 4px 11px; border-radius: 5px; border: 1px solid #334155; cursor: pointer; background: #1E293B; color: #E2E8F0; transition: all 0.15s; }}
button:hover {{ background: #334155; color: #FFFFFF; }}
button.toggle-active {{ background: #172554; color: #BAE6FD; border-color: #38BDF8; }}
#scope-wrapper {{ position: relative; width: 100%; height: 420px; background: #090D16; border: 1px solid #1E293B; border-top: none; border-radius: 0 0 8px 8px; }}
canvas {{ position: absolute; left: 0; top: 0; width: 100%; height: 100%; }}
</style></head><body>
<div class="toolbar">
  <div class="toolbar-title"><span>{header_title}</span></div>
  <div class="btn-group">
    <button id="btn-toggle-bm" class="toggle-active" style="display: {'none' if is_custom_mode else 'inline-block'};">Benchmark: ON</button>
    <button id="btn-dl-radar">Plot (.png)</button>
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
let W = 800, H = 420;

function quantile(arr, q) {{
  if (!arr || arr.length === 0) return 0;
  const sorted = [...arr].sort((a, b) => a - b);
  const pos = (sorted.length - 1) * q;
  const base = Math.floor(pos);
  const rest = pos - base;
  if (sorted[base + 1] !== undefined) return sorted[base] + rest * (sorted[base + 1] - sorted[base]);
  return sorted[base];
}}

function draw() {{
  const dpr = window.devicePixelRatio || 1;
  W = wrapper.clientWidth; H = wrapper.clientHeight;
  canvas.width = W * dpr; canvas.height = H * dpr;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.fillStyle = "#090D16"; ctx.fillRect(0, 0, W, H);
  if (D.view_mode === "word_swarm") drawWordSwarmBox();
  else if (D.view_mode === "metrics_1d") drawMetrics1DSwarm();
  else drawScatter2D();
}}

function drawScatter2D() {{
  const MARGIN = {{ left: 78, right: 28, top: 20, bottom: 46 }};
  const plotW = W - MARGIN.left - MARGIN.right;
  const plotH = H - MARGIN.top - MARGIN.bottom;
  const isRcf = (D.x_metric === "rcf");
  function xToPx(x) {{ return MARGIN.left + ((x - D.x_min) / (D.x_max - D.x_min)) * plotW; }}
  function yToPx(y) {{ return (H - MARGIN.bottom) - ((y - D.y_min) / (D.y_max - D.y_min)) * plotH; }}

  ctx.strokeStyle = "rgba(148, 163, 184, 0.10)"; ctx.lineWidth = 1.0;
  const stepX = isRcf ? 0.1 : ((D.x_max - D.x_min) > 20 ? 4 : 2);
  const firstTickX = Math.ceil(D.x_min / stepX) * stepX;
  for (let x = firstTickX; x <= D.x_max + 1e-5; x += stepX) {{
    const px = xToPx(x);
    ctx.beginPath(); ctx.moveTo(px, MARGIN.top); ctx.lineTo(px, H - MARGIN.bottom); ctx.stroke();
    ctx.fillStyle = "#64748B"; ctx.font = "12px sans-serif"; ctx.textAlign = "center";
    ctx.fillText(isRcf ? x.toFixed(1) + "x" : x.toFixed(0) + "%", px, H - MARGIN.bottom + 18);
  }}
  for (let y = Math.ceil(D.y_min / 10) * 10; y <= D.y_max; y += 10) {{
    const py = yToPx(y);
    ctx.beginPath(); ctx.moveTo(MARGIN.left, py); ctx.lineTo(W - MARGIN.right, py); ctx.stroke();
    ctx.fillStyle = "#64748B"; ctx.font = "12px sans-serif"; ctx.textAlign = "right"; ctx.textBaseline = "middle";
    ctx.fillText(y + "ms", MARGIN.left - 8, py);
  }}

  ctx.strokeStyle = "#334155"; ctx.lineWidth = 1.5;
  ctx.strokeRect(MARGIN.left, MARGIN.top, plotW, plotH);
  ctx.fillStyle = "#CBD5E1"; ctx.font = "bold 13px sans-serif"; ctx.textAlign = "center";
  ctx.fillText(isRcf ? "内容語 / 機能語 母音持続時間比 (R_CF)" : "母音占有率 (%V) [%]", MARGIN.left + plotW / 2, H - 7);
  ctx.save(); ctx.translate(20, MARGIN.top + plotH / 2); ctx.rotate(-Math.PI / 2);
  ctx.fillText("弱母音持続時間 (Schwa P75) [ms]", 0, 0); ctx.restore();

  ctx.save();
  ctx.beginPath(); ctx.rect(MARGIN.left, MARGIN.top, plotW, plotH); ctx.clip();

  function drawEllipse(mu_x, mu_y, v1, v2) {{
    ctx.beginPath();
    const N_PTS = 64;
    for (let i = 0; i <= N_PTS; i++) {{
      const theta = (i / N_PTS) * 2 * Math.PI;
      const dx = mu_x + Math.cos(theta) * v1[0] + Math.sin(theta) * v2[0];
      const dy = mu_y + Math.cos(theta) * v1[1] + Math.sin(theta) * v2[1];
      const px = xToPx(dx), py = yToPx(dy);
      if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
    }}
    ctx.closePath();
  }}

  if (!D.is_custom_mode) {{
    drawEllipse(D.mu_x, D.mu_y, D.v1_95, D.v2_95);
    ctx.fillStyle = "rgba(16, 185, 129, 0.13)"; ctx.fill();
    ctx.strokeStyle = "#10B981"; ctx.lineWidth = 2.0; ctx.setLineDash([5, 4]); ctx.stroke();

    drawEllipse(D.mu_x, D.mu_y, D.v1_ci, D.v2_ci);
    ctx.fillStyle = "rgba(56, 189, 248, 0.30)"; ctx.fill();
    ctx.strokeStyle = "#38BDF8"; ctx.lineWidth = 1.2; ctx.setLineDash([]); ctx.stroke();

    if (showBenchmark && D.benchmark && D.benchmark.length > 0) {{
      D.benchmark.forEach(b => {{
        const bxVal = isRcf ? b.rcf : b.pct_v;
        const bx = xToPx(bxVal), by = yToPx(b.schwa_p75_ms);
        ctx.save(); ctx.globalAlpha = 0.85; ctx.fillStyle = b.color;
        ctx.strokeStyle = "#090D16"; ctx.lineWidth = 1.2;
        ctx.beginPath(); ctx.arc(bx, by, 5.2, 0, 2 * Math.PI); ctx.fill(); ctx.stroke(); ctx.restore();
      }});

      const legX = W - MARGIN.right - 150, legY = MARGIN.top + 8;
      ctx.fillStyle = "rgba(9, 13, 22, 0.88)"; ctx.fillRect(legX, legY, 142, 68);
      ctx.strokeStyle = "#1E293B"; ctx.strokeRect(legX, legY, 142, 68);
      ctx.font = "10.5px sans-serif"; ctx.textAlign = "left"; ctx.textBaseline = "middle";
      const items = [
        ["#38BDF8", "US-Native (米)"],
        ["#10B981", "UK-Native (英)"],
        ["#F59E0B", "JP-Accented"],
        ["#EF4444", "JP-Mora (等時拍)"]
      ];
      items.forEach((it, i) => {{
        const iy = legY + 12 + i * 15;
        ctx.fillStyle = it[0]; ctx.beginPath(); ctx.arc(legX + 12, iy, 4.0, 0, 2 * Math.PI); ctx.fill();
        ctx.fillStyle = "#CBD5E1"; ctx.fillText(it[1], legX + 22, iy);
      }});
    }}
  }} else {{
    const zx1 = xToPx(isRcf ? 1.30 : 35.0), zx2 = xToPx(isRcf ? 1.85 : 49.0);
    const zy1 = yToPx(65.0), zy2 = yToPx(35.0);
    ctx.fillStyle = "rgba(16, 185, 129, 0.14)"; ctx.fillRect(zx1, zy1, zx2 - zx1, zy2 - zy1);
    ctx.strokeStyle = "#10B981"; ctx.lineWidth = 2.0; ctx.setLineDash([6, 4]); ctx.strokeRect(zx1, zy1, zx2 - zx1, zy2 - zy1); ctx.setLineDash([]);
    ctx.fillStyle = "#10B981"; ctx.font = "bold 12px sans-serif"; ctx.textAlign = "left";
    ctx.fillText("ストレス拍合格領域 (R_CF ≥ 1.3x, Schwa ≤ 65ms)", zx1 + 8, zy1 + 16);
  }}

  const total = D.history.length;
  if (total > 0) {{
    if (total > 1) {{
      ctx.strokeStyle = "rgba(240, 249, 255, 0.45)"; ctx.lineWidth = 1.8; ctx.setLineDash([3, 3]); ctx.beginPath();
      for (let i = 0; i < total; i++) {{
        const hxVal = isRcf ? D.history[i].r_cf : D.history[i].pct_v;
        const px = xToPx(hxVal), py = yToPx(D.history[i].schwa_p75_ms);
        if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
      }}
      ctx.stroke(); ctx.setLineDash([]);
    }}

    D.history.forEach((pt, idx) => {{
      const isLatest = (idx === total - 1);
      const hxVal = isRcf ? pt.r_cf : pt.pct_v;
      const px = xToPx(hxVal), py = yToPx(pt.schwa_p75_ms);
      const alpha = total === 1 ? 1.0 : 0.45 + (idx / (total - 1)) * 0.55;
      ctx.save(); ctx.globalAlpha = alpha;
      if (isLatest) {{
        ctx.fillStyle = pt.status_color + "44"; ctx.beginPath(); ctx.arc(px, py, 16, 0, 2 * Math.PI); ctx.fill();
        ctx.fillStyle = pt.status_color; ctx.strokeStyle = "#F0F9FF"; ctx.lineWidth = 2.5; ctx.beginPath(); ctx.arc(px, py, 7.5, 0, 2 * Math.PI); ctx.fill(); ctx.stroke();
        ctx.fillStyle = "#F0F9FF"; ctx.font = "bold 12px sans-serif"; ctx.textAlign = px > W - 200 ? "right" : "left";
        const xStr = isRcf ? `${{pt.r_cf}}x` : `${{pt.pct_v}}%`;
        ctx.fillText(`#${{idx+1}} (${{xStr}}, ${{pt.schwa_p75_ms}}ms | V_diag:${{pt.diag_median_ms}}ms)`, px + (px > W - 200 ? -12 : 12), py - 10);
      }} else {{
        ctx.fillStyle = pt.status_color; ctx.strokeStyle = "#F0F9FF"; ctx.lineWidth = 1.5; ctx.beginPath(); ctx.arc(px, py, 5.5, 0, 2 * Math.PI); ctx.fill(); ctx.stroke();
        ctx.fillStyle = "#E2E8F0"; ctx.font = "bold 10px sans-serif"; ctx.textAlign = "center"; ctx.fillText(`#${{idx+1}}`, px, py - 9);
      }}
      ctx.restore();
    }});
  }}
  ctx.restore();
}}

function drawWordSwarmBox() {{
  const MARGIN = {{ left: 68, right: 24, top: 24, bottom: 52 }};
  const plotW = W - MARGIN.left - MARGIN.right;
  const plotH = H - MARGIN.top - MARGIN.bottom;
  const words = D.words_alpha || [];
  const yMin = 0, yMax = 155;
  function yToPx(y) {{ return (H - MARGIN.bottom) - ((Math.min(y, yMax) - yMin) / (yMax - yMin)) * plotH; }}

  for (let y = 0; y <= 150; y += 25) {{
    const py = yToPx(y);
    ctx.strokeStyle = "rgba(148, 163, 184, 0.12)"; ctx.lineWidth = 1.0;
    ctx.beginPath(); ctx.moveTo(MARGIN.left, py); ctx.lineTo(W - MARGIN.right, py); ctx.stroke();
    ctx.fillStyle = "#64748B"; ctx.font = "11.5px sans-serif"; ctx.textAlign = "right"; ctx.textBaseline = "middle";
    ctx.fillText(y + "ms", MARGIN.left - 8, py);
  }}

  const y45 = yToPx(45), y65 = yToPx(65);
  ctx.strokeStyle = "rgba(16, 185, 129, 0.7)"; ctx.lineWidth = 1.4; ctx.setLineDash([5, 4]);
  ctx.beginPath(); ctx.moveTo(MARGIN.left, y45); ctx.lineTo(W - MARGIN.right, y45); ctx.stroke();
  ctx.strokeStyle = "rgba(239, 68, 68, 0.7)";
  ctx.beginPath(); ctx.moveTo(MARGIN.left, y65); ctx.lineTo(W - MARGIN.right, y65); ctx.stroke();
  ctx.setLineDash([]);
  ctx.strokeStyle = "#334155"; ctx.lineWidth = 1.5;
  ctx.strokeRect(MARGIN.left, MARGIN.top, plotW, plotH);
  ctx.fillStyle = "#CBD5E1"; ctx.font = "bold 12.5px sans-serif"; ctx.textAlign = "center";
  ctx.fillText("弱化必須語 (アルファベット順: 緑箱=Native / 赤箱=JP-Mora / 白枠星点=あなた)", MARGIN.left + plotW / 2, H - 8);
  ctx.save(); ctx.translate(18, MARGIN.top + plotH / 2); ctx.rotate(-Math.PI / 2);
  ctx.fillText("母音持続時間 [ms]", 0, 0); ctx.restore();

  if (words.length === 0) return;
  const colW = plotW / words.length;

  words.forEach((w, idx) => {{
    const cx = MARGIN.left + (idx + 0.5) * colW;
    ctx.fillStyle = "#F0F9FF"; ctx.font = "bold 12px sans-serif"; ctx.textAlign = "center";
    ctx.fillText(w, cx, H - MARGIN.bottom + 18);
    const bInfo = D.word_bench[w] || {{ native: [], mora: [], us: [], uk: [] }};
    const uPts = D.user_word_raw[w] || [];

    if (showBenchmark && bInfo.native && bInfo.native.length > 0) {{
      const nx = cx - colW * 0.20;
      const q25 = quantile(bInfo.native, 0.25), med = quantile(bInfo.native, 0.5), q75 = quantile(bInfo.native, 0.75);
      const boxW = Math.min(14, colW * 0.22);
      ctx.fillStyle = "rgba(16, 185, 129, 0.28)"; ctx.strokeStyle = "#10B981"; ctx.lineWidth = 1.2;
      const topY = yToPx(q75), botY = yToPx(q25);
      ctx.fillRect(nx - boxW/2, topY, boxW, Math.max(2, botY - topY));
      ctx.strokeRect(nx - boxW/2, topY, boxW, Math.max(2, botY - topY));
      ctx.beginPath(); ctx.moveTo(nx - boxW/2, yToPx(med)); ctx.lineTo(nx + boxW/2, yToPx(med)); ctx.stroke();
      (bInfo.us || []).forEach((v, j) => {{
        const jx = nx + ((j % 3) - 1) * 3.0;
        ctx.fillStyle = "#38BDF8"; ctx.beginPath(); ctx.arc(jx, yToPx(v), 2.8, 0, 2 * Math.PI); ctx.fill();
      }});
      (bInfo.uk || []).forEach((v, j) => {{
        const jx = nx + (((j + 1) % 3) - 1) * 3.0;
        ctx.fillStyle = "#10B981"; ctx.beginPath(); ctx.arc(jx, yToPx(v), 2.8, 0, 2 * Math.PI); ctx.fill();
      }});
    }}

    if (showBenchmark && bInfo.mora && bInfo.mora.length > 0) {{
      const mx = cx + colW * 0.20;
      const q25 = quantile(bInfo.mora, 0.25), med = quantile(bInfo.mora, 0.5), q75 = quantile(bInfo.mora, 0.75);
      const boxW = Math.min(14, colW * 0.22);
      ctx.fillStyle = "rgba(239, 68, 68, 0.22)"; ctx.strokeStyle = "#EF4444"; ctx.lineWidth = 1.2;
      const topY = yToPx(q75), botY = yToPx(q25);
      ctx.fillRect(mx - boxW/2, topY, boxW, Math.max(2, botY - topY));
      ctx.strokeRect(mx - boxW/2, topY, boxW, Math.max(2, botY - topY));
      ctx.beginPath(); ctx.moveTo(mx - boxW/2, yToPx(med)); ctx.lineTo(mx + boxW/2, yToPx(med)); ctx.stroke();
      bInfo.mora.forEach((v, j) => {{
        const jx = mx + ((j % 3) - 1) * 3.0;
        ctx.fillStyle = "#EF4444"; ctx.beginPath(); ctx.arc(jx, yToPx(v), 2.8, 0, 2 * Math.PI); ctx.fill();
      }});
    }}

    uPts.forEach((v, j) => {{
      const ux = cx + ((j % 2 === 0 ? 1 : -1) * (Math.floor(j / 2) * 3.5));
      const uy = yToPx(v);
      const col = v <= 48 ? "#10B981" : (v <= 65 ? "#F59E0B" : "#EF4444");
      ctx.fillStyle = col; ctx.strokeStyle = "#F0F9FF"; ctx.lineWidth = 1.8;
      ctx.beginPath(); ctx.arc(ux, uy, 5.2, 0, 2 * Math.PI); ctx.fill(); ctx.stroke();
    }});
  }});
}}

function drawMetrics1DSwarm() {{
  const MARGIN = {{ left: 165, right: 35, top: 28, bottom: 40 }};
  const plotW = W - MARGIN.left - MARGIN.right;
  const plotH = H - MARGIN.top - MARGIN.bottom;
  const tracks = [
    {{ key: "schwa_p75_ms", label: "弱母音長 Schwa P75 [ms]", min: 35, max: 105, target: "≤ 60ms" }},
    {{ key: "rcf", label: "内容/機能語比 R_CF [倍]", min: 0.7, max: 1.85, target: "≥ 1.35x" }},
    {{ key: "npvi", label: "母音伸縮指数 nPVI-V", min: 35, max: 75, target: "42〜58" }}
  ];
  const rowH = plotH / tracks.length;

  tracks.forEach((tr, idx) => {{
    const cy = MARGIN.top + (idx + 0.5) * rowH;
    ctx.fillStyle = "#F0F9FF"; ctx.font = "bold 12px sans-serif"; ctx.textAlign = "right"; ctx.textBaseline = "middle";
    ctx.fillText(tr.label, MARGIN.left - 14, cy - 7);
    ctx.fillStyle = "#64748B"; ctx.font = "11px sans-serif";
    ctx.fillText("目標: " + tr.target, MARGIN.left - 14, cy + 9);
    ctx.strokeStyle = "#334155"; ctx.lineWidth = 2.0;
    ctx.beginPath(); ctx.moveTo(MARGIN.left, cy); ctx.lineTo(W - MARGIN.right, cy); ctx.stroke();

    function valToX(v) {{
      const cl = Math.max(tr.min, Math.min(tr.max, v));
      return MARGIN.left + ((cl - tr.min) / (tr.max - tr.min)) * plotW;
    }}

    if (showBenchmark && D.benchmark) {{
      D.benchmark.forEach((b, j) => {{
        const v = b[tr.key];
        if (v === undefined) return;
        const px = valToX(v);
        const py = cy + ((j % 5) - 2) * 3.5;
        ctx.fillStyle = b.color; ctx.globalAlpha = 0.75;
        ctx.beginPath(); ctx.arc(px, py, 4.5, 0, 2 * Math.PI); ctx.fill();
        ctx.globalAlpha = 1.0;
      }});
    }}

    D.history.forEach((h, i) => {{
      const v = tr.key === "npvi" ? h.npvi_clipped : h[tr.key];
      if (v === undefined) return;
      const px = valToX(v);
      ctx.fillStyle = h.status_color; ctx.strokeStyle = "#F0F9FF"; ctx.lineWidth = 2.2;
      ctx.beginPath(); ctx.arc(px, cy, 7.5, 0, 2 * Math.PI); ctx.fill(); ctx.stroke();
      ctx.fillStyle = "#F0F9FF"; ctx.font = "bold 11px sans-serif"; ctx.textAlign = "center";
      ctx.fillText(`#${{i+1}}: ${{v}}`, px, cy - 14);
    }});
  }});
}}

if (btnToggleBm) {{
  btnToggleBm.addEventListener("click", () => {{
    showBenchmark = !showBenchmark;
    btnToggleBm.textContent = showBenchmark ? "Benchmark: ON" : "Benchmark: OFF";
    btnToggleBm.classList.toggle("toggle-active", showBenchmark);
    draw();
  }});
}}
btnDl.addEventListener("click", () => {{
  const link = document.createElement("a");
  link.download = `NinjaPhonetics_E2_${{D.passage_key}}_${{D.view_mode}}.png`; link.href = canvas.toDataURL("image/png"); link.click();
}});
window.addEventListener("resize", draw);
draw();
</script></body></html>"""

# =====================================================================
# 単語区間クリック再生付き「発話実測マップ」 (HTML)
# =====================================================================
def build_clickable_word_map_html(word_breakdown: list, audio_bytes: bytes, trial_no: int) -> str:
    wav_b64 = base64.b64encode(audio_bytes).decode("utf-8") if audio_bytes else ""
    badges_html = ""
    for idx, wb in enumerate(word_breakdown):
        w = wb["word"]
        v_ms = wb["vowel_ms"]
        s_t = wb.get("start", 0.0)
        e_t = wb.get("end", 0.0)

        if wb.get("is_diag"):
            if v_ms <= 48.0:
                style = "background:#064E3B; color:#A7F3D0; border:1px solid #10B981;"
                sub_col = "#34D399"
            elif v_ms <= 65.0:
                style = "background:#78350F; color:#FDE68A; border:1px solid #F59E0B;"
                sub_col = "#FBBF24"
            else:
                style = "background:#7F1D1D; color:#FECACA; border:1px solid #EF4444;"
                sub_col = "#F87171"
            badges_html += (
                f'<span class="w-token diag-token" style="{style}" onclick="playSlice({s_t}, {e_t}, this)" '
                f'title="クリックで再生 ({s_t:.2f}s - {e_t:.2f}s)">'
                f'{w} <small style="color:{sub_col}; font-weight:700;">{v_ms:.0f}ms</small></span>'
            )
        else:
            badges_html += (
                f'<span class="w-token plain-token" onclick="playSlice({s_t}, {e_t}, this)" '
                f'title="クリックで再生 ({w}: 母音 {v_ms:.0f}ms)">'
                f'{w}</span>'
            )

    return f"""<!DOCTYPE html><html><head><style>
body {{ margin: 0; padding: 0; background: transparent; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; color: #E2E8F0; }}
.map-box {{ background: #0F172A; border: 1px solid #1E293B; border-left: 3px solid #BAE6FD; border-radius: 8px; padding: 8px 12px; }}
.map-hdr {{ display: flex; justify-content: space-between; align-items: center; font-size: 12px; font-weight: 700; color: #BAE6FD; margin-bottom: 6px; }}
.map-sub {{ font-size: 11px; font-weight: 500; color: #94A3B8; }}
.tokens-wrap {{ line-height: 2.1; max-height: 105px; overflow-y: auto; padding-right: 4px; }}
.w-token {{ display: inline-block; padding: 1px 5px; margin-right: 3px; margin-bottom: 2px; border-radius: 4px; font-size: 13px; cursor: pointer; transition: all 0.12s; user-select: none; }}
.diag-token {{ font-weight: 700; }}
.plain-token {{ color: #CBD5E1; background: transparent; border: 1px solid transparent; }}
.plain-token:hover {{ background: #1E293B; border-color: #334155; color: #FFFFFF; }}
.diag-token:hover {{ filter: brightness(1.2); transform: translateY(-1px); }}
.playing-now {{ outline: 2px solid #BAE6FD !important; box-shadow: 0 0 8px rgba(186, 230, 253, 0.6); }}
</style></head><body>
<div class="map-box">
  <div class="map-hdr">
    <span>発話実測マップ (Trial #{trial_no} | 弱化必須語のみ色分け: 緑 ≤48ms / 黄 49-65ms / 赤 >65ms)</span>
    <span class="map-sub">各単語をクリックすると該当区間の音声を再生</span>
  </div>
  <div class="tokens-wrap">{badges_html}</div>
</div>
<script>
const WAV_B64 = "{wav_b64}";
let audioCtx = null;
let audioBuffer = null;
let currentSource = null;

async function ensureAudioDecoded() {{
  if (!WAV_B64) return false;
  if (!audioCtx) audioCtx = new (window.AudioContext || window.webkitAudioContext)();
  if (audioCtx.state === "suspended") await audioCtx.resume();
  if (!audioBuffer) {{
    const raw = atob(WAV_B64);
    const buf = new ArrayBuffer(raw.length);
    const view = new Uint8Array(buf);
    for (let i = 0; i < raw.length; i++) view[i] = raw.charCodeAt(i);
    audioBuffer = await audioCtx.decodeAudioData(buf);
  }}
  return true;
}}

async function playSlice(startSec, endSec, el) {{
  const ok = await ensureAudioDecoded();
  if (!ok || !audioBuffer) return;
  if (currentSource) {{
    try {{ currentSource.stop(); }} catch(e) {{}}
  }}
  document.querySelectorAll('.w-token').forEach(t => t.classList.remove('playing-now'));
  el.classList.add('playing-now');
  const padStart = Math.max(0, startSec - 0.035);
  const padEnd = Math.min(audioBuffer.duration, Math.max(endSec + 0.045, startSec + 0.12));
  const dur = Math.max(0.08, padEnd - padStart);
  const source = audioCtx.createBufferSource();
  source.buffer = audioBuffer;
  source.connect(audioCtx.destination);
  source.start(0, padStart, dur);
  currentSource = source;
  source.onended = () => el.classList.remove('playing-now');
}}
</script></body></html>"""

# =====================================================================
# メイン画面ルーティング
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
                f"<div class='sec-header'>{SHURIKEN_ICON_SM} 表示中: <span style='color: #F8FAFC;'>{st.session_state.current_label}</span></div>",
                unsafe_allow_html=True
            )
        clean_export_name = re.sub(r'[^a-zA-Z0-9_\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF]', '_', st.session_state.current_label)
        clean_export_name = re.sub(r'_+', '_', clean_export_name).strip('_')

        scope_data = extract_scope_data_from_bytes(st.session_state.current_audio_bytes)
        html_code = build_spike2_html(scope_data, auto_play=st.session_state.auto_play, export_name=clean_export_name or "sample")
        st.session_state.auto_play = False
        components.html(html_code, height=620, scrolling=False)

else:
    # E2 モード画面
    sub_t1, sub_t2, _ = st.columns([1.4, 1.4, 3.2])
    with sub_t1:
        if st.button("短文フォーカス・ドリル (S1〜S3)", type="primary" if st.session_state.e2_category == "drill" else "secondary", use_container_width=True):
            if st.session_state.e2_category != "drill":
                st.session_state.e2_category = "drill"
                st.session_state.e2_selected_passage = "S1_stress"
                target_txt = E2_PASSAGE_BENCHMARKS["S1_stress"]["target_text"]
                st.session_state.e2_custom_text = target_txt
                st.session_state.e2_last_analyzed_text = target_txt
                st.session_state["e2_text_input_widget"] = target_txt
                st.session_state.e2_eval_history = []
                st.session_state.e2_audio_bytes = None
                st.rerun()
    with sub_t2:
        if st.button("総合アセスメント (長文・標準規格)", type="primary" if st.session_state.e2_category == "passage" else "secondary", use_container_width=True):
            if st.session_state.e2_category != "passage":
                st.session_state.e2_category = "passage"
                st.session_state.e2_selected_passage = "P1_northwind"
                target_txt = E2_PASSAGE_BENCHMARKS["P1_northwind"]["target_text"]
                st.session_state.e2_custom_text = target_txt
                st.session_state.e2_last_analyzed_text = target_txt
                st.session_state["e2_text_input_widget"] = target_txt
                st.session_state.e2_eval_history = []
                st.session_state.e2_audio_bytes = None
                st.rerun()

    if st.session_state.e2_category == "drill":
        d_col1, d_col2, d_col3, _ = st.columns([1.2, 1.2, 1.2, 2.4])
        with d_col1:
            if st.button("S1: 語彙強弱拍", type="primary" if st.session_state.e2_selected_passage == "S1_stress" else "secondary", use_container_width=True):
                st.session_state.e2_selected_passage = "S1_stress"
                target_txt = E2_PASSAGE_BENCHMARKS["S1_stress"]["target_text"]
                st.session_state.e2_custom_text = target_txt
                st.session_state.e2_last_analyzed_text = target_txt
                st.session_state["e2_text_input_widget"] = target_txt
                st.session_state.e2_eval_history = []
                st.rerun()
        with d_col2:
            if st.button("S2: 子音連結保持", type="primary" if st.session_state.e2_selected_passage == "S2_clusters" else "secondary", use_container_width=True):
                st.session_state.e2_selected_passage = "S2_clusters"
                target_txt = E2_PASSAGE_BENCHMARKS["S2_clusters"]["target_text"]
                st.session_state.e2_custom_text = target_txt
                st.session_state.e2_last_analyzed_text = target_txt
                st.session_state["e2_text_input_widget"] = target_txt
                st.session_state.e2_eval_history = []
                st.rerun()
        with d_col3:
            if st.button("S3: 機能語弱化", type="primary" if st.session_state.e2_selected_passage == "S3_connected" else "secondary", use_container_width=True):
                st.session_state.e2_selected_passage = "S3_connected"
                target_txt = E2_PASSAGE_BENCHMARKS["S3_connected"]["target_text"]
                st.session_state.e2_custom_text = target_txt
                st.session_state.e2_last_analyzed_text = target_txt
                st.session_state["e2_text_input_widget"] = target_txt
                st.session_state.e2_eval_history = []
                st.rerun()
    else:
        p_col1, p_col2, p_col3, _ = st.columns([1.3, 1.3, 1.3, 2.1])
        with p_col1:
            if st.button("P1: 北風と太陽 (113語)", type="primary" if st.session_state.e2_selected_passage == "P1_northwind" else "secondary", use_container_width=True):
                st.session_state.e2_selected_passage = "P1_northwind"
                target_txt = E2_PASSAGE_BENCHMARKS["P1_northwind"]["target_text"]
                st.session_state.e2_custom_text = target_txt
                st.session_state.e2_last_analyzed_text = target_txt
                st.session_state["e2_text_input_widget"] = target_txt
                st.session_state.e2_eval_history = []
                st.rerun()
        with p_col2:
            st.button("P2: 虹のパッセージ", disabled=True, use_container_width=True)
        with p_col3:
            st.button("P3: 祖父のパッセージ", disabled=True, use_container_width=True)

    active_bm = E2_PASSAGE_BENCHMARKS[st.session_state.e2_selected_passage]

    txt_col1, txt_col2 = st.columns([4.8, 1.0], vertical_alignment="bottom")
    with txt_col1:
        text_height = 85 if st.session_state.e2_category == "passage" else 55
        user_target_text = st.text_area(
            "課題テキスト (自由に書き換え可能)",
            value=st.session_state.e2_custom_text,
            height=text_height,
            key="e2_text_input_widget"
        )
    with txt_col2:
        if st.button("例文に戻す", use_container_width=True):
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

    card_col1, card_col2 = st.columns([1.45, 1.05])

    with card_col1:
        # マイク録音入力
        e2_input = st.audio_input("この英文を録音してプロソディ診断を実行")

        # 音声ファイルアップロード入力 (m4a, mp3, wav 対応)
        e2_uploaded_file = st.file_uploader(
            "または音声ファイルをアップロード (WAV / MP3 / M4A)",
            type=["wav", "mp3", "m4a"],
            key="e2_file_uploader_widget"
        )

        # どちらか新しく入力された方のバイトデータを取得
        incoming_audio_bytes = None
        if e2_input is not None:
            raw_rec = e2_input.read()
            if raw_rec != st.session_state.get("e2_last_rec_bytes"):
                st.session_state["e2_last_rec_bytes"] = raw_rec
                incoming_audio_bytes = raw_rec
        elif e2_uploaded_file is not None:
            raw_upl = e2_uploaded_file.read()
            if raw_upl != st.session_state.get("e2_last_upl_bytes"):
                st.session_state["e2_last_upl_bytes"] = raw_upl
                incoming_audio_bytes = raw_upl

        # 音声データが更新された場合にアライメント＆プロソディ解析を実行
        if incoming_audio_bytes is not None and incoming_audio_bytes != st.session_state.e2_audio_bytes:
            st.session_state.e2_audio_bytes = incoming_audio_bytes
            
            # 形式を問わず一度 pydub 経由で一時標準 WAV として保存
            audio_arr, sr = decode_audio_bytes(incoming_audio_bytes)
            tmp_wav = "tmp_e2_eval.wav"
            sf.write(tmp_wav, audio_arr, sr, format="WAV")

            eval_res = evaluate_user_rhythm(
                tmp_wav,
                st.session_state.e2_selected_passage,
                st.session_state.e2_custom_text
            )
            if os.path.exists(tmp_wav):
                os.remove(tmp_wav)
            if "error" not in eval_res:
                st.session_state.e2_eval_history.append(eval_res)

        word_bench_dist = get_benchmark_word_raw_distributions(st.session_state.e2_selected_passage) if not is_custom_now else {}

        radar_html = build_e2_multiview_canvas_html(
            st.session_state.e2_selected_passage,
            st.session_state.e2_eval_history,
            is_custom_mode=is_custom_now,
            view_mode=st.session_state.e2_plot_view,
            word_bench_dist=word_bench_dist
        )
        components.html(radar_html, height=465, scrolling=False)

    with card_col2:
        st.markdown(f"<div class='sec-header'>{SHURIKEN_ICON_SM} 多次元プロソディ診断結果 (プロット表示切替)</div>", unsafe_allow_html=True)

        v_btn1, v_btn2, v_btn3 = st.columns(3)
        with v_btn1:
            if st.button("1. 2D 総合楕円", type="primary" if st.session_state.e2_plot_view == "scatter2d" else "secondary", use_container_width=True):
                st.session_state.e2_plot_view = "scatter2d"
                st.rerun()
        with v_btn2:
            if st.button("2. 機能語 Swarm", type="primary" if st.session_state.e2_plot_view == "word_swarm" else "secondary", use_container_width=True):
                st.session_state.e2_plot_view = "word_swarm"
                st.rerun()
        with v_btn3:
            if st.button("3. 指標 1D 分布", type="primary" if st.session_state.e2_plot_view == "metrics_1d" else "secondary", use_container_width=True):
                st.session_state.e2_plot_view = "metrics_1d"
                st.rerun()

        if st.session_state.e2_eval_history:
            latest_ev = st.session_state.e2_eval_history[-1]
            trial_no = len(st.session_state.e2_eval_history)
            dm_str = f"{latest_ev['d_m_sq']} (≤9.21 合格)" if latest_ev["d_m_sq"] is not None else "自由文判定"

            c_diag = "#10B981" if latest_ev["diag_median_ms"] <= 48.0 else ("#F59E0B" if latest_ev["diag_median_ms"] <= 65.0 else "#EF4444")
            c_schwa = "#10B981" if latest_ev["schwa_p75_ms"] <= 65.0 else "#EF4444"
            c_rcf = "#10B981" if latest_ev["r_cf"] >= 1.30 else "#EF4444"
            c_npvi = "#10B981" if 40.0 <= latest_ev["npvi_clipped"] <= 62.0 else "#F59E0B"

            summary_html = (
                f'<div style="background: #0F172A; border: 1px solid #1E293B; border-radius: 8px; padding: 12px; margin-top: 6px; margin-bottom: 10px;">'
                f'<div style="display: flex; justify-content: space-between; align-items: baseline;">'
                f'<span style="font-size: 0.82rem; color: #94A3B8; font-weight: 600;">VERDICT (Trial #{trial_no})</span>'
                f'<span style="font-size: 0.78rem; color: #64748B;">蓄積データ: {trial_no} 件</span>'
                f'</div>'
                f'<div style="font-size: 1.45rem; font-weight: 800; color: {latest_ev["status_color"]}; margin-bottom: 8px;">{latest_ev["verdict"]}</div>'
                f'<div style="display: flex; justify-content: space-between; border-bottom: 1px solid #1E293B; padding-bottom: 4px; margin-bottom: 4px;">'
                f'<span style="color: #BAE6FD; font-size: 0.86rem; font-weight: 700;">弱化必須語 中央値 (V<sub>diag</sub>):</span>'
                f'<span style="font-weight: 800; color: {c_diag};">{latest_ev["diag_median_ms"]} ms <small style="font-size:0.75rem; color:#94A3B8;">(目標 ≤45ms)</small></span>'
                f'</div>'
                f'<div style="display: flex; justify-content: space-between; border-bottom: 1px solid #1E293B; padding-bottom: 4px; margin-bottom: 4px;">'
                f'<span style="color: #94A3B8; font-size: 0.86rem;">弱母音持続時間 (Schwa P75):</span>'
                f'<span style="font-weight: bold; color: {c_schwa};">{latest_ev["schwa_p75_ms"]:.1f} ms <small style="font-size:0.75rem; color:#94A3B8;">(目標 ≤65ms)</small></span>'
                f'</div>'
                f'<div style="display: flex; justify-content: space-between; border-bottom: 1px solid #1E293B; padding-bottom: 4px; margin-bottom: 4px;">'
                f'<span style="color: #94A3B8; font-size: 0.86rem;">内容語/機能語比 (R<sub>CF</sub>):</span>'
                f'<span style="font-weight: bold; color: {c_rcf};">{latest_ev["r_cf"]}x <small style="font-size:0.75rem; color:#94A3B8;">(内容 {latest_ev["mean_c_ms"]}ms / 機能 {latest_ev["mean_f_ms"]}ms)</small></span>'
                f'</div>'
                f'<div style="display: flex; justify-content: space-between; border-bottom: 1px solid #1E293B; padding-bottom: 4px; margin-bottom: 4px;">'
                f'<span style="color: #94A3B8; font-size: 0.86rem;">リズム伸縮指数 (nPVI-V):</span>'
                f'<span style="font-weight: bold; color: {c_npvi};">{latest_ev["npvi_clipped"]} <small style="font-size:0.75rem; color:#94A3B8;">(適正 40〜60)</small></span>'
                f'</div>'
                f'<div style="display: flex; justify-content: space-between; margin-bottom: 8px;">'
                f'<span style="color: #94A3B8; font-size: 0.86rem;">2D Mahalanobis (D<sub>M</sub><sup>2</sup>):</span>'
                f'<span style="font-weight: bold; color: #F8FAFC;">{dm_str}</span>'
                f'</div>'
                f'<div style="background: #090D16; border-radius: 6px; padding: 8px; font-size: 0.83rem; color: #E2E8F0; line-height: 1.4;">'
                f'💡 <b>フィードバック:</b> {latest_ev["feedback"]}'
                f'</div>'
                f'</div>'
            )
            st.markdown(summary_html, unsafe_allow_html=True)

            wb_list = latest_ev.get("word_breakdown", [])
            diag_words_meas = [w for w in wb_list if w.get("is_diag")]

            if diag_words_meas:
                st.markdown("<div style='font-size: 0.88rem; font-weight: bold; color: #BAE6FD; margin-top: 2px; margin-bottom: 4px;'>弱化必須語 (冠詞除く) の実測母音長一覧</div>", unsafe_allow_html=True)
                rows_html = ""
                for idx_w, dw in enumerate(diag_words_meas):
                    v_val = dw["vowel_ms"]
                    if v_val <= 48.0:
                        status_txt = "合格"
                        bar_c = "#10B981"
                        badge_style = "background:#064E3B; color:#A7F3D0; border:1px solid #10B981;"
                    elif v_val <= 65.0:
                        status_txt = "やや長"
                        bar_c = "#F59E0B"
                        badge_style = "background:#78350F; color:#FDE68A; border:1px solid #F59E0B;"
                    else:
                        status_txt = "引き伸ばし"
                        bar_c = "#EF4444"
                        badge_style = "background:#7F1D1D; color:#FECACA; border:1px solid #EF4444;"

                    bar_w = min(100, int((v_val / 130.0) * 100))
                    rows_html += (
                        f'<tr style="border-bottom: 1px solid #1E293B; font-size: 12px;">'
                        f'<td style="padding: 4px 6px; color: #64748B;">#{idx_w+1}</td>'
                        f'<td style="padding: 4px 6px; font-weight: 700; color: #F8FAFC;">{dw["word"]}</td>'
                        f'<td style="padding: 4px 6px; font-weight: 700; color: {bar_c}; font-variant-numeric: tabular-nums;">{v_val:.0f} ms</td>'
                        f'<td style="padding: 4px 6px; width: 34%;">'
                        f'<div style="background:#1E293B; height:7px; border-radius:4px; overflow:hidden;">'
                        f'<div style="width:{bar_w}%; background:{bar_c}; height:100%;"></div>'
                        f'</div></td>'
                        f'<td style="padding: 4px 6px; text-align: right;">'
                        f'<span style="{badge_style} padding: 1px 5px; border-radius: 4px; font-size: 10.5px; font-weight: bold;">{status_txt}</span>'
                        f'</td></tr>'
                    )

                table_html = f"""<!DOCTYPE html><html><head><style>
body {{ margin: 0; padding: 0; background: #0F172A; color: #E2E8F0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }}
.tbl-box {{ max-height: 175px; overflow-y: auto; border: 1px solid #1E293B; border-radius: 6px; padding: 2px 6px; background: #0F172A; }}
table {{ width: 100%; border-collapse: collapse; }}
th {{ position: sticky; top: 0; background: #0F172A; color: #94A3B8; font-size: 11px; text-align: left; padding: 4px 6px; border-bottom: 1px solid #334155; }}
</style></head><body>
<div class="tbl-box">
  <table>
    <thead><tr><th>順</th><th>単語</th><th>母音長</th><th>ゲージ (目標≤45ms)</th><th style="text-align:right;">判定</th></tr></thead>
    <tbody>{rows_html}</tbody>
  </table>
</div></body></html>"""
                components.html(table_html, height=185, scrolling=False)

            if st.button("この課題の試行履歴をクリア", use_container_width=True):
                st.session_state.e2_eval_history = []
                st.session_state.e2_audio_bytes = None
                st.rerun()

        else:
            st.markdown("""<div style="background: #0F172A; border: 1px dashed #334155; border-radius: 8px; padding: 24px; text-align: center; color: #94A3B8;">
左側のマイクボタンを押して、英文を読み上げてください。<br><br>
2D ストライクゾーン (R_CF × Schwa P75) による客観的な判定が行われます。
</div>""", unsafe_allow_html=True)

        if not is_custom_now and os.path.exists(active_bm["ref_audio"]):
            st.markdown("<div style='font-size: 0.9rem; font-weight: bold; color: #94A3B8; margin-top: 8px; margin-bottom: 4px;'>ネイティブお手本音声</div>", unsafe_allow_html=True)
            with open(active_bm["ref_audio"], "rb") as rf:
                st.audio(rf.read(), format="audio/wav")

    # =====================================================================
    # 最下段：発話実測マップ (単語クリックで該当区間の音声を即時再生)
    # =====================================================================
    if st.session_state.e2_eval_history and st.session_state.e2_audio_bytes:
        latest_wb = st.session_state.e2_eval_history[-1].get("word_breakdown", [])
        trial_num = len(st.session_state.e2_eval_history)
        map_html = build_clickable_word_map_html(latest_wb, st.session_state.e2_audio_bytes, trial_num)
        components.html(map_html, height=155, scrolling=False)