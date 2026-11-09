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
    /* アクティブボタン（Primary）をBenchmark ONと同じブルーティントに統一 */
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
    /* カスタムテキスト入力欄のスタイル */
    div[data-testid="stTextArea"] textarea {
        background-color: #0F172A !important;
        color: #F8FAFC !important;
        border: 1px solid #334155 !important;
        font-size: 1.15rem !important;
        font-weight: 600 !important;
        border-radius: 8px !important;
    }
    div[data-testid="stTextArea"] textarea:focus {
        border-color: #38BDF8 !important;
        box-shadow: 0 0 0 1px #38BDF8 !important;
    }
</style>
""", unsafe_allow_html=True)

# =====================================================================
# MODE 2 定数 & 機能語辞書 (セッション初期化より先に定義)
# =====================================================================
E2_PASSAGE_BENCHMARKS = {
    "S1_stress": {
        "title": "課題 1: Alternating Stress & Reduction (語彙強弱拍)",
        "target_text": "A photograph of a famous photographer was taken in the station.",
        "phonetics_desc": "photograph /ˈfoʊ.tə.ɡræf/ と photographer /fəˈtɑː.ɡrə.fɚ/ の強弱母音コントラストを測定。弱母音(/ə/)の鋭い短縮が不可欠です。",
        "mu": np.array([47.01, 58.61]),
        "cov": np.array([[1.045, 3.120], [3.120, 39.810]]),
        "x_range": (42.0, 58.0),
        "y_range": (35.0, 110.0),
        "ref_audio": "benchmarks/S1_stress/UK-Native/UK-Native_01_ArcherConver_male.wav"
    },
    "S2_clusters": {
        "title": "課題 2: Consonant Clusters & Coda Holding (子音連結保持)",
        "target_text": "Strict instructors asked complex questions about the project tests.",
        "phonetics_desc": "/str/, /mpl/, /sts/ などの連続子音に余計な母音([ɯ], [o])を挟まずに子音を保持できるかを診断。ネイティブは %V が 36% 付近まで急激に低下します。",
        "mu": np.array([35.85, 59.12]),
        "cov": np.array([[1.380, 2.850], [2.850, 48.200]]),
        "x_range": (30.0, 46.0),
        "y_range": (35.0, 120.0),
        "ref_audio": "benchmarks/S2_clusters/US-Native/US-Native_01_RogerLaidBac_male.wav"
    },
    "S3_connected": {
        "title": "課題 3: Connected Speech & Function Word Weakening (機能語弱化)",
        "target_text": "Can you give him a cup of water from the kitchen at lunch?",
        "phonetics_desc": "can, him, a, of, from, at などの機能語を極小化し、息の流れに乗せるテスト。全単語を等時拍で読むと Schwa P75 が 80ms を突破します。",
        "mu": np.array([45.42, 45.18]),
        "cov": np.array([[1.150, 1.950], [1.950, 36.400]]),
        "x_range": (38.0, 54.0),
        "y_range": (30.0, 140.0),
        "ref_audio": "benchmarks/S3_connected/UK-Native/UK-Native_01_ArcherConver_male.wav"
    }
}

# 英語機能語セット（冠詞・前置詞・代名詞・助動詞・接続詞）
FUNCTION_WORDS = {
    "a", "an", "the",
    "in", "on", "at", "to", "for", "of", "from", "with", "by", "about", "as", "into", "like", "through", "after", "over", "between", "out", "against", "during", "without", "before", "under", "around", "among",
    "i", "you", "he", "she", "it", "we", "they", "me", "him", "her", "us", "them", "my", "your", "his", "its", "our", "their", "mine", "yours", "hers", "ours", "theirs", "myself", "yourself", "himself", "herself", "itself", "ourselves", "themselves",
    "this", "that", "these", "those", "who", "whom", "whose", "which", "what", "where", "when", "why", "how",
    "am", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "having", "do", "does", "did",
    "will", "would", "shall", "should", "can", "could", "may", "might", "must",
    "and", "but", "or", "nor", "so", "yet", "if", "because", "although", "though", "while", "since", "unless", "until", "whether",
    "there", "here", "some", "any", "no", "not", "too", "very", "just", "than"
}

VOWEL_SET = {
    "AA", "AE", "AH", "AO", "AW", "AY", "EH", "ER", "EY", "IH", "IY", "OW", "OY", "UH", "UW"
}

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

# --- ヘッダー & 最上段モード切替ボタン ---
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
    {"label": "Right (US)", "file": "right_us.wav", "title": "Right /raɪt/ (US)", "phonetics": "/r/ のくびれ (谷)", "desc": "語頭でF3が急降下してF2に接近し、差分ライン(Δ)が1000Hz破線より深く潜り込みます。"},
    {"label": "Right (UK)", "file": "right_uk.wav", "title": "Right /raɪt/ (UK)", "phonetics": "/r/ の狭窄", "desc": "語頭でF3が急降下して1000Hz基準線を下回ります。二重母音 /aɪ/ への移行部でF2が上昇します。"},
    {"label": "Light (US)", "file": "light_us.wav", "title": "Light /laɪt/ (US)", "phonetics": "Dark L（軟口蓋化）", "desc": "開始部で F2（緑）が 1200Hz 付近と低く、F3（青）との差分（Δ）が 2500Hz 以上に開きます。"},
    {"label": "Light (UK)", "file": "light_uk.wav", "title": "Light /laɪt/ (UK)", "phonetics": "Clear L", "desc": "語頭の /l/ でF3は2700〜2800Hzの高域にとどまり、差分(Δ)は1000Hzを割らずに推移します。"},
    {"label": "White (UK)", "file": "white_uk.wav", "title": "White /waɪt/ (UK)", "phonetics": "円唇によるF2低下", "desc": "語頭で唇を丸めるためF2が下がりますが、F3は下がらないため、差分ラインは谷になりません。"},
    {"label": "ライト (JP)", "file": "raito_jp.wav", "title": "日本語「ライト」/ɾaɪto/", "phonetics": "タップ破裂と語尾母音付加", "desc": "「ラ」で瞬間的な弾き音が起き、語尾には英語の /t/ にはない母音「オ」が明瞭に残ります。"},
    {"label": "Really (US)", "file": "really_us.wav", "title": "Really (US)", "phonetics": "/r/ の深谷から /l/ の天井へのジャンプ", "desc": "語頭 /r/ で青（F3）が緑（F2）へ沈み、後半の /l/ に向かって F3 が 3000Hz 超へ跳ね上がります。"},
    {"label": "Really (UK)", "file": "really_uk.wav", "title": "Really (UK)", "phonetics": "センタリング二重母音の渡り", "desc": "前半の /r/ で谷を作った後、イギリス英語特有の曖昧母音 /ə/ を挟みながら滑らかに立ち上がります。"},
    {"label": "Very (US)", "file": "very_us.wav", "title": "Very (US)", "phonetics": "有声摩擦 /v/ ＋ r-colored母音", "desc": "有声摩擦 /v/ の振動が続き、続く母音〜/r/ で F3 が引きずり下ろされ、Δ が 500Hz 未満になります。"},
    {"label": "Very (UK)", "file": "very_uk.wav", "title": "Very (UK)", "phonetics": "/v/ から短母音を経て鋭い /r/ へ", "desc": "米語に比べて母音のR音性化が少なく、谷の幅（持続時間）がコンパクトで引き締まっています。"}
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
  body {{
    margin: 0; padding: 0;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    background: #0B0F19; color: #E2E8F0;
    user-select: none; overflow: hidden;
  }}
  .toolbar {{
    display: flex; align-items: center; gap: 10px;
    padding: 10px 14px; background: #0F172A;
    border: 1px solid #1E293B; border-radius: 8px 8px 0 0;
  }}
  button {{
    font-family: inherit; font-size: 14px; font-weight: bold;
    padding: 8px 14px; border-radius: 6px;
    border: 1px solid #334155; cursor: pointer;
    background: #1E293B; color: #E2E8F0;
    transition: all 0.15s; white-space: nowrap;
  }}
  button:hover {{ background: #334155; color: #FFFFFF; }}
  button.btn-play-ready {{ background: #16A34A; color: #FFFFFF; border-color: #15803D; }}
  button.btn-play-ready:hover {{ background: #15803D; }}
  button.btn-play-active {{ background: #DC2626; color: #FFFFFF; border-color: #B91C1C; }}
  button.toggle-active {{ background: #334155; color: #F8FAFC; border-color: #64748B; }}
  select {{
    font-family: inherit; font-size: 14px; padding: 6px 10px;
    border-radius: 6px; border: 1px solid #334155;
    background: #1E293B; color: #E2E8F0; font-weight: bold;
  }}
  .readout {{
    font-family: inherit; font-size: 14px; font-weight: 500; margin-left: auto;
    font-variant-numeric: tabular-nums; color: #E2E8F0; white-space: nowrap;
  }}
  #scope-wrapper {{
    position: relative; width: 100%; height: 580px;
    border: 1px solid #1E293B; border-top: none; background: #090D16;
  }}
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
    <select id="sel-rate">
      <option value="1.0">1.0x</option>
      <option value="0.7">0.7x</option>
      <option value="0.5">0.5x</option>
    </select>
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
  audioEl.preservesPitch = true;
  audioEl.mozPreservesPitch = true;
  audioEl.webkitPreservesPitch = true;
}}
initAudio();

function tToX(t) {{
  const plotW = W - MARGIN.left - MARGIN.right;
  return MARGIN.left + ((t - xMin) / (xMax - xMin)) * plotW;
}}
function xToT(x) {{
  const plotW = W - MARGIN.left - MARGIN.right;
  const t = xMin + ((x - MARGIN.left) / plotW) * (xMax - xMin);
  return Math.max(xMin, Math.min(xMax, t));
}}
function getPanels() {{
  const totalH = H - MARGIN.top - MARGIN.bottom;
  const gap = 24, h1 = 120;
  const remainH = totalH - h1 - (gap * 2);
  const h2 = remainH * 0.50, h3 = remainH * 0.50;
  const y1 = MARGIN.top, y2 = y1 + h1 + gap, y3 = y2 + h2 + gap;
  return [
    {{ top: y1, bottom: y1 + h1, height: h1 }},
    {{ top: y2, bottom: y2 + h2, height: h2 }},
    {{ top: y3, bottom: y3 + h3, height: h3 }}
  ];
}}
function valToY(val, vMin, vMax, panel) {{
  return panel.bottom - ((val - vMin) / (vMax - vMin)) * panel.height;
}}

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
    bgCtx.imageSmoothingEnabled = true; bgCtx.imageSmoothingQuality = "high";
    bgCtx.drawImage(specCanvas, MARGIN.left, p2.top, plotW, p2.height);
    bgCtx.restore();
  }}

  const APP_FONT = '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif';
  bgCtx.font = "13px " + APP_FONT;
  const durationSpan = xMax - xMin;
  const stepT = durationSpan <= 0.6 ? 0.05 : (durationSpan <= 1.5 ? 0.1 : 0.2);
  const firstTick = Math.ceil(xMin / stepT) * stepT;

  panels.forEach((p) => {{
    for (let t = firstTick; t <= xMax + 1e-6; t += stepT) {{
      const x = tToX(t);
      bgCtx.strokeStyle = "rgba(148, 163, 184, 0.14)"; bgCtx.lineWidth = 1.0;
      bgCtx.beginPath(); bgCtx.moveTo(x, p.top); bgCtx.lineTo(x, p.bottom); bgCtx.stroke();
      bgCtx.strokeStyle = "#475569"; bgCtx.lineWidth = 1.5;
      bgCtx.beginPath(); bgCtx.moveTo(x, p.bottom); bgCtx.lineTo(x, p.bottom + 5); bgCtx.stroke();
      bgCtx.fillStyle = "#94A3B8"; bgCtx.textAlign = "center"; bgCtx.textBaseline = "top";
      bgCtx.fillText(t.toFixed(2), x, p.bottom + 6);
    }}
  }});

  const waveMid = (p1.top + p1.bottom) / 2;
  bgCtx.strokeStyle = "#1E293B";
  bgCtx.beginPath(); bgCtx.moveTo(MARGIN.left, waveMid); bgCtx.lineTo(W - MARGIN.right, waveMid); bgCtx.stroke();

  const maxAmp = Math.max(...D.wave_y.map(Math.abs), 0.05);
  [-1.0, 0.0, 1.0].forEach(relVal => {{
    const absVal = relVal * maxAmp;
    const y = valToY(absVal, -maxAmp * 1.1, maxAmp * 1.1, p1);
    bgCtx.strokeStyle = "#334155"; bgCtx.lineWidth = 1.5;
    bgCtx.beginPath(); bgCtx.moveTo(MARGIN.left - 5, y); bgCtx.lineTo(MARGIN.left, y); bgCtx.stroke();
    bgCtx.fillStyle = "#94A3B8"; bgCtx.textAlign = "right"; bgCtx.textBaseline = "middle";
    bgCtx.fillText(relVal.toFixed(1), MARGIN.left - 8, y);
  }});

  bgCtx.save();
  bgCtx.beginPath(); bgCtx.rect(MARGIN.left, p1.top, plotW, p1.height); bgCtx.clip();
  bgCtx.strokeStyle = "#94A3B8"; bgCtx.lineWidth = 1.5;
  bgCtx.beginPath();
  for (let i = 0; i < D.wave_t.length; i++) {{
    const x = tToX(D.wave_t[i]);
    const y = valToY(D.wave_y[i], -maxAmp * 1.1, maxAmp * 1.1, p1);
    if (i === 0) bgCtx.moveTo(x, y); else bgCtx.lineTo(x, y);
  }}
  bgCtx.stroke(); bgCtx.restore();

  const fMin = 500, fMax = 4000;
  [1000, 1500, 2000, 2500, 3000, 3500, 4000].forEach(hz => {{
    const y = valToY(hz, fMin, fMax, p2);
    bgCtx.strokeStyle = "rgba(148, 163, 184, 0.14)"; bgCtx.lineWidth = 1.0;
    bgCtx.beginPath(); bgCtx.moveTo(MARGIN.left, y); bgCtx.lineTo(W - MARGIN.right, y); bgCtx.stroke();
    bgCtx.strokeStyle = "#475569"; bgCtx.lineWidth = 1.5;
    bgCtx.beginPath(); bgCtx.moveTo(MARGIN.left - 5, y); bgCtx.lineTo(MARGIN.left, y); bgCtx.stroke();
    bgCtx.fillStyle = "#94A3B8"; bgCtx.textAlign = "right"; bgCtx.textBaseline = "middle";
    bgCtx.fillText(hz.toString(), MARGIN.left - 8, y);
  }});

  bgCtx.save();
  bgCtx.beginPath(); bgCtx.rect(MARGIN.left, p2.top, plotW, p2.height); bgCtx.clip();
  drawSeries(D.times, D.f3, fMin, fMax, p2, "#38BDF8", 3.0);
  drawSeries(D.times, D.f2, fMin, fMax, p2, "#10B981", 3.0);
  bgCtx.restore();

  const legendX = W - MARGIN.right - 82;
  bgCtx.fillStyle = "rgba(9, 13, 22, 0.85)";
  bgCtx.fillRect(legendX - 8, p2.top + 6, 76, 44);
  bgCtx.font = "bold 14px " + APP_FONT;
  bgCtx.strokeStyle = "#38BDF8"; bgCtx.lineWidth = 3.0;
  bgCtx.beginPath(); bgCtx.moveTo(legendX, p2.top + 18); bgCtx.lineTo(legendX + 18, p2.top + 18); bgCtx.stroke();
  bgCtx.fillStyle = "#38BDF8"; bgCtx.textAlign = "left"; bgCtx.textBaseline = "middle";
  bgCtx.fillText("F3", legendX + 24, p2.top + 18);
  bgCtx.strokeStyle = "#10B981"; bgCtx.lineWidth = 3.0;
  bgCtx.beginPath(); bgCtx.moveTo(legendX, p2.top + 38); bgCtx.lineTo(legendX + 18, p2.top + 38); bgCtx.stroke();
  bgCtx.fillStyle = "#10B981"; bgCtx.fillText("F2", legendX + 24, p2.top + 38);

  const dMin = 0, dMax = 2500;
  [0, 500, 1000, 1500, 2000, 2500].forEach(hz => {{
    const y = valToY(hz, dMin, dMax, p3);
    bgCtx.strokeStyle = "rgba(148, 163, 184, 0.14)"; bgCtx.lineWidth = 1.0;
    bgCtx.beginPath(); bgCtx.moveTo(MARGIN.left, y); bgCtx.lineTo(W - MARGIN.right, y); bgCtx.stroke();
    bgCtx.strokeStyle = "#475569"; bgCtx.lineWidth = 1.5;
    bgCtx.beginPath(); bgCtx.moveTo(MARGIN.left - 5, y); bgCtx.lineTo(MARGIN.left, y); bgCtx.stroke();
    bgCtx.fillStyle = "#94A3B8"; bgCtx.textAlign = "right"; bgCtx.textBaseline = "middle";
    bgCtx.fillText(hz.toString(), MARGIN.left - 8, y);
  }});

  const yTarget = valToY(1000, dMin, dMax, p3);
  const yZero = valToY(0, dMin, dMax, p3);
  bgCtx.save();
  bgCtx.fillStyle = "rgba(239, 68, 68, 0.08)";
  bgCtx.fillRect(MARGIN.left, yTarget, plotW, yZero - yTarget);
  bgCtx.strokeStyle = "rgba(239, 68, 68, 0.65)"; bgCtx.lineWidth = 1.5; bgCtx.setLineDash([6, 5]);
  bgCtx.beginPath(); bgCtx.moveTo(MARGIN.left, yTarget); bgCtx.lineTo(W - MARGIN.right, yTarget); bgCtx.stroke();
  bgCtx.restore();
  bgCtx.font = "bold 12px " + APP_FONT;
  bgCtx.fillStyle = "rgba(239, 68, 68, 0.85)"; bgCtx.textAlign = "right"; bgCtx.textBaseline = "bottom";
  bgCtx.fillText("R STRIKE ZONE (<1000Hz)", W - MARGIN.right - 8, yTarget - 4);

  bgCtx.save();
  bgCtx.beginPath(); bgCtx.rect(MARGIN.left, p3.top, plotW, p3.height); bgCtx.clip();
  drawSeries(D.times, D.delta, dMin, dMax, p3, "#F59E0B", 3.2);
  bgCtx.restore();

  const leg3X = W - MARGIN.right - 145;
  bgCtx.fillStyle = "rgba(9, 13, 22, 0.85)"; bgCtx.fillRect(leg3X - 8, p3.top + 6, 140, 26);
  bgCtx.font = "bold 13px " + APP_FONT;
  bgCtx.strokeStyle = "#F59E0B"; bgCtx.lineWidth = 3.0;
  bgCtx.beginPath(); bgCtx.moveTo(leg3X, p3.top + 18); bgCtx.lineTo(leg3X + 18, p3.top + 18); bgCtx.stroke();
  bgCtx.fillStyle = "#F59E0B"; bgCtx.textAlign = "left"; bgCtx.textBaseline = "middle";
  bgCtx.fillText("Δ(F3 - F2)", leg3X + 24, p3.top + 18);

  panels.forEach(p => {{
    bgCtx.strokeStyle = "#334155"; bgCtx.lineWidth = 1.5;
    bgCtx.beginPath(); bgCtx.moveTo(MARGIN.left, p.top); bgCtx.lineTo(MARGIN.left, p.bottom); bgCtx.lineTo(W - MARGIN.right, p.bottom); bgCtx.stroke();
  }});

  bgCtx.font = "bold 15px " + APP_FONT; bgCtx.fillStyle = "#CBD5E1";
  drawRotatedLabel("Waveform", 28, (p1.top + p1.bottom) / 2);
  drawRotatedLabel("F2 and F3 (Hz)", 28, (p2.top + p2.bottom) / 2);
  drawRotatedLabel("Δ(F3 - F2)", 28, (p3.top + p3.bottom) / 2);

  bgCtx.fillStyle = "#94A3B8"; bgCtx.textAlign = "center"; bgCtx.textBaseline = "bottom";
  bgCtx.fillText("Time (s)", MARGIN.left + plotW / 2, H - 2);
}}

function drawSeries(times, values, vMin, vMax, panel, color, width) {{
  bgCtx.strokeStyle = color; bgCtx.lineWidth = width; bgCtx.lineJoin = "round";
  bgCtx.beginPath(); let penDown = false;
  for (let i = 0; i < times.length; i++) {{
    const val = values[i];
    if (val === null || val === undefined) {{ penDown = false; continue; }}
    const x = tToX(times[i]); const y = valToY(val, vMin, vMax, panel);
    if (!penDown) {{ bgCtx.moveTo(x, y); penDown = true; }} else {{ bgCtx.lineTo(x, y); }}
  }}
  bgCtx.stroke();
}}
function drawRotatedLabel(text, x, y) {{
  bgCtx.save(); bgCtx.translate(x, y); bgCtx.rotate(-Math.PI / 2);
  bgCtx.textAlign = "center"; bgCtx.textBaseline = "middle"; bgCtx.fillText(text, 0, 0); bgCtx.restore();
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
  ovCtx.fillStyle = "#EF4444";
  ovCtx.beginPath(); ovCtx.moveTo(cx - 6, topY); ovCtx.lineTo(cx + 6, topY); ovCtx.lineTo(cx, topY + 8); ovCtx.closePath(); ovCtx.fill();
  ovCtx.beginPath(); ovCtx.moveTo(cx - 6, botY); ovCtx.lineTo(cx + 6, botY); ovCtx.lineTo(cx, botY - 8); ovCtx.closePath(); ovCtx.fill();

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
  const rate = parseFloat(selRate.value);
  audioEl.playbackRate = rate;
  audioEl.preservesPitch = true; audioEl.mozPreservesPitch = true; audioEl.webkitPreservesPitch = true;
  const s = Math.min(selStart, selEnd), e = Math.max(selStart, selEnd);
  const playFrom = (cursorT >= s && cursorT < e - 0.02) ? cursorT : s;
  audioEl.currentTime = playFrom;
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
selRate.addEventListener("change", () => {{
  audioEl.playbackRate = parseFloat(selRate.value);
  audioEl.preservesPitch = true;
}});
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
  stopPlayback(); const rect = ovCanvas.getBoundingClientRect(); const mouseX = e.clientX - rect.left; const clickT = xToT(mouseX);
  if (Math.abs(mouseX - tToX(cursorT)) <= 12) dragMode = "cursor";
  else {{ dragMode = "select"; selStart = clickT; selEnd = clickT; cursorT = clickT; drawOverlay(); }}
}});
window.addEventListener("mouseup", () => {{
  if (dragMode === "select") {{
    if (Math.abs(selEnd - selStart) < 0.015) {{ selStart = xMin; selEnd = xMax; }}
    else cursorT = Math.min(selStart, selEnd);
  }}
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
# MODE 2: E2 3大プロソディ診断 & ルート1 自由文ハイブリッド評価エンジン
# =====================================================================
@st.cache_data
def load_cached_benchmark_df():
    p1 = "benchmarks/benchmark_results_all.csv"
    p2 = "benchmarks/benchmark_results.csv"
    target = p1 if os.path.exists(p1) else (p2 if os.path.exists(p2) else None)
    if target:
        df = pd.read_csv(target)
        if "sentence_key" not in df.columns:
            df["sentence_key"] = "S1_stress"
        return df
    return None


def compute_content_function_metrics(w_aligns, phones):
    """
    単語アライメントと音素列から、内容語(Content)と機能語(Function)の
    母音平均長比 (R_CF) および単語ごとの持続時間リストを算出
    """
    word_breakdown = []
    content_vowel_durs = []
    func_vowel_durs = []

    for w in w_aligns:
        raw_word = w.get("word", "")
        clean_w = re.sub(r"[^a-zA-Z']", "", raw_word).lower()
        if not clean_w:
            continue
        is_func = clean_w in FUNCTION_WORDS
        w_start = w.get("start", 0.0)
        w_end = w.get("end", 0.0)
        w_dur_ms = max(0.0, (w_end - w_start) * 1000.0)

        # この単語区間に含まれる母音を抽出
        w_vowels = [
            p for p in phones
            if p.get("start", 0.0) >= w_start - 0.01
            and p.get("end", 0.0) <= w_end + 0.01
            and re.sub(r"[0-9]", "", p.get("phone", "")) in VOWEL_SET
        ]
        v_durs = [(p["end"] - p["start"]) * 1000.0 for p in w_vowels]

        if v_durs:
            if is_func:
                func_vowel_durs.extend(v_durs)
            else:
                content_vowel_durs.extend(v_durs)
        else:
            # 母音が拾えなかった場合は単語長を音節近似で代用
            if is_func:
                func_vowel_durs.append(w_dur_ms * 0.45)
            else:
                content_vowel_durs.append(w_dur_ms * 0.45)

        word_breakdown.append({
            "word": raw_word,
            "is_func": is_func,
            "dur_ms": round(w_dur_ms, 1),
            "vowel_ms": round(float(np.mean(v_durs)) if v_durs else w_dur_ms * 0.45, 1)
        })

    mean_c = float(np.mean(content_vowel_durs)) if content_vowel_durs else 120.0
    mean_f = float(np.mean(func_vowel_durs)) if func_vowel_durs else 60.0
    r_cf = round(mean_c / max(mean_f, 15.0), 2)

    return r_cf, round(mean_c, 1), round(mean_f, 1), word_breakdown


def evaluate_user_rhythm(audio_path: str, passage_key: str, custom_text: str):
    """
    テキストがプリセットと一致する場合はマハラノビス距離で判定し、
    自由文に書き換えられている場合は R_CF + nPVI + Schwa P75 で動的判定する。
    """
    if not ALIGNMENT_AVAILABLE:
        return {"error": "アライメントモジュールがロードできませんでした。"}

    bm = E2_PASSAGE_BENCHMARKS[passage_key]
    norm_custom = " ".join(custom_text.strip().split())
    norm_preset = " ".join(bm["target_text"].strip().split())
    is_custom_mode = (norm_custom.lower() != norm_preset.lower())

    w_aligns = align_word_characters(audio_path, norm_custom)
    phones = map_words_to_phonemes(w_aligns)
    metrics = compute_robust_metrics(phones)

    r_cf, mean_c_ms, mean_f_ms, word_breakdown = compute_content_function_metrics(w_aligns, phones)

    pct_v = metrics["pct_v"]
    schwa_p75 = metrics["schwa_p75_ms"]
    npvi = metrics["npvi_clipped"]

    if not is_custom_mode:
        # --- プリセット文モード: 実測共分散マハラノビス判定 ---
        user_pt = np.array([pct_v, schwa_p75])
        mu = bm["mu"]
        cov = bm["cov"]
        inv_cov = np.linalg.inv(cov)
        diff = user_pt - mu
        d_m_sq = round(float(diff.T @ inv_cov @ diff), 2)

        is_strike = d_m_sq <= 5.991
        is_bullseye = d_m_sq <= (5.991 / 20.0)

        if is_bullseye:
            verdict = "PERFECT STRIKE"
            status_color = "#38BDF8"
            feedback = f"ネイティブ母集団の平均重心に完全一致しています（機能語圧縮比 R_CF = {r_cf}x）。理想的なストレス拍です。"
        elif is_strike:
            verdict = "NATIVE STRIKE"
            status_color = "#10B981"
            feedback = f"95% ネイティブ許容ゾーンに合格しています（R_CF = {r_cf}x, nPVI = {npvi:.1f}）。自然な英語のリズム骨格です。"
        else:
            status_color = "#EF4444"
            delta_v = pct_v - mu[0]
            delta_u = schwa_p75 - mu[1]
            if delta_u > 15.0 and delta_v > 3.0:
                verdict = "MORA TRAP (等時拍)"
                feedback = f"機能語や弱母音が引き伸ばされています（R_CF = {r_cf}x）。下のバーで赤くなっている機能語を 40ms 以下に素早く脱落させてください。"
            elif delta_v > 4.0:
                verdict = "VOWEL HEAVY (不要母音)"
                feedback = f"母音占有率(%V)が基準より +{delta_v:.1f}% 高いです。子音連結部に余計な母音が入っていないか確認してください。"
            elif delta_u > 15.0:
                verdict = "WEAK VOWEL TOO LONG"
                feedback = f"弱母音(Schwa)が +{delta_u:.1f}ms 長すぎます。強勢のない音節を喉の奥で短く処理してください。"
            else:
                verdict = "OFF TARGET"
                feedback = "ストライクゾーンから外れています。内容語を長く、機能語を短く読むコントラストを強めてください。"
    else:
        # --- ルート1: 自由文カスタムモード (R_CF + nPVI + Schwa P75 による普遍則判定) ---
        d_m_sq = None
        if r_cf >= 2.0 and schwa_p75 <= 65.0 and npvi >= 52.0:
            verdict = "DYNAMIC STRIKE (自由文合格)"
            status_color = "#10B981"
            is_strike = True
            is_bullseye = (r_cf >= 2.5 and schwa_p75 <= 55.0)
            if is_bullseye:
                verdict = "PERFECT PROSODY (自由文)"
                status_color = "#38BDF8"
            feedback = f"自由文で理想的なストレス拍コントラストが出ています！（内容語/機能語比 R_CF = {r_cf}x、nPVI = {npvi:.1f}）"
        elif r_cf >= 1.5 and schwa_p75 <= 75.0:
            verdict = "PASSABLE RHYTHM (惜しい)"
            status_color = "#F59E0B"
            is_strike = False
            is_bullseye = False
            feedback = f"英語のリズム骨格は出ていますが、あと一歩機能語を圧縮できます（現在 R_CF = {r_cf}x、目標 ≥ 2.0x）。"
        else:
            verdict = "MORA RHYTHM (等時拍傾向)"
            status_color = "#EF4444"
            is_strike = False
            is_bullseye = False
            feedback = f"内容語と機能語の長さに差がありません（R_CF = {r_cf}x、弱母音P75 = {schwa_p75:.1f}ms）。下の単語バーで赤い機能語を徹底的に短く呟くように発音してください。"

    return {
        "is_custom_mode": is_custom_mode,
        "pct_v": pct_v,
        "schwa_p75_ms": schwa_p75,
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
    """Canvas上にStrike Zone（または自由文用Universal Target Box）、ユーザー履歴、ベンチマーク点群を描画"""
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

    bench_points = []
    df_all = load_cached_benchmark_df()
    if df_all is not None and not is_custom_mode:
        sub = df_all[df_all["sentence_key"] == passage_key]
        for _, row in sub.iterrows():
            c_name = row["cohort"]
            col = "#38BDF8" if c_name == "US-Native" else (
                  "#10B981" if c_name == "UK-Native" else (
                  "#F59E0B" if c_name == "JP-Accented" else "#EF4444"))
            bench_points.append({
                "pct_v": float(row["pct_v"]),
                "schwa_p75_ms": float(row["schwa_p75_ms"]),
                "color": col,
                "cohort": c_name
            })

    history_list = eval_history if eval_history is not None else []

    canvas_data = {
        "passage_key": passage_key,
        "is_custom_mode": is_custom_mode,
        "mu_x": float(mu[0]),
        "mu_y": float(mu[1]),
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

    mode_badge = "✏️ CUSTOM TEXT MODE" if is_custom_mode else f"🎯 2D Strike Zone Radar (Trials: {len(history_list)})"

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
  #scope-wrapper {{ position: relative; width: 100%; height: 400px; background: #090D16; border: 1px solid #1E293B; border-top: none; border-radius: 0 0 8px 8px; }}
  canvas {{ position: absolute; left: 0; top: 0; width: 100%; height: 100%; }}
</style>
</head>
<body>
<div class="toolbar">
  <div class="toolbar-title">
    <span>{mode_badge}</span>
  </div>
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
let W = 800, H = 400;

function xToPx(x) {{ return MARGIN.left + ((x - D.x_min) / (D.x_max - D.x_min)) * (W - MARGIN.left - MARGIN.right); }}
function yToPx(y) {{ return (H - MARGIN.bottom) - ((y - D.y_min) / (D.y_max - D.y_min)) * (H - MARGIN.top - MARGIN.bottom); }}

function createEllipsePath(ctx, mu_x, mu_y, v1, v2) {{
  ctx.beginPath();
  const N_PTS = 64;
  for (let i = 0; i <= N_PTS; i++) {{
    const theta = (i / N_PTS) * 2 * Math.PI;
    const cosT = Math.cos(theta);
    const sinT = Math.sin(theta);
    const dataX = mu_x + cosT * v1[0] + sinT * v2[0];
    const dataY = mu_y + cosT * v1[1] + sinT * v2[1];
    const px = xToPx(dataX);
    const py = yToPx(dataY);
    if (i === 0) ctx.moveTo(px, py);
    else ctx.lineTo(px, py);
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

  // グリッド線描画
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
  ctx.beginPath();
  ctx.rect(MARGIN.left, MARGIN.top, plotW, plotH);
  ctx.clip();

  if (!D.is_custom_mode) {{
    // 1. プリセット時: 95% Native Strike Zone 楕円
    createEllipsePath(ctx, D.mu_x, D.mu_y, D.v1_95, D.v2_95);
    ctx.fillStyle = "rgba(16, 185, 129, 0.15)";
    ctx.fill();
    ctx.strokeStyle = "#10B981";
    ctx.lineWidth = 2.0;
    ctx.setLineDash([5, 4]);
    ctx.stroke();

    // 2. Centroid (95% CI)
    createEllipsePath(ctx, D.mu_x, D.mu_y, D.v1_ci, D.v2_ci);
    ctx.fillStyle = "rgba(56, 189, 248, 0.35)";
    ctx.fill();
    ctx.strokeStyle = "#38BDF8";
    ctx.lineWidth = 1.2;
    ctx.setLineDash([]);
    ctx.stroke();

    // 3. ベンチマーク背景ゴースト点群
    if (showBenchmark && D.benchmark && D.benchmark.length > 0) {{
      D.benchmark.forEach(b => {{
        const bx = xToPx(b.pct_v);
        const by = yToPx(b.schwa_p75_ms);
        ctx.save();
        ctx.globalAlpha = 0.38;
        ctx.fillStyle = b.color;
        ctx.beginPath(); ctx.arc(bx, by, 4.5, 0, 2 * Math.PI); ctx.fill();
        ctx.restore();
      }});

      const legX = W - MARGIN.right - 145;
      const legY = MARGIN.top + 8;
      ctx.fillStyle = "rgba(9, 13, 22, 0.82)";
      ctx.fillRect(legX, legY, 135, 52);
      ctx.strokeStyle = "#1E293B"; ctx.lineWidth = 1.0;
      ctx.strokeRect(legX, legY, 135, 52);

      ctx.font = "10.5px sans-serif"; ctx.textAlign = "left"; ctx.textBaseline = "middle";
      ctx.fillStyle = "#38BDF8"; ctx.beginPath(); ctx.arc(legX + 12, legY + 12, 3.5, 0, 2 * Math.PI); ctx.fill();
      ctx.fillStyle = "#94A3B8"; ctx.fillText("Native (US/UK)", legX + 22, legY + 12);
      ctx.fillStyle = "#F59E0B"; ctx.beginPath(); ctx.arc(legX + 12, legY + 26, 3.5, 0, 2 * Math.PI); ctx.fill();
      ctx.fillStyle = "#94A3B8"; ctx.fillText("JP-Accented", legX + 22, legY + 26);
      ctx.fillStyle = "#EF4444"; ctx.beginPath(); ctx.arc(legX + 12, legY + 40, 3.5, 0, 2 * Math.PI); ctx.fill();
      ctx.fillStyle = "#94A3B8"; ctx.fillText("JP-Mora-Forced", legX + 22, legY + 40);
    }}
  }} else {{
    // 自由文モード時: Universal Stress-Timed Target Zone (Schwa <= 65ms, %V: 35~49%)
    const zx1 = xToPx(35.0), zx2 = xToPx(49.0);
    const zy1 = yToPx(65.0), zy2 = yToPx(32.0);
    ctx.fillStyle = "rgba(16, 185, 129, 0.14)";
    ctx.fillRect(zx1, zy1, zx2 - zx1, zy2 - zy1);
    ctx.strokeStyle = "#10B981"; ctx.lineWidth = 2.0; ctx.setLineDash([6, 4]);
    ctx.strokeRect(zx1, zy1, zx2 - zx1, zy2 - zy1);
    ctx.setLineDash([]);
    ctx.fillStyle = "#10B981"; ctx.font = "bold 12px sans-serif"; ctx.textAlign = "left";
    ctx.fillText("UNIVERSAL STRESS-TIMED ZONE (Schwa ≤ 65ms)", zx1 + 8, zy1 + 16);
  }}

  // 4. ユーザーの試行履歴（フェード描画）
  const total = D.history.length;
  if (total > 0) {{
    if (total > 1) {{
      ctx.strokeStyle = "rgba(248, 250, 252, 0.30)";
      ctx.lineWidth = 1.5;
      ctx.setLineDash([3, 3]);
      ctx.beginPath();
      for (let i = 0; i < total; i++) {{
        const pt = D.history[i];
        const px = xToPx(pt.pct_v);
        const py = yToPx(pt.schwa_p75_ms);
        if (i === 0) ctx.moveTo(px, py);
        else ctx.lineTo(px, py);
      }}
      ctx.stroke();
      ctx.setLineDash([]);
    }}

    D.history.forEach((pt, idx) => {{
      const isLatest = (idx === total - 1);
      const px = xToPx(pt.pct_v);
      const py = yToPx(pt.schwa_p75_ms);
      const alpha = total === 1 ? 1.0 : 0.35 + (idx / (total - 1)) * 0.65;

      ctx.save();
      ctx.globalAlpha = alpha;

      if (isLatest) {{
        ctx.fillStyle = pt.status_color + "44";
        ctx.beginPath(); ctx.arc(px, py, 18, 0, 2 * Math.PI); ctx.fill();

        ctx.fillStyle = pt.status_color;
        ctx.strokeStyle = "#FFFFFF"; ctx.lineWidth = 2.5;
        ctx.beginPath(); ctx.arc(px, py, 8, 0, 2 * Math.PI); ctx.fill(); ctx.stroke();

        ctx.fillStyle = "#F8FAFC"; ctx.font = "bold 12px sans-serif";
        ctx.textAlign = px > W - 140 ? "right" : "left";
        ctx.fillText(`#${{idx+1}} (R_CF:${{pt.r_cf}}x): ${{pt.pct_v}}%, ${{pt.schwa_p75_ms}}ms`, px + (px > W - 140 ? -12 : 12), py - 8);
      }} else {{
        ctx.fillStyle = pt.status_color;
        ctx.strokeStyle = "rgba(11, 15, 25, 0.9)"; ctx.lineWidth = 1.5;
        ctx.beginPath(); ctx.arc(px, py, 6, 0, 2 * Math.PI); ctx.fill(); ctx.stroke();

        ctx.fillStyle = "#E2E8F0"; ctx.font = "10px sans-serif";
        ctx.textAlign = "center";
        ctx.fillText(`#${{idx+1}}`, px, py - 8);
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
  link.download = `NinjaPhonetics_E2_${{D.passage_key}}_radar.png`;
  link.href = canvas.toDataURL("image/png");
  link.click();
}});

window.addEventListener("resize", draw);
draw();
</script>
</body>
</html>
"""


# =====================================================================
# メイン画面レンダリング分岐
# =====================================================================
if st.session_state.app_mode == "RL":
    cols = st.columns(len(PRESETS))
    preset_clicked = False
    for idx, col in enumerate(cols):
        p_info = PRESETS[idx]
        is_active = (st.session_state.current_label == f"お手本: {p_info['label']}")
        if col.button(
            p_info["label"],
            type="primary" if is_active else "secondary",
            use_container_width=True
        ):
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
    # --- MODE 2 (E2 3大プロソディ診断 ＋ ルート1 自由文カスタマイズ) ---
    # =====================================================================
    st.markdown("""
    <div style='color: #94A3B8; font-size: 0.95rem; margin-bottom: 10px;'>
      🎯 <b>診断方法</b>: 課題（S1〜S3）を選ぶか、下のテキスト欄に<b>任意の英文</b>を入力してマイクで読み上げてください。<br>
      プリセット文では実測95%楕円（$D_M^2$）で判定し、英文を書き換えると自動的に<b>自由文プロソディ診断（機能語圧縮比 $R_{CF}$ ＆ nPVI）</b>に切り替わります。
    </div>
    """, unsafe_allow_html=True)

    # 課題選択ボタン (3分割)
    btn_col1, btn_col2, btn_col3 = st.columns(3)
    with btn_col1:
        if st.button("課題 1: 語彙強弱 (S1)", type="primary" if st.session_state.e2_selected_passage == "S1_stress" else "secondary", use_container_width=True):
            st.session_state.e2_selected_passage = "S1_stress"
            st.session_state.e2_custom_text = E2_PASSAGE_BENCHMARKS["S1_stress"]["target_text"]
            st.session_state.e2_last_analyzed_text = st.session_state.e2_custom_text
            st.session_state.e2_eval_history = []
            st.session_state.e2_audio_bytes = None
            st.rerun()
    with btn_col2:
        if st.button("課題 2: 子音連結 (S2)", type="primary" if st.session_state.e2_selected_passage == "S2_clusters" else "secondary", use_container_width=True):
            st.session_state.e2_selected_passage = "S2_clusters"
            st.session_state.e2_custom_text = E2_PASSAGE_BENCHMARKS["S2_clusters"]["target_text"]
            st.session_state.e2_last_analyzed_text = st.session_state.e2_custom_text
            st.session_state.e2_eval_history = []
            st.session_state.e2_audio_bytes = None
            st.rerun()
    with btn_col3:
        if st.button("課題 3: 機能語弱化 (S3)", type="primary" if st.session_state.e2_selected_passage == "S3_connected" else "secondary", use_container_width=True):
            st.session_state.e2_selected_passage = "S3_connected"
            st.session_state.e2_custom_text = E2_PASSAGE_BENCHMARKS["S3_connected"]["target_text"]
            st.session_state.e2_last_analyzed_text = st.session_state.e2_custom_text
            st.session_state.e2_eval_history = []
            st.session_state.e2_audio_bytes = None
            st.rerun()

    active_bm = E2_PASSAGE_BENCHMARKS[st.session_state.e2_selected_passage]

    # カスタム英文入力テキストエリア (初期値はお題英文)
    txt_col1, txt_col2 = st.columns([4.5, 1.0], vertical_alignment="bottom")
    with txt_col1:
        user_target_text = st.text_area(
            "📝 診断ターゲット英文（自由に書き換え可能）",
            value=st.session_state.e2_custom_text,
            height=68,
            key="e2_text_input_widget"
        )
    with txt_col2:
        if st.button("↺ 例文に戻す", use_container_width=True):
            st.session_state.e2_custom_text = active_bm["target_text"]
            st.session_state.e2_last_analyzed_text = active_bm["target_text"]
            st.session_state.e2_eval_history = []
            st.session_state.e2_audio_bytes = None
            st.rerun()

    # テキストが書き換えられたら履歴を自動リセットしてモード切替
    if user_target_text.strip() != st.session_state.e2_last_analyzed_text.strip():
        st.session_state.e2_custom_text = user_target_text.strip()
        st.session_state.e2_last_analyzed_text = user_target_text.strip()
        st.session_state.e2_eval_history = []
        st.session_state.e2_audio_bytes = None

    is_custom_now = (
        " ".join(st.session_state.e2_custom_text.lower().split())
        != " ".join(active_bm["target_text"].lower().split())
    )

    # 現在のテキストを「内容語(青 badge)」と「機能語(灰 badge)」に色分けプレビュー表示
    words_preview = st.session_state.e2_custom_text.split()
    preview_badges = []  # ← インデントの余分なスペースを解消
    for w in words_preview:
        clean_w = re.sub(r"[^a-zA-Z']", "", w).lower()
        if clean_w in FUNCTION_WORDS:
            preview_badges.append(
                f"<span style='background:#1E293B; color:#94A3B8; border:1px solid #334155; padding:2px 7px; border-radius:4px; font-size:0.95rem; margin-right:4px;'>{w} <small style='font-size:0.7rem; color:#64748B;'>[機能]</small></span>"
            )
        else:
            preview_badges.append(
                f"<span style='background:#1E3A8A; color:#F8FAFC; border:1px solid #3B82F6; padding:2px 8px; border-radius:4px; font-size:1.0rem; font-weight:700; margin-right:4px;'>{w} <small style='font-size:0.7rem; color:#93C5FD;'>[内容]</small></span>"
            )

    mode_header = "✏️ CUSTOM FREE-TEXT PROSODY MODE (動的アライメント ＋ 機能語圧縮診断)" if is_custom_now else active_bm["title"]
    mode_desc = "青色の【内容語】にしっかり時間をかけ、灰色の【機能語】を極限まで短く（40〜60ms）圧縮して発音してください。" if is_custom_now else active_bm["phonetics_desc"]

    st.markdown(f"""
    <div style="background: #0F172A; border: 1px solid #1E293B; border-left: 4px solid {'#F59E0B' if is_custom_now else '#38BDF8'}; border-radius: 8px; padding: 12px 18px; margin-bottom: 12px;">
      <div style="font-size: 0.82rem; font-weight: 700; color: {'#F59E0B' if is_custom_now else '#38BDF8'}; text-transform: uppercase; margin-bottom: 6px;">{mode_header}</div>
      <div style="line-height: 2.1; margin-bottom: 6px;">{''.join(preview_badges)}</div>
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
        components.html(radar_html, height=450, scrolling=False)

    with card_col2:
        st.markdown("<div style='font-size: 1.05rem; font-weight: bold; color: #F8FAFC; margin-bottom: 8px;'>🎯 最新の診断結果</div>", unsafe_allow_html=True)

        if st.session_state.e2_eval_history:
            latest_ev = st.session_state.e2_eval_history[-1]
            trial_no = len(st.session_state.e2_eval_history)
            dm_str = f"{latest_ev['d_m_sq']} (≤5.99 合格)" if latest_ev["d_m_sq"] is not None else "自由文モード (R_CF / nPVI 判定)"
            
            st.markdown(f"""
            <div style="background: #1E293B; border: 1px solid #334155; border-radius: 8px; padding: 16px; margin-bottom: 12px;">
              <div style="display: flex; justify-content: space-between; align-items: baseline;">
                <span style="font-size: 0.85rem; color: #94A3B8; font-weight: 600;">VERDICT (Trial #{trial_no})</span>
                <span style="font-size: 0.8rem; color: #64748B;">蓄積データ: {trial_no} 件</span>
              </div>
              <div style="font-size: 1.5rem; font-weight: 800; color: {latest_ev['status_color']}; margin-bottom: 10px;">{latest_ev['verdict']}</div>
              
              <div style="display: flex; justify-content: space-between; border-bottom: 1px solid #334155; padding-bottom: 5px; margin-bottom: 5px;">
                <span style="color: #93C5FD; font-size: 0.9rem; font-weight: 600;">機能語圧縮比 ($R_{{CF}}$):</span>
                <span style="font-weight: bold; color: #38BDF8;">{latest_ev['r_cf']}x (内容 {latest_ev['mean_c_ms']}ms / 機能 {latest_ev['mean_f_ms']}ms)</span>
              </div>
              <div style="display: flex; justify-content: space-between; border-bottom: 1px solid #334155; padding-bottom: 5px; margin-bottom: 5px;">
                <span style="color: #94A3B8; font-size: 0.9rem;">リズム伸縮指数 (nPVI-V):</span>
                <span style="font-weight: bold; color: #F8FAFC;">{latest_ev['npvi_clipped']} (目標 ≥ 52)</span>
              </div>
              <div style="display: flex; justify-content: space-between; border-bottom: 1px solid #334155; padding-bottom: 5px; margin-bottom: 5px;">
                <span style="color: #94A3B8; font-size: 0.9rem;">母音占有率 (%V) / 弱母音P75:</span>
                <span style="font-weight: bold; color: #F8FAFC;">{latest_ev['pct_v']:.1f}% / {latest_ev['schwa_p75_ms']:.1f}ms</span>
              </div>
              <div style="display: flex; justify-content: space-between; margin-bottom: 12px;">
                <span style="color: #94A3B8; font-size: 0.9rem;">Mahalanobis ($D_M^2$):</span>
                <span style="font-weight: bold; color: #F8FAFC;">{dm_str}</span>
              </div>
              
              <div style="background: #0F172A; border-radius: 6px; padding: 10px; font-size: 0.88rem; color: #E2E8F0; line-height: 1.45;">
                💡 <b>アドバイス:</b><br>{latest_ev['feedback']}
              </div>
            </div>
            """, unsafe_allow_html=True)

            # 単語ごとの持続時間バー可視化（機能語が間延びしていないか一目瞭然にする）
            if latest_ev.get("word_breakdown"):
                st.markdown("<div style='font-size: 0.9rem; font-weight: bold; color: #CBD5E1; margin-top: 8px; margin-bottom: 6px;'>📊 単語別の母音持続時間 (赤＝引き伸ばしすぎた機能語)</div>", unsafe_allow_html=True)
                bars_html = "<div style='background:#0F172A; border:1px solid #1E293B; border-radius:8px; padding:10px; max-height:190px; overflow-y:auto;'>"
                for wb in latest_ev["word_breakdown"]:
                    v_ms = wb["vowel_ms"]
                    width_pct = min(100, int((v_ms / 180.0) * 100))
                    if wb["is_func"]:
                        # 機能語なのに 75ms を超えている場合は赤で警告
                        bar_col = "#EF4444" if v_ms > 75.0 else "#10B981"
                        tag_lbl = "機能語"
                    else:
                        bar_col = "#3B82F6"
                        tag_lbl = "内容語"
                    bars_html += f"""
                    <div style="display:flex; align-items:center; gap:8px; margin-bottom:5px; font-size:0.82rem;">
                      <div style="width:95px; text-align:right; font-weight:600; color:#E2E8F0; overflow:hidden; text-overflow:ellipsis; white-space:nowrap;">{wb['word']} <span style="font-size:0.7rem; color:#64748B;">({tag_lbl[0]})</span></div>
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
              左側のマイクボタンを押して、上のターゲット英文を読み上げてください。<br><br>
              英文を自由に書き換えた場合も、自動で内容語と機能語を判別してプロソディ診断を行います。
            </div>
            """, unsafe_allow_html=True)

        if not is_custom_now and os.path.exists(active_bm["ref_audio"]):
            st.markdown("<div style='font-size: 0.95rem; font-weight: bold; color: #94A3B8; margin-top: 14px; margin-bottom: 4px;'>🔊 ネイティブお手本音声</div>", unsafe_allow_html=True)
            with open(active_bm["ref_audio"], "rb") as rf:
                st.audio(rf.read(), format="audio/wav")