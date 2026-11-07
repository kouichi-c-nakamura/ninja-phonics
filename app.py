import streamlit as st
import streamlit.components.v1 as components
import parselmouth
import soundfile as sf
import numpy as np
import base64
import json
import io
import os
import re

st.set_page_config(page_title="NinjaPhonetics Scope", layout="wide", initial_sidebar_state="collapsed")

# 画面パディングとスタイル調整
st.markdown("""
<style>
    .block-container {
        padding-top: 1.5rem !important;
        padding-bottom: 0.5rem !important;
        padding-left: 1.5rem !important;
        padding-right: 1.5rem !important;
    }
    h1 {
        font-size: 1.5rem !important;
        font-weight: 700 !important;
        margin-bottom: 0.4rem !important;
        color: #F8FAFC !important;
        line-height: 1.2 !important;
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
    /* モード切替用アクティブボタン強調 */
    div[data-testid="column"] .stButton button[kind="primary"] {
        background-color: #2563EB !important;
        border-color: #3B82F6 !important;
        color: #FFFFFF !important;
    }
</style>
""", unsafe_allow_html=True)

# --- セッション状態の初期化 ---
if "app_mode" not in st.session_state:
    st.session_state.app_mode = "RL"
if "current_audio_bytes" not in st.session_state:
    st.session_state.current_audio_bytes = None
if "current_label" not in st.session_state:
    st.session_state.current_label = ""
if "last_recorded_bytes" not in st.session_state:
    st.session_state.last_recorded_bytes = None
if "auto_play" not in st.session_state:
    st.session_state.auto_play = False
if "rhythm_audio_bytes" not in st.session_state:
    st.session_state.rhythm_audio_bytes = None

# --- ヘッダー & 最上段モード切替ボタン ---
head_col1, head_col2, head_col3 = st.columns([2.2, 1.2, 1.2])
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
        "E2: 強弱リズム・母音長比",
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
# MODE 1: R / L ホルマントスコープ (既存完成版)
# =====================================================================
PRESET_DIR = "presets"
PRESETS = [
    {
        "label": "Right (US)",
        "file": "right_us.wav",
        "title": "Right /raɪt/ (US)",
        "phonetics": "/r/ のくびれ (谷)",
        "desc": "語頭でF3が急降下してF2に接近し、差分ライン(Δ)が1000Hz破線より深く潜り込みます。舌の両側面を上顎の奥歯に密着させることでこの深い谷が生まれます。"
    },
    {
        "label": "Right (UK)",
        "file": "right_uk.wav",
        "title": "Right /raɪt/ (UK)",
        "phonetics": "/r/ の狭窄",
        "desc": "語頭でF3が急降下して1000Hz基準線を下回ります。その後の二重母音 /aɪ/ への移行部でF2が右肩上がりに上昇します。"
    },
    {
        "label": "Light (US)",
        "file": "light_us.wav",
        "title": "Light /laɪt/ (US)",
        "phonetics": "Dark L（軟口蓋化）による低F2・高Δ",
        "desc": "米語の語頭 /l/ は奥舌が引かれるため（Dark L傾向）、開始部で F2（緑）が 1200Hz 付近と低く、F3（青）との差分（Δ）が 2500Hz 以上に大きく開くのが特徴です。"
    },
    {
        "label": "Light (UK)",
        "file": "light_uk.wav",
        "title": "Light /laɪt/ (UK)",
        "phonetics": "F3高域維持 (Clear L)",
        "desc": "語頭の /l/ でF3は2700〜2800Hzの高域にとどまり、差分(Δ)は1000Hzを割らずに推移します。"
    },
    {
        "label": "White (UK)",
        "file": "white_uk.wav",
        "title": "White /waɪt/ (UK)",
        "phonetics": "円唇によるF2低下",
        "desc": "語頭で唇を丸めるためF2が一時的に下がりますが、F3は下がらないため、差分ラインはRのようなくびれになりません。"
    },
    {
        "label": "ライト (JP)",
        "file": "raito_jp.wav",
        "title": "日本語「ライト」/ɾaɪto/",
        "phonetics": "タップ破裂と語尾母音付加",
        "desc": "「ラ」で瞬間的な弾き音（破裂）が起き、語尾には英語の /t/ にはない母音「オ」のホルマントが明瞭に残ります。"
    },
    {
        "label": "Really (US)",
        "file": "really_us.wav",
        "title": "Really /ˈrɪli/ ~ /ˈriːli/ (US)",
        "phonetics": "/r/ の深谷から /l/ の天井へのジャンプ",
        "desc": "語頭の /r/ で青（F3）が緑（F2）へ沈み込み、黄色（Δ）が 500Hz 前後まで深く潜ります。そこから後半の /l/ に向かって F3 が 3000Hz 超へ跳ね上がるコントラストが特徴です。"
    },
    {
        "label": "Really (UK)",
        "file": "really_uk.wav",
        "title": "Really /ˈrɪəli/ (UK)",
        "phonetics": "センタリング二重母音を経由する滑らかな渡り",
        "desc": "前半の /r/ でしっかり 1000Hz 破線を潜る谷を作った後、イギリス英語特有の曖昧母音 /ə/ を挟みながら、後半の /l/ に向かって滑らかに F3 が立ち上がります。"
    },
    {
        "label": "Very (US)",
        "file": "very_us.wav",
        "title": "Very /ˈveri/ (US)",
        "phonetics": "有声摩擦 /v/ の振動持続 ＋ r-colored母音のくびれ",
        "desc": "語頭は有声摩擦 /v/ の規則的な声帯振動が続き、破裂音（b）のようなスパイクがありません。続く母音〜/r/ で F3 が引きずり下ろされ、Δ が 500Hz 未満の深い谷を作ります。"
    },
    {
        "label": "Very (UK)",
        "file": "very_uk.wav",
        "title": "Very /ˈveri/ (UK)",
        "phonetics": "/v/ から短母音を経て短く鋭い /r/ へ",
        "desc": "前半の摩擦音から母音 /e/ を経て、F3 が鋭く落ち込みます。米語に比べて母音のR音性化が少なく、谷の幅（持続時間）がコンパクトで引き締まっています。"
    }
]


def extract_scope_data_from_bytes(audio_bytes: bytes, max_formant_hz: float = 5000.0):
    audio_data, sample_rate = decode_audio_bytes(audio_bytes)

    snd = parselmouth.Sound(audio_data.astype(np.float64), sampling_frequency=float(sample_rate))
    pitch = snd.to_pitch()
    formant = snd.to_formant_burg(time_step=0.005, max_number_of_formants=5.0, maximum_formant=max_formant_hz)
    intensity = snd.to_intensity()

    times = formant.xs()
    f2_list, f3_list, delta_list = [], [], []
    valid_times = []

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

    if valid_times:
        x_min = max(0.0, float(min(valid_times)) - 0.08)
        x_max = min(float(snd.duration), float(max(valid_times)) + 0.25)
    else:
        x_min, x_max = 0.0, float(snd.duration)

    spec = snd.to_spectrogram(window_length=0.005, maximum_frequency=4200.0, time_step=0.003, frequency_step=25.0)
    spec_vals = 10.0 * np.log10(np.maximum(spec.values, 1e-12))
    spec_t = spec.xs()
    spec_f = spec.ys()

    t_mask = (spec_t >= x_min) & (spec_t <= x_max)
    f_mask = (spec_f >= 500.0) & (spec_f <= 4000.0)

    if np.any(t_mask) and np.any(f_mask):
        sub_spec = spec_vals[np.ix_(f_mask, t_mask)]
        sub_spec = np.flipud(sub_spec)
        max_db = np.max(sub_spec)
        dynamic_range = 45.0
        norm_spec = np.clip((sub_spec - (max_db - dynamic_range)) / dynamic_range, 0.0, 1.0)
        spec_u8 = (norm_spec * 255).astype(np.uint8)
        spec_h, spec_w = spec_u8.shape
        spec_b64 = base64.b64encode(spec_u8.tobytes()).decode("utf-8")
    else:
        spec_h, spec_w, spec_b64 = 0, 0, ""

    raw_samples = snd.values[0]
    raw_times = np.linspace(0, snd.duration, len(raw_samples))
    step = max(1, len(raw_samples) // 3000)
    wave_t = np.round(raw_times[::step], 4).tolist()
    wave_y = np.round(raw_samples[::step], 4).tolist()

    wav_io = io.BytesIO()
    sf.write(wav_io, audio_data, sample_rate, format='WAV')
    wav_b64 = base64.b64encode(wav_io.getvalue()).decode("utf-8")

    return {
        "duration": float(snd.duration),
        "x_min": round(x_min, 3),
        "x_max": round(x_max, 3),
        "times": np.round(times, 4).tolist(),
        "f2": f2_list,
        "f3": f3_list,
        "delta": delta_list,
        "wave_t": wave_t,
        "wave_y": wave_y,
        "wav_b64": wav_b64,
        "spec_w": spec_w,
        "spec_h": spec_h,
        "spec_b64": spec_b64
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

  // R Strike Zone Patch (<1000Hz)
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
if (D.auto_play) setTimeout(() => {{ cursorT = xMin; startPlayback(); }}, 1000);
</script>
</body>
</html>
"""


# =====================================================================
# MODE 2: E2 強弱リズム・母音長比プロトタイプ
# =====================================================================
def extract_rhythm_data_from_bytes(audio_bytes: bytes):
    audio_data, sample_rate = decode_audio_bytes(audio_bytes)
    snd = parselmouth.Sound(audio_data.astype(np.float64), sampling_frequency=float(sample_rate))
    intensity = snd.to_intensity(minimum_pitch=75.0, time_step=0.005)
    pitch = snd.to_pitch(time_step=0.005)

    int_times = intensity.xs()
    int_vals = intensity.values[0] if len(intensity.values) > 0 else np.array([])

    max_int = float(np.max(int_vals)) if len(int_vals) > 0 else 70.0
    min_int = max(30.0, max_int - 40.0)

    # 有声かつ一定音圧以上のフレームから音節チャンクを自動推定
    active_mask = np.zeros(len(int_times), dtype=bool)
    for i, t in enumerate(int_times):
        f0 = pitch.get_value_at_time(t)
        if not np.isnan(f0) and f0 > 0 and int_vals[i] >= max_int - 22.0:
            active_mask[i] = True

    segments = []
    in_seg = False
    seg_start = 0.0
    for i, active in enumerate(active_mask):
        t = float(int_times[i])
        if active and not in_seg:
            in_seg = True
            seg_start = t
        elif not active and in_seg:
            in_seg = False
            if t - seg_start >= 0.025:
                seg_slice = int_vals[(int_times >= seg_start) & (int_times <= t)]
                peak_db = float(np.max(seg_slice)) if len(seg_slice) > 0 else 0.0
                segments.append({"start": round(seg_start, 3), "end": round(t, 3), "peak_db": round(peak_db, 1)})
    if in_seg:
        t = float(int_times[-1])
        if t - seg_start >= 0.025:
            seg_slice = int_vals[(int_times >= seg_start) & (int_times <= t)]
            peak_db = float(np.max(seg_slice)) if len(seg_slice) > 0 else 0.0
            segments.append({"start": round(seg_start, 3), "end": round(t, 3), "peak_db": round(peak_db, 1)})

    # 全体表示範囲
    if segments:
        x_min = max(0.0, segments[0]["start"] - 0.12)
        x_max = min(float(snd.duration), segments[-1]["end"] + 0.18)
    else:
        x_min, x_max = 0.0, float(snd.duration)

    # 自動初期選択: 最大エネルギーの区間を Stressed、2番目または隣接区間を Unstressed にセット
    if len(segments) >= 2:
        sorted_segs = sorted(segments, key=lambda s: (s["end"] - s["start"]) * s["peak_db"], reverse=True)
        stressed_seg = sorted_segs[0]
        unstressed_seg = sorted_segs[1]
    elif len(segments) == 1:
        s0 = segments[0]
        mid = round((s0["start"] + s0["end"]) / 2, 3)
        stressed_seg = {"start": s0["start"], "end": mid}
        unstressed_seg = {"start": round(mid + 0.02, 3), "end": s0["end"]}
    else:
        dur = float(snd.duration)
        stressed_seg = {"start": round(dur * 0.2, 3), "end": round(dur * 0.45, 3)}
        unstressed_seg = {"start": round(dur * 0.55, 3), "end": round(dur * 0.65, 3)}

    raw_samples = snd.values[0]
    raw_times = np.linspace(0, snd.duration, len(raw_samples))
    step = max(1, len(raw_samples) // 3000)
    wave_t = np.round(raw_times[::step], 4).tolist()
    wave_y = np.round(raw_samples[::step], 4).tolist()

    wav_io = io.BytesIO()
    sf.write(wav_io, audio_data, sample_rate, format='WAV')
    wav_b64 = base64.b64encode(wav_io.getvalue()).decode("utf-8")

    return {
        "duration": float(snd.duration),
        "x_min": round(x_min, 3),
        "x_max": round(x_max, 3),
        "int_t": np.round(int_times, 4).tolist(),
        "int_y": np.round(int_vals, 1).tolist(),
        "max_int": round(max_int, 1),
        "min_int": round(min_int, 1),
        "stressed": stressed_seg,
        "unstressed": unstressed_seg,
        "wave_t": wave_t,
        "wave_y": wave_y,
        "wav_b64": wav_b64
    }


def build_rhythm_html(data: dict) -> str:
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
    font-family: inherit; font-size: 13px; font-weight: bold;
    padding: 8px 12px; border-radius: 6px;
    border: 1px solid #334155; cursor: pointer;
    background: #1E293B; color: #E2E8F0;
    transition: all 0.15s; white-space: nowrap;
  }}
  button:hover {{ background: #334155; color: #FFFFFF; }}
  button.btn-play-ready {{ background: #16A34A; color: #FFFFFF; border-color: #15803D; }}
  button.btn-play-active {{ background: #DC2626; color: #FFFFFF; border-color: #B91C1C; }}
  button.mode-stressed {{ background: rgba(245, 158, 11, 0.25); color: #FBBF24; border-color: #F59E0B; }}
  button.mode-unstressed {{ background: rgba(56, 189, 248, 0.25); color: #38BDF8; border-color: #0284C7; }}
  select {{
    font-family: inherit; font-size: 13px; padding: 6px 8px;
    border-radius: 6px; border: 1px solid #334155;
    background: #1E293B; color: #E2E8F0; font-weight: bold;
  }}
  .readout {{
    font-family: inherit; font-size: 15px; font-weight: 700; margin-left: auto;
    font-variant-numeric: tabular-nums; color: #F8FAFC; white-space: nowrap;
  }}
  #scope-wrapper {{
    position: relative; width: 100%; height: 560px;
    border: 1px solid #1E293B; border-top: none; background: #090D16;
  }}
  canvas {{ position: absolute; left: 0; top: 0; width: 100%; height: 100%; }}
</style>
</head>
<body>
<div class="toolbar">
  <button id="btn-play" class="btn-play-ready">▶ Play (Space)</button>
  <span style="color:#64748B; font-size:13px;">区間ドラッグ修正:</span>
  <button id="btn-drag-s" class="mode-stressed">🟧 Drag: Stressed (強母音)</button>
  <button id="btn-drag-u">🟦 Drag: Unstressed (弱母音)</button>
  <label style="font-size: 13px; font-weight: 500; display: flex; align-items: center; gap: 4px; color: #94A3B8;">
    Speed:
    <select id="sel-rate">
      <option value="1.0">1.0x</option>
      <option value="0.7">0.7x</option>
      <option value="0.5">0.5x</option>
    </select>
  </label>
  <div class="readout" id="ratio-readout">Ratio: --</div>
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
const btnDragS = document.getElementById('btn-drag-s');
const btnDragU = document.getElementById('btn-drag-u');
const selRate = document.getElementById('sel-rate');
const readout = document.getElementById('ratio-readout');

const MARGIN = {{ left: 110, right: 30, top: 20, bottom: 45 }};
let W = 1000, H = 560;
let xMin = D.x_min, xMax = D.x_max;

let sSpan = {{ start: D.stressed.start, end: D.stressed.end }};
let uSpan = {{ start: D.unstressed.start, end: D.unstressed.end }};
let editTarget = "stressed"; // "stressed" or "unstressed"
let cursorT = xMin;

let audioEl = new Audio();
let isPlaying = false;
let animId = null;

function initAudio() {{
  const binary = atob(D.wav_b64);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
  const blob = new Blob([bytes], {{ type: "audio/wav" }});
  audioEl.src = URL.createObjectURL(blob);
  audioEl.preservesPitch = true;
  audioEl.mozPreservesPitch = true;
  audioEl.webkitPreservesPitch = true;
}}
initAudio();

btnDragS.addEventListener("click", () => {{
  editTarget = "stressed";
  btnDragS.className = "mode-stressed";
  btnDragU.className = "";
}});
btnDragU.addEventListener("click", () => {{
  editTarget = "unstressed";
  btnDragU.className = "mode-unstressed";
  btnDragS.className = "";
}});

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
  const gap = 24;
  const h1 = 150, h2 = 160, h3 = totalH - h1 - h2 - (gap * 2);
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

function drawAll() {{
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
  const APP_FONT = '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif';

  // グリッドと時間目盛り (p1, p2)
  const durationSpan = xMax - xMin;
  const stepT = durationSpan <= 0.6 ? 0.05 : (durationSpan <= 1.5 ? 0.1 : 0.2);
  const firstTick = Math.ceil(xMin / stepT) * stepT;
  bgCtx.font = "13px " + APP_FONT;

  [p1, p2].forEach((p) => {{
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

  // --- 第1段: 生波形 ---
  const maxAmp = Math.max(...D.wave_y.map(Math.abs), 0.05);
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

  // --- 第2段: Intensity (dB) ---
  const iMin = D.min_int, iMax = D.max_int + 5;
  bgCtx.save();
  bgCtx.beginPath(); bgCtx.rect(MARGIN.left, p2.top, plotW, p2.height); bgCtx.clip();
  bgCtx.strokeStyle = "#10B981"; bgCtx.lineWidth = 3.0;
  bgCtx.beginPath();
  for (let i = 0; i < D.int_t.length; i++) {{
    const x = tToX(D.int_t[i]);
    const y = valToY(D.int_y[i], iMin, iMax, p2);
    if (i === 0) bgCtx.moveTo(x, y); else bgCtx.lineTo(x, y);
  }}
  bgCtx.stroke(); bgCtx.restore();

  // --- Stressed / Unstressed 選択区間のオーバーレイ (p1, p2) ---
  const sDurMs = Math.max(1, Math.round(Math.abs(sSpan.end - sSpan.start) * 1000));
  const uDurMs = Math.max(1, Math.round(Math.abs(uSpan.end - uSpan.start) * 1000));
  const ratio = (sDurMs / uDurMs).toFixed(2);

  function drawSpanBox(span, colorFill, colorStroke, labelText) {{
    const sx = tToX(Math.min(span.start, span.end));
    const ex = tToX(Math.max(span.start, span.end));
    [p1, p2].forEach(p => {{
      bgCtx.fillStyle = colorFill;
      bgCtx.fillRect(sx, p.top, ex - sx, p.height);
      bgCtx.strokeStyle = colorStroke; bgCtx.lineWidth = 2.0;
      bgCtx.beginPath(); bgCtx.moveTo(sx, p.top); bgCtx.lineTo(sx, p.bottom); bgCtx.moveTo(ex, p.top); bgCtx.lineTo(ex, p.bottom); bgCtx.stroke();
    }});
    bgCtx.font = "bold 13px " + APP_FONT;
    bgCtx.fillStyle = colorStroke; bgCtx.textAlign = "center"; bgCtx.textBaseline = "top";
    bgCtx.fillText(labelText, (sx + ex) / 2, p1.top + 6);
  }}

  drawSpanBox(sSpan, "rgba(245, 158, 11, 0.22)", "#F59E0B", `Stressed: ${{sDurMs}} ms`);
  drawSpanBox(uSpan, "rgba(56, 189, 248, 0.22)", "#38BDF8", `Unstressed: ${{uDurMs}} ms`);

  // --- 第3段: Duration Ratio ストライクゾーン横ゲージ (0.0x ~ 5.0x) ---
  const rMax = 5.0;
  function rToX(r) {{
    return MARGIN.left + (Math.min(rMax, Math.max(0, r)) / rMax) * plotW;
  }}

  // MORA TRAP Zone (< 1.5x)
  const xMora = rToX(1.5);
  bgCtx.fillStyle = "rgba(239, 68, 68, 0.12)";
  bgCtx.fillRect(MARGIN.left, p3.top, xMora - MARGIN.left, p3.height);
  bgCtx.font = "bold 12px " + APP_FONT; bgCtx.fillStyle = "#EF4444"; bgCtx.textAlign = "left"; bgCtx.textBaseline = "top";
  bgCtx.fillText("MORA TRAP (1:1 均等読み <1.5x)", MARGIN.left + 8, p3.top + 8);

  // STRIKE ZONE (2.5x ~ 5.0x)
  const xStrike = rToX(2.5);
  bgCtx.fillStyle = "rgba(16, 185, 129, 0.14)";
  bgCtx.fillRect(xStrike, p3.top, (W - MARGIN.right) - xStrike, p3.height);
  bgCtx.strokeStyle = "#10B981"; bgCtx.lineWidth = 2.0; bgCtx.setLineDash([6, 4]);
  bgCtx.beginPath(); bgCtx.moveTo(xStrike, p3.top); bgCtx.lineTo(xStrike, p3.bottom); bgCtx.stroke(); bgCtx.setLineDash([]);
  bgCtx.fillStyle = "#10B981"; bgCtx.textAlign = "right";
  bgCtx.fillText("STRESS-TIMED STRIKE ZONE (≥ 2.5 : 1)", W - MARGIN.right - 10, p3.top + 8);

  // 現在の Ratio バー描画
  const barY = p3.top + p3.height * 0.45;
  const barH = p3.height * 0.35;
  const curX = rToX(parseFloat(ratio));
  bgCtx.fillStyle = parseFloat(ratio) >= 2.5 ? "#10B981" : (parseFloat(ratio) < 1.5 ? "#EF4444" : "#F59E0B");
  bgCtx.fillRect(MARGIN.left, barY, Math.max(4, curX - MARGIN.left), barH);

  // Ratio 目盛り (0x, 1x, 2x, 2.5x, 3x, 4x, 5x)
  [0, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0].forEach(r => {{
    const rx = rToX(r);
    bgCtx.strokeStyle = "#475569"; bgCtx.lineWidth = 1.5;
    bgCtx.beginPath(); bgCtx.moveTo(rx, p3.bottom); bgCtx.lineTo(rx, p3.bottom + 5); bgCtx.stroke();
    bgCtx.fillStyle = "#94A3B8"; bgCtx.font = "13px " + APP_FONT; bgCtx.textAlign = "center"; bgCtx.textBaseline = "top";
    bgCtx.fillText(r.toFixed(1) + " : 1", rx, p3.bottom + 6);
  }});

  panels.forEach(p => {{
    bgCtx.strokeStyle = "#334155"; bgCtx.lineWidth = 1.5;
    bgCtx.beginPath(); bgCtx.moveTo(MARGIN.left, p.top); bgCtx.lineTo(MARGIN.left, p.bottom); bgCtx.lineTo(W - MARGIN.right, p.bottom); bgCtx.stroke();
  }});

  bgCtx.font = "bold 15px " + APP_FONT; bgCtx.fillStyle = "#CBD5E1";
  drawRotatedLabel("Waveform", 28, (p1.top + p1.bottom) / 2);
  drawRotatedLabel("Intensity (dB)", 28, (p2.top + p2.bottom) / 2);
  drawRotatedLabel("Ratio Gauge", 28, (p3.top + p3.bottom) / 2);

  let statusColor = parseFloat(ratio) >= 2.5 ? "#10B981" : (parseFloat(ratio) < 1.5 ? "#EF4444" : "#F59E0B");
  let verdict = parseFloat(ratio) >= 2.5 ? "STRIKE (英語リズム)" : (parseFloat(ratio) < 1.5 ? "MORA (日本語等時拍)" : "惜しい (あと少し短縮)");
  readout.innerHTML = `<span style="color:#F59E0B;">強 ${{sDurMs}}ms</span> / <span style="color:#38BDF8;">弱 ${{uDurMs}}ms</span> = <span style="color:${{statusColor}}; font-size:18px;">${{ratio}} : 1 (${{verdict}})</span>`;

  drawCursorOverlay();
}}

function drawRotatedLabel(text, x, y) {{
  bgCtx.save(); bgCtx.translate(x, y); bgCtx.rotate(-Math.PI / 2);
  bgCtx.textAlign = "center"; bgCtx.textBaseline = "middle"; bgCtx.fillText(text, 0, 0); bgCtx.restore();
}}

function drawCursorOverlay() {{
  ovCtx.clearRect(0, 0, W, H);
  const panels = getPanels();
  const cx = tToX(cursorT);
  ovCtx.strokeStyle = "#EF4444"; ovCtx.lineWidth = 2.0;
  ovCtx.beginPath(); ovCtx.moveTo(cx, panels[0].top); ovCtx.lineTo(cx, panels[1].bottom); ovCtx.stroke();
}}

function startPlayback() {{
  if (!audioEl.src) return; stopPlayback();
  audioEl.playbackRate = parseFloat(selRate.value);
  audioEl.preservesPitch = true;
  audioEl.currentTime = xMin;
  isPlaying = true; btnPlay.textContent = "⏹ Stop (Space)"; btnPlay.className = "btn-play-active";
  audioEl.play().catch(() => stopPlayback());

  function tick() {{
    if (!isPlaying) return;
    cursorT = audioEl.currentTime;
    if (cursorT >= xMax || audioEl.ended) {{
      stopPlayback(); return;
    }}
    drawCursorOverlay(); animId = requestAnimationFrame(tick);
  }}
  animId = requestAnimationFrame(tick);
}}
function stopPlayback() {{
  audioEl.pause(); if (animId) cancelAnimationFrame(animId);
  isPlaying = false; btnPlay.textContent = "▶ Play (Space)"; btnPlay.className = "btn-play-ready";
}}
btnPlay.addEventListener("click", () => isPlaying ? stopPlayback() : startPlayback());

let isDragging = false;
ovCanvas.addEventListener("mousedown", (e) => {{
  stopPlayback();
  const rect = ovCanvas.getBoundingClientRect();
  const clickT = xToT(e.clientX - rect.left);
  isDragging = true;
  if (editTarget === "stressed") {{ sSpan.start = clickT; sSpan.end = clickT + 0.01; }}
  else {{ uSpan.start = clickT; uSpan.end = clickT + 0.01; }}
  cursorT = clickT;
  drawAll();
}});
ovCanvas.addEventListener("mousemove", (e) => {{
  if (!isDragging) return;
  const rect = ovCanvas.getBoundingClientRect();
  const curT = xToT(e.clientX - rect.left);
  if (editTarget === "stressed") sSpan.end = curT;
  else uSpan.end = curT;
  cursorT = curT;
  drawAll();
}});
window.addEventListener("mouseup", () => {{
  if (isDragging) {{
    isDragging = false;
    if (editTarget === "stressed" && sSpan.end < sSpan.start) {{
      const tmp = sSpan.start; sSpan.start = sSpan.end; sSpan.end = tmp;
    }}
    if (editTarget === "unstressed" && uSpan.end < uSpan.start) {{
      const tmp = uSpan.start; uSpan.start = uSpan.end; uSpan.end = tmp;
    }}
    drawAll();
  }}
}});
window.addEventListener("resize", drawAll);
drawAll();
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
        if col.button(p_info["label"], use_container_width=True):
            file_path = os.path.join(PRESET_DIR, p_info["file"])
            if os.path.exists(file_path):
                with open(file_path, "rb") as f:
                    st.session_state.current_audio_bytes = f.read()
                    st.session_state.current_label = f"お手本: {p_info['label']}"
                    st.session_state.auto_play = True
                preset_clicked = True
                st.rerun()
            else:
                st.warning(f"`{file_path}` が見つかりません。")

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
        html_code = build_spike2_html(
            scope_data,
            auto_play=st.session_state.auto_play,
            export_name=clean_export_name or "sample"
        )
        st.session_state.auto_play = False
        components.html(html_code, height=640, scrolling=False)

        info = next((p for p in PRESETS if p["label"] in st.session_state.current_label), None)
        if info:
            st.markdown(f"""
            <div style="background: #0F172A; border: 1px solid #1E293B; border-left: 3px solid #64748B; border-radius: 6px; padding: 12px 18px; margin-top: 10px;">
                <div style="display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 6px;">
                    <span style="font-size: 1.05rem; font-weight: 600; color: #F8FAFC;">{info['title']}</span>
                    <span style="font-size: 0.85rem; font-weight: 500; color: #94A3B8; background: #1E293B; padding: 2px 8px; border-radius: 4px; border: 1px solid #334155;">{info['phonetics']}</span>
                </div>
                <p style="font-size: 0.9rem; color: #94A3B8; margin: 0; line-height: 1.5;">{info['desc']}</p>
            </div>
            """, unsafe_allow_html=True)

else:
    # --- E2: 強弱リズム・母音長比プロトタイプ画面 ---
    st.markdown(
        "<div style='color:#94A3B8; font-size:0.95rem; margin-bottom:8px;'>"
        "💡 <b>テスト方法</b>: マイクで「<b>ba-NA-na</b>（英語リズム）」や「<b>バ・ナ・ナ</b>（日本語リズム）」、「<b>CA-m-ra</b>」などを録音してください。自動で強母音と弱母音を検出し、比率をストライクゾーン判定します（波形上のドラッグで手動調整も可能）。"
        "</div>",
        unsafe_allow_html=True
    )
    rhythm_input = st.audio_input("自分の声を録音してリズム・母音長比をテスト")
    if rhythm_input is not None:
        st.session_state.rhythm_audio_bytes = rhythm_input.read()

    if st.session_state.rhythm_audio_bytes is not None:
        r_data = extract_rhythm_data_from_bytes(st.session_state.rhythm_audio_bytes)
        r_html = build_rhythm_html(r_data)
        components.html(r_html, height=620, scrolling=False)