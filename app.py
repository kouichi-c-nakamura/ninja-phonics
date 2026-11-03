import streamlit as st
import streamlit.components.v1 as components
import parselmouth
import soundfile as sf
import numpy as np
import base64
import json
import io
import os

st.set_page_config(page_title="NinjaPhonetics Scope", layout="wide", initial_sidebar_state="collapsed")

# 画面全体をスクロール不要な1ビューに収めるためのCSS
st.markdown("""
<style>
    .block-container {
        padding-top: 0.8rem !important;
        padding-bottom: 0.2rem !important;
        padding-left: 1.5rem !important;
        padding-right: 1.5rem !important;
    }
    h1 {
        font-size: 1.6rem !important;
        margin-bottom: 0.4rem !important;
        color: #F8FAFC !important;
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
        padding: 6px 10px !important;
        border-radius: 6px !important;
    }
    .stButton button:hover {
        background-color: #334155 !important;
        border-color: #38BDF8 !important;
        color: #38BDF8 !important;
    }
</style>
""", unsafe_allow_html=True)

st.title("🥷 NinjaPhonetics: Biofeedback Scope")

if "current_audio_bytes" not in st.session_state:
    st.session_state.current_audio_bytes = None
if "current_label" not in st.session_state:
    st.session_state.current_label = ""
if "last_recorded_bytes" not in st.session_state:
    st.session_state.last_recorded_bytes = None

# --- お手本音声プリセット設定（6種） ---
PRESET_DIR = "presets"
PRESETS = [
    {"label": "🇺🇸 Right (US)", "file": "right_us.wav"},
    {"label": "🇬🇧 Right (UK)", "file": "right_uk.wav"},
    {"label": "🇬🇧 Light (UK)", "file": "light_uk.wav"},
    {"label": "⚪ White (UK)", "file": "white_uk.wav"},
    {"label": "🇯🇵 ライト (JP)", "file": "raito_jp.wav"},
]

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
            preset_clicked = True
            st.rerun()

# マイク入力
audio_input = st.audio_input("🎙️ または自分の声を録音して解析")
if not preset_clicked and audio_input is not None:
    new_bytes = audio_input.read()
    if new_bytes != st.session_state.last_recorded_bytes:
        st.session_state.last_recorded_bytes = new_bytes
        st.session_state.current_audio_bytes = new_bytes
        st.session_state.current_label = "あなたの発音録音"


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
    f2_list, f3_list, delta_list = [], [], []
    valid_times = []

    max_int = np.max(intensity.values) if len(intensity.values) > 0 else 0
    threshold_int = max(30.0, max_int - 28.0)

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
        x_max = min(float(snd.duration), float(max(valid_times)) + 0.08)
    else:
        x_min, x_max = 0.0, float(snd.duration)

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
        "wav_b64": wav_b64
    }


def build_spike2_html(data: dict) -> str:
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
    user-select: none;
    overflow: hidden;
  }}
  .toolbar {{
    display: flex; align-items: center; gap: 14px;
    padding: 10px 16px; background: #0F172A;
    border: 1px solid #1E293B; border-radius: 8px 8px 0 0;
  }}
  button {{
    font-size: 16px; font-weight: bold;
    padding: 8px 18px; border-radius: 6px;
    border: 1px solid #334155; cursor: pointer;
    background: #1E293B; color: #E2E8F0;
    transition: all 0.15s;
    white-space: nowrap;
  }}
  button.primary {{ background: #2563EB; color: white; border-color: #3B82F6; }}
  button.primary:hover {{ background: #1D4ED8; }}
  button:hover {{ background: #334155; color: white; }}
  
  select {{
    font-size: 15px; padding: 6px 10px;
    border-radius: 6px; border: 1px solid #334155;
    background: #1E293B; color: #E2E8F0;
    font-weight: bold;
  }}
  .readout {{
    font-size: 17px; font-weight: bold; margin-left: auto;
    font-variant-numeric: tabular-nums; color: #38BDF8;
    white-space: nowrap;
  }}
  #scope-wrapper {{
    position: relative; width: 100%; height: 580px;
    border: 1px solid #1E293B; border-top: none;
    background: #090D16;
  }}
  canvas {{
    position: absolute; left: 0; top: 0;
    width: 100%; height: 100%;
  }}
  #overlay-canvas {{
    mix-blend-mode: screen;
  }}
</style>
</head>
<body>

<div class="toolbar">
  <button id="btn-play" class="primary">▶ Play (Space)</button>
  <button id="btn-reset">⏮ 全体選択</button>
  <label style="font-size: 15px; font-weight: bold; display: flex; align-items: center; gap: 6px; color: #94A3B8;">
    Speed:
    <select id="sel-rate">
      <option value="1.0">1.0x (標準)</option>
      <option value="0.7">0.7x (スロー)</option>
      <option value="0.5">0.5x (極スロー)</option>
    </select>
  </label>
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
const selRate = document.getElementById('sel-rate');
const readout = document.getElementById('time-readout');

const MARGIN = {{ left: 110, right: 30, top: 20, bottom: 45 }};
let W = 1000, H = 580;

let xMin = D.x_min;
let xMax = D.x_max;
let selStart = xMin;
let selEnd = xMax;
let cursorT = (xMin + xMax) / 2;

let audioCtx = null;
let audioBuffer = null;
let sourceNode = null;
let isPlaying = false;
let playStartWallTime = 0;
let playStartOffset = 0;
let playbackRate = 1.0;
let animId = null;

async function initAudio() {{
  audioCtx = new (window.AudioContext || window.webkitAudioContext)();
  const binary = atob(D.wav_b64);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
  audioBuffer = await audioCtx.decodeAudioData(bytes.buffer);
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
  const gap = 24;
  const h1 = 120; // 1段目の波形
  const remainH = totalH - h1 - (gap * 2);
  const h2 = remainH * 0.50; // F3/F2
  const h3 = remainH * 0.50; // 差分

  const y1 = MARGIN.top;
  const y2 = y1 + h1 + gap;
  const y3 = y2 + h2 + gap;
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
  W = wrapper.clientWidth;
  H = wrapper.clientHeight;
  bgCanvas.width = W * dpr; bgCanvas.height = H * dpr;
  ovCanvas.width = W * dpr; ovCanvas.height = H * dpr;
  bgCtx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ovCtx.setTransform(dpr, 0, 0, dpr, 0, 0);

  bgCtx.fillStyle = "#090D16";
  bgCtx.fillRect(0, 0, W, H);

  const panels = getPanels();
  const plotW = W - MARGIN.left - MARGIN.right;

  bgCtx.font = "14px monospace";
  bgCtx.fillStyle = "#64748B";

  const durationSpan = xMax - xMin;
  const stepT = durationSpan <= 0.6 ? 0.05 : (durationSpan <= 1.5 ? 0.1 : 0.2);
  const firstTick = Math.ceil(xMin / stepT) * stepT;

  panels.forEach((p) => {{
    for (let t = firstTick; t <= xMax + 1e-6; t += stepT) {{
      const x = tToX(t);

      // 暗色グリッド
      bgCtx.strokeStyle = "#172033";
      bgCtx.lineWidth = 1.0;
      bgCtx.beginPath();
      bgCtx.moveTo(x, p.top);
      bgCtx.lineTo(x, p.bottom);
      bgCtx.stroke();

      // 外向き目盛り線
      bgCtx.strokeStyle = "#475569";
      bgCtx.lineWidth = 1.5;
      bgCtx.beginPath();
      bgCtx.moveTo(x, p.bottom);
      bgCtx.lineTo(x, p.bottom + 5);
      bgCtx.stroke();

      // 目盛り数値
      bgCtx.fillStyle = "#94A3B8";
      bgCtx.textAlign = "center";
      bgCtx.textBaseline = "top";
      bgCtx.fillText(t.toFixed(2), x, p.bottom + 6);
    }}
  }});

  // --- 第1段: 生波形 (シアン発光) ---
  const p1 = panels[0];
  const waveMid = (p1.top + p1.bottom) / 2;
  bgCtx.strokeStyle = "#1E293B";
  bgCtx.beginPath(); bgCtx.moveTo(MARGIN.left, waveMid); bgCtx.lineTo(W - MARGIN.right, waveMid); bgCtx.stroke();

  const maxAmp = Math.max(...D.wave_y.map(Math.abs), 0.05);
  [-1.0, 0.0, 1.0].forEach(relVal => {{
    const absVal = relVal * maxAmp;
    const y = valToY(absVal, -maxAmp * 1.1, maxAmp * 1.1, p1);
    bgCtx.strokeStyle = "#334155"; bgCtx.lineWidth = 1.5;
    bgCtx.beginPath(); bgCtx.moveTo(MARGIN.left - 5, y); bgCtx.lineTo(MARGIN.left, y); bgCtx.stroke();
    bgCtx.textAlign = "right"; bgCtx.textBaseline = "middle";
    bgCtx.fillText(relVal.toFixed(1), MARGIN.left - 8, y);
  }});

  bgCtx.save();
  bgCtx.beginPath(); bgCtx.rect(MARGIN.left, p1.top, plotW, p1.height); bgCtx.clip();
  bgCtx.strokeStyle = "#38BDF8";
  bgCtx.lineWidth = 1.5;
  bgCtx.beginPath();
  for (let i = 0; i < D.wave_t.length; i++) {{
    const x = tToX(D.wave_t[i]);
    const y = valToY(D.wave_y[i], -maxAmp * 1.1, maxAmp * 1.1, p1);
    if (i === 0) bgCtx.moveTo(x, y); else bgCtx.lineTo(x, y);
  }}
  bgCtx.stroke();
  bgCtx.restore();

  // --- 第2段: F3 / F2 (ブルー ＆ エメラルド) ---
  const p2 = panels[1];
  const fMin = 500, fMax = 3200;
  [1000, 1500, 2000, 2500, 3000].forEach(hz => {{
    const y = valToY(hz, fMin, fMax, p2);
    bgCtx.strokeStyle = "#172033"; bgCtx.lineWidth = 1.0;
    bgCtx.beginPath(); bgCtx.moveTo(MARGIN.left, y); bgCtx.lineTo(W - MARGIN.right, y); bgCtx.stroke();
    bgCtx.strokeStyle = "#475569"; bgCtx.lineWidth = 1.5;
    bgCtx.beginPath(); bgCtx.moveTo(MARGIN.left - 5, y); bgCtx.lineTo(MARGIN.left, y); bgCtx.stroke();
    bgCtx.textAlign = "right"; bgCtx.textBaseline = "middle";
    bgCtx.fillText(hz.toString(), MARGIN.left - 8, y);
  }});

  bgCtx.save();
  bgCtx.beginPath(); bgCtx.rect(MARGIN.left, p2.top, plotW, p2.height); bgCtx.clip();
  drawSeries(D.times, D.f3, fMin, fMax, p2, "#60A5FA", 3.0);
  drawSeries(D.times, D.f2, fMin, fMax, p2, "#34D399", 3.0);
  bgCtx.restore();

  // 凡例 (縦並び)
  const legendX = W - MARGIN.right - 80;
  const legendY1 = p2.top + 18;
  const legendY2 = p2.top + 38;
  bgCtx.font = "bold 15px -apple-system, sans-serif";

  bgCtx.strokeStyle = "#60A5FA"; bgCtx.lineWidth = 3.0;
  bgCtx.beginPath(); bgCtx.moveTo(legendX, legendY1); bgCtx.lineTo(legendX + 18, legendY1); bgCtx.stroke();
  bgCtx.fillStyle = "#60A5FA"; bgCtx.textAlign = "left"; bgCtx.textBaseline = "middle";
  bgCtx.fillText("F3", legendX + 24, legendY1);

  bgCtx.strokeStyle = "#34D399"; bgCtx.lineWidth = 3.0;
  bgCtx.beginPath(); bgCtx.moveTo(legendX, legendY2); bgCtx.lineTo(legendX + 18, legendY2); bgCtx.stroke();
  bgCtx.fillStyle = "#34D399";
  bgCtx.fillText("F2", legendX + 24, legendY2);

  // --- 第3段: Δ(F3 - F2) (アンバーゴールド) ---
  const p3 = panels[2];
  const dMin = 0, dMax = 2500;
  [0, 500, 1000, 1500, 2000, 2500].forEach(hz => {{
    const y = valToY(hz, dMin, dMax, p3);
    bgCtx.strokeStyle = "#172033"; bgCtx.lineWidth = 1.0;
    bgCtx.beginPath(); bgCtx.moveTo(MARGIN.left, y); bgCtx.lineTo(W - MARGIN.right, y); bgCtx.stroke();
    bgCtx.strokeStyle = "#475569"; bgCtx.lineWidth = 1.5;
    bgCtx.beginPath(); bgCtx.moveTo(MARGIN.left - 5, y); bgCtx.lineTo(MARGIN.left, y); bgCtx.stroke();
    bgCtx.textAlign = "right"; bgCtx.textBaseline = "middle";
    bgCtx.fillText(hz.toString(), MARGIN.left - 8, y);
  }});

  // R Target 破線 (1000Hz)
  const yTarget = valToY(1000, dMin, dMax, p3);
  bgCtx.save();
  bgCtx.strokeStyle = "#64748B"; bgCtx.lineWidth = 1.5; bgCtx.setLineDash([6, 5]);
  bgCtx.beginPath(); bgCtx.moveTo(MARGIN.left, yTarget); bgCtx.lineTo(W - MARGIN.right, yTarget); bgCtx.stroke();
  bgCtx.restore();
  bgCtx.font = "14px sans-serif";
  bgCtx.fillStyle = "#94A3B8"; bgCtx.textAlign = "right"; bgCtx.textBaseline = "bottom";
  bgCtx.fillText("R Target (<1000Hz)", W - MARGIN.right - 6, yTarget - 3);

  bgCtx.save();
  bgCtx.beginPath(); bgCtx.rect(MARGIN.left, p3.top, plotW, p3.height); bgCtx.clip();
  drawSeries(D.times, D.delta, dMin, dMax, p3, "#F59E0B", 3.5);
  bgCtx.restore();

  // 枠線
  panels.forEach(p => {{
    bgCtx.strokeStyle = "#334155"; bgCtx.lineWidth = 1.5;
    bgCtx.beginPath();
    bgCtx.moveTo(MARGIN.left, p.top);
    bgCtx.lineTo(MARGIN.left, p.bottom);
    bgCtx.lineTo(W - MARGIN.right, p.bottom);
    bgCtx.stroke();
  }});

  // Y軸タイトル
  bgCtx.fillStyle = "#CBD5E1";
  bgCtx.font = "bold 16px -apple-system, sans-serif";
  drawRotatedLabel("Waveform", 28, (p1.top + p1.bottom) / 2);
  drawRotatedLabel("F3 / F2 (Hz)", 28, (p2.top + p2.bottom) / 2);
  drawRotatedLabel("Δ(F3 - F2)", 28, (p3.top + p3.bottom) / 2);

  // X軸タイトル
  bgCtx.textAlign = "center"; bgCtx.textBaseline = "bottom";
  bgCtx.fillText("Time (s)", MARGIN.left + plotW / 2, H - 2);
}}

function drawSeries(times, values, vMin, vMax, panel, color, width) {{
  bgCtx.strokeStyle = color; bgCtx.lineWidth = width; bgCtx.lineJoin = "round";
  bgCtx.beginPath();
  let penDown = false;
  for (let i = 0; i < times.length; i++) {{
    const val = values[i];
    if (val === null || val === undefined) {{ penDown = false; continue; }}
    const x = tToX(times[i]);
    const y = valToY(val, vMin, vMax, panel);
    if (!penDown) {{ bgCtx.moveTo(x, y); penDown = true; }}
    else {{ bgCtx.lineTo(x, y); }}
  }}
  bgCtx.stroke();
}}

function drawRotatedLabel(text, x, y) {{
  bgCtx.save();
  bgCtx.translate(x, y);
  bgCtx.rotate(-Math.PI / 2);
  bgCtx.textAlign = "center"; bgCtx.textBaseline = "middle";
  bgCtx.fillText(text, 0, 0);
  bgCtx.restore();
}}

function drawOverlay() {{
  ovCtx.clearRect(0, 0, W, H);
  const panels = getPanels();
  const topY = panels[0].top;
  const botY = panels[2].bottom;

  // 選択ハイライト（ネオンシアンの半透明ベール）
  if (Math.abs(selEnd - selStart) > 0.005) {{
    const sx = tToX(Math.min(selStart, selEnd));
    const ex = tToX(Math.max(selStart, selEnd));

    ovCtx.fillStyle = "rgba(14, 165, 233, 0.20)";
    ovCtx.fillRect(sx, topY, ex - sx, botY - topY);

    ovCtx.strokeStyle = "#38BDF8";
    ovCtx.lineWidth = 1.5;
    ovCtx.beginPath();
    ovCtx.moveTo(sx, topY); ovCtx.lineTo(sx, botY);
    ovCtx.moveTo(ex, topY); ovCtx.lineTo(ex, botY);
    ovCtx.stroke();
  }}

  // クリムゾンレッド縦カーソル
  const cx = tToX(cursorT);
  ovCtx.strokeStyle = "#F43F5E";
  ovCtx.lineWidth = 2.0;
  ovCtx.beginPath();
  ovCtx.moveTo(cx, topY);
  ovCtx.lineTo(cx, botY);
  ovCtx.stroke();

  // カーソルつまみ
  ovCtx.fillStyle = "#F43F5E";
  ovCtx.beginPath();
  ovCtx.moveTo(cx - 6, topY); ovCtx.lineTo(cx + 6, topY); ovCtx.lineTo(cx, topY + 8); ovCtx.closePath();
  ovCtx.fill();
  ovCtx.beginPath();
  ovCtx.moveTo(cx - 6, botY); ovCtx.lineTo(cx + 6, botY); ovCtx.lineTo(cx, botY - 8); ovCtx.closePath();
  ovCtx.fill();

  let idx = D.times.findIndex(t => t >= cursorT);
  let info = `Cursor: ${{cursorT.toFixed(3)}} s`;
  if (idx !== -1 && D.delta[idx] !== null) {{
    info += `  |  F3: ${{D.f3[idx]}} Hz   F2: ${{D.f2[idx]}} Hz   Δ: ${{D.delta[idx]}} Hz`;
  }}
  readout.textContent = info;
}}

function startPlayback() {{
  if (!audioBuffer) return;
  if (audioCtx.state === "suspended") audioCtx.resume();
  stopPlayback();

  playbackRate = parseFloat(selRate.value);
  const s = Math.min(selStart, selEnd);
  const e = Math.max(selStart, selEnd);
  const playFrom = (cursorT >= s && cursorT < e - 0.02) ? cursorT : s;
  const playDuration = Math.max(0.01, e - playFrom);

  sourceNode = audioCtx.createBufferSource();
  sourceNode.buffer = audioBuffer;
  sourceNode.playbackRate.value = playbackRate;
  sourceNode.connect(audioCtx.destination);

  playStartWallTime = audioCtx.currentTime;
  playStartOffset = playFrom;
  isPlaying = true;
  btnPlay.textContent = "⏸ Pause (Space)";

  sourceNode.start(0, playFrom, playDuration);
  sourceNode.onended = () => {{
    if (isPlaying) {{
      isPlaying = false;
      cursorT = e;
      btnPlay.textContent = "▶ Play (Space)";
      drawOverlay();
    }}
  }};

  function tick() {{
    if (!isPlaying) return;
    const elapsed = (audioCtx.currentTime - playStartWallTime) * playbackRate;
    cursorT = playStartOffset + elapsed;
    if (cursorT >= e) {{
      cursorT = e;
      drawOverlay();
      return;
    }}
    drawOverlay();
    animId = requestAnimationFrame(tick);
  }}
  animId = requestAnimationFrame(tick);
}}

function stopPlayback() {{
  if (sourceNode) {{
    try {{ sourceNode.onended = null; sourceNode.stop(); }} catch (err) {{}}
    sourceNode = null;
  }}
  if (animId) cancelAnimationFrame(animId);
  isPlaying = false;
  btnPlay.textContent = "▶ Play (Space)";
}}

btnPlay.addEventListener("click", () => {{
  if (isPlaying) stopPlayback();
  else startPlayback();
}});

btnReset.addEventListener("click", () => {{
  stopPlayback();
  selStart = xMin;
  selEnd = xMax;
  cursorT = (xMin + xMax) / 2;
  drawOverlay();
}});

window.addEventListener("keydown", (e) => {{
  if (e.code === "Space" && e.target === document.body) {{
    e.preventDefault();
    if (isPlaying) stopPlayback();
    else startPlayback();
  }}
}});

let dragMode = "none";
function isNearCursor(mouseX) {{
  const cx = tToX(cursorT);
  return Math.abs(mouseX - cx) <= 12;
}}

ovCanvas.addEventListener("mousemove", (e) => {{
  const rect = ovCanvas.getBoundingClientRect();
  const mouseX = e.clientX - rect.left;

  if (dragMode === "none") {{
    if (isNearCursor(mouseX)) ovCanvas.style.cursor = "ew-resize";
    else ovCanvas.style.cursor = "crosshair";
  }} else if (dragMode === "cursor") {{
    cursorT = xToT(mouseX);
    drawOverlay();
  }} else if (dragMode === "select") {{
    selEnd = xToT(mouseX);
    cursorT = selEnd;
    drawOverlay();
  }}
}});

ovCanvas.addEventListener("mousedown", (e) => {{
  stopPlayback();
  const rect = ovCanvas.getBoundingClientRect();
  const mouseX = e.clientX - rect.left;
  const clickT = xToT(mouseX);

  if (isNearCursor(mouseX)) {{
    dragMode = "cursor";
    ovCanvas.style.cursor = "ew-resize";
  }} else {{
    dragMode = "select";
    selStart = clickT;
    selEnd = clickT;
    cursorT = clickT;
    ovCanvas.style.cursor = "crosshair";
    drawOverlay();
  }}
}});

window.addEventListener("mouseup", () => {{
  if (dragMode === "select") {{
    if (Math.abs(selEnd - selStart) < 0.015) {{
      selStart = xMin;
      selEnd = xMax;
    }} else {{
      cursorT = Math.min(selStart, selEnd);
    }}
  }}
  dragMode = "none";
  drawOverlay();
}});

window.addEventListener("resize", () => {{
  drawBackground();
  drawOverlay();
}});

drawBackground();
drawOverlay();
</script>
</body>
</html>
"""

if st.session_state.current_audio_bytes is not None:
    if st.session_state.current_label:
        st.markdown(f"<span style='color:#38BDF8; font-weight:bold;'>📊 表示中: {st.session_state.current_label}</span>", unsafe_allow_html=True)
    scope_data = extract_scope_data_from_bytes(st.session_state.current_audio_bytes)
    html_code = build_spike2_html(scope_data)
    components.html(html_code, height=640, scrolling=False)