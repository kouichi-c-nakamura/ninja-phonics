import streamlit as st
import streamlit.components.v1 as components
import sounddevice as sd
import soundfile as sf
import parselmouth
import numpy as np
import base64
import json

st.set_page_config(page_title="NinjaPhonetics - Spike2 Viewer", layout="wide")

st.markdown("""
<style>
    .block-container { padding-top: 1.5rem; padding-bottom: 1rem; }
</style>
""", unsafe_allow_html=True)

st.title("🥷 NinjaPhonetics: Spike2 Biofeedback Scope")

if "wav_path" not in st.session_state:
    st.session_state.wav_path = "temp_spike2.wav"
if "recorded" not in st.session_state:
    st.session_state.recorded = False

# --- 録音コントロール ---
col1, col2 = st.columns([1, 4])
with col1:
    if st.button("🎙️ 2秒間 録音開始", type="primary", use_container_width=True):
        duration = 2.0
        sr = 44100
        with st.spinner("録音中... 発音してください"):
            audio = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype="float32")
            sd.wait()
            sf.write(st.session_state.wav_path, audio, sr)
            st.session_state.recorded = True
        st.rerun()

with col2:
    st.info("💡 **操作方法**: **赤線（カーソル）を直接ドラッグ** して位置調整 | **背景をドラッグ** して区間選択 | **Spaceキー** で再生/一時停止")


def extract_scope_data(wav_path: str, max_formant_hz: float = 5000.0):
    snd = parselmouth.Sound(wav_path)
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
        x_min = max(0.0, float(min(valid_times)) - 0.1)
        x_max = min(float(snd.duration), float(max(valid_times)) + 0.1)
    else:
        x_min, x_max = 0.0, float(snd.duration)

    raw_samples = snd.values[0]
    raw_times = np.linspace(0, snd.duration, len(raw_samples))
    step = max(1, len(raw_samples) // 2000)
    wave_t = np.round(raw_times[::step], 4).tolist()
    wave_y = np.round(raw_samples[::step], 4).tolist()

    with open(wav_path, "rb") as f:
        wav_b64 = base64.b64encode(f.read()).decode("utf-8")

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
    font-family: Arial, Helvetica, sans-serif;
    background: #FFFFFF; color: #0F172A;
    user-select: none;
  }}
  .toolbar {{
    display: flex; align-items: center; gap: 14px;
    padding: 12px 18px; background: #F8FAFC;
    border: 2px solid #CBD5E1; border-radius: 8px 8px 0 0;
  }}
  button {{
    font-family: Arial, Helvetica, sans-serif;
    font-size: 18px; font-weight: bold;
    padding: 8px 18px; border-radius: 6px;
    border: 2px solid #334155; cursor: pointer;
    background: #FFFFFF; color: #0F172A;
    transition: all 0.15s;
  }}
  button.primary {{ background: #2563EB; color: white; border-color: #1D4ED8; }}
  button.primary:hover {{ background: #1D4ED8; }}
  button:hover {{ background: #E2E8F0; }}
  select {{
    font-size: 18px; padding: 6px 10px;
    border-radius: 6px; border: 2px solid #334155;
    font-family: Arial, Helvetica, sans-serif; font-weight: bold;
  }}
  .readout {{
    font-size: 20px; font-weight: bold; margin-left: auto;
    font-variant-numeric: tabular-nums; color: #1E293B;
  }}
  #scope-wrapper {{
    position: relative; width: 100%; height: 680px;
    border: 2px solid #334155; border-top: none;
    background: #FFFFFF;
  }}
  canvas {{
    position: absolute; left: 0; top: 0;
    width: 100%; height: 100%;
  }}
</style>
</head>
<body>

<div class="toolbar">
  <button id="btn-play" class="primary">▶ Play (Space)</button>
  <button id="btn-reset">⏮ 全体選択に戻す</button>
  <label style="font-size: 18px; font-weight: bold;">Speed:
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

const MARGIN = {{ left: 120, right: 40, top: 25, bottom: 60 }};
let W = 1000, H = 680;

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
  const availH = totalH - gap * 2;
  const h1 = availH * 0.22;
  const h2 = availH * 0.39;
  const h3 = availH * 0.39;
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

  bgCtx.clearRect(0, 0, W, H);
  const panels = getPanels();
  const plotW = W - MARGIN.left - MARGIN.right;

  bgCtx.font = "20px Arial, Helvetica, sans-serif";
  bgCtx.fillStyle = "#0F172A";

  const durationSpan = xMax - xMin;
  const stepT = durationSpan <= 0.6 ? 0.05 : (durationSpan <= 1.5 ? 0.1 : 0.2);
  const firstTick = Math.ceil(xMin / stepT) * stepT;

  panels.forEach((p, idx) => {{
    bgCtx.strokeStyle = "#E2E8F0";
    bgCtx.lineWidth = 1.5;
    for (let t = firstTick; t <= xMax + 1e-6; t += stepT) {{
      const x = tToX(t);
      bgCtx.beginPath();
      bgCtx.moveTo(x, p.top);
      bgCtx.lineTo(x, p.bottom);
      bgCtx.stroke();

      if (idx === 2) {{
        bgCtx.strokeStyle = "#334155";
        bgCtx.lineWidth = 2;
        bgCtx.beginPath();
        bgCtx.moveTo(x, p.bottom);
        bgCtx.lineTo(x, p.bottom + 6);
        bgCtx.stroke();

        bgCtx.textAlign = "center";
        bgCtx.textBaseline = "top";
        bgCtx.fillText(t.toFixed(2), x, p.bottom + 10);
        bgCtx.strokeStyle = "#E2E8F0";
        bgCtx.lineWidth = 1.5;
      }}
    }}
  }});

  // 1段目: 生波形
  const p1 = panels[0];
  const waveMid = (p1.top + p1.bottom) / 2;
  bgCtx.strokeStyle = "#CBD5E1";
  bgCtx.beginPath(); bgCtx.moveTo(MARGIN.left, waveMid); bgCtx.lineTo(W - MARGIN.right, waveMid); bgCtx.stroke();

  bgCtx.save();
  bgCtx.beginPath(); bgCtx.rect(MARGIN.left, p1.top, plotW, p1.height); bgCtx.clip();
  bgCtx.strokeStyle = "#475569";
  bgCtx.lineWidth = 1.5;
  bgCtx.beginPath();
  const maxAmp = Math.max(...D.wave_y.map(Math.abs), 0.05);
  for (let i = 0; i < D.wave_t.length; i++) {{
    const x = tToX(D.wave_t[i]);
    const y = valToY(D.wave_y[i], -maxAmp * 1.1, maxAmp * 1.1, p1);
    if (i === 0) bgCtx.moveTo(x, y); else bgCtx.lineTo(x, y);
  }}
  bgCtx.stroke();
  bgCtx.restore();

  // 2段目: F3 / F2
  const p2 = panels[1];
  const fMin = 500, fMax = 3200;
  [1000, 1500, 2000, 2500, 3000].forEach(hz => {{
    const y = valToY(hz, fMin, fMax, p2);
    bgCtx.strokeStyle = "#E2E8F0"; bgCtx.lineWidth = 1.5;
    bgCtx.beginPath(); bgCtx.moveTo(MARGIN.left, y); bgCtx.lineTo(W - MARGIN.right, y); bgCtx.stroke();
    bgCtx.strokeStyle = "#334155"; bgCtx.lineWidth = 2;
    bgCtx.beginPath(); bgCtx.moveTo(MARGIN.left - 6, y); bgCtx.lineTo(MARGIN.left, y); bgCtx.stroke();
    bgCtx.textAlign = "right"; bgCtx.textBaseline = "middle";
    bgCtx.fillText(hz.toString(), MARGIN.left - 10, y);
  }});

  bgCtx.save();
  bgCtx.beginPath(); bgCtx.rect(MARGIN.left, p2.top, plotW, p2.height); bgCtx.clip();
  drawSeries(D.times, D.f3, fMin, fMax, p2, "#2563EB", 3.5);
  drawSeries(D.times, D.f2, fMin, fMax, p2, "#059669", 3.5);
  bgCtx.restore();

  // 3段目: Δ(F3 - F2)
  const p3 = panels[2];
  const dMin = 0, dMax = 1800;
  [0, 500, 1000, 1500].forEach(hz => {{
    const y = valToY(hz, dMin, dMax, p3);
    bgCtx.strokeStyle = "#E2E8F0"; bgCtx.lineWidth = 1.5;
    bgCtx.beginPath(); bgCtx.moveTo(MARGIN.left, y); bgCtx.lineTo(W - MARGIN.right, y); bgCtx.stroke();
    bgCtx.strokeStyle = "#334155"; bgCtx.lineWidth = 2;
    bgCtx.beginPath(); bgCtx.moveTo(MARGIN.left - 6, y); bgCtx.lineTo(MARGIN.left, y); bgCtx.stroke();
    bgCtx.textAlign = "right"; bgCtx.textBaseline = "middle";
    bgCtx.fillText(hz.toString(), MARGIN.left - 10, y);
  }});

  const yTarget = valToY(1000, dMin, dMax, p3);
  bgCtx.save();
  bgCtx.strokeStyle = "#64748B"; bgCtx.lineWidth = 2; bgCtx.setLineDash([8, 6]);
  bgCtx.beginPath(); bgCtx.moveTo(MARGIN.left, yTarget); bgCtx.lineTo(W - MARGIN.right, yTarget); bgCtx.stroke();
  bgCtx.restore();
  bgCtx.font = "18px Arial, Helvetica, sans-serif";
  bgCtx.fillStyle = "#475569"; bgCtx.textAlign = "right"; bgCtx.textBaseline = "bottom";
  bgCtx.fillText("R Target (<1000Hz)", W - MARGIN.right - 6, yTarget - 4);

  bgCtx.save();
  bgCtx.beginPath(); bgCtx.rect(MARGIN.left, p3.top, plotW, p3.height); bgCtx.clip();
  drawSeries(D.times, D.delta, dMin, dMax, p3, "#0F172A", 4.0);
  bgCtx.restore();

  panels.forEach(p => {{
    bgCtx.strokeStyle = "#334155"; bgCtx.lineWidth = 2;
    bgCtx.beginPath();
    bgCtx.moveTo(MARGIN.left, p.top);
    bgCtx.lineTo(MARGIN.left, p.bottom);
    bgCtx.lineTo(W - MARGIN.right, p.bottom);
    bgCtx.stroke();
  }});

  bgCtx.fillStyle = "#0F172A";
  bgCtx.font = "bold 22px Arial, Helvetica, sans-serif";
  drawRotatedLabel("Waveform", 28, (p1.top + p1.bottom) / 2);
  drawRotatedLabel("F3 / F2 (Hz)", 28, (p2.top + p2.bottom) / 2);
  drawRotatedLabel("Δ(F3 - F2) (Hz)", 28, (p3.top + p3.bottom) / 2);

  bgCtx.textAlign = "center"; bgCtx.textBaseline = "bottom";
  bgCtx.fillText("Time (s)", MARGIN.left + plotW / 2, H - 6);
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

  // 範囲選択の薄黄色ハイライト
  if (Math.abs(selEnd - selStart) > 0.005) {{
    const sx = tToX(Math.min(selStart, selEnd));
    const ex = tToX(Math.max(selStart, selEnd));
    ovCtx.fillStyle = "rgba(245, 158, 11, 0.18)";
    ovCtx.fillRect(sx, topY, ex - sx, botY - topY);

    ovCtx.strokeStyle = "#D97706";
    ovCtx.lineWidth = 2;
    ovCtx.beginPath();
    ovCtx.moveTo(sx, topY); ovCtx.lineTo(sx, botY);
    ovCtx.moveTo(ex, topY); ovCtx.lineTo(ex, botY);
    ovCtx.stroke();
  }}

  // Spike2 赤色同期縦カーソル
  const cx = tToX(cursorT);
  ovCtx.strokeStyle = "#EF4444";
  ovCtx.lineWidth = 2.5;
  ovCtx.beginPath();
  ovCtx.moveTo(cx, topY);
  ovCtx.lineTo(cx, botY);
  ovCtx.stroke();

  // カーソルつまみ（上部と下部に小さな赤い三角マークを描画して掴めることを明示）
  ovCtx.fillStyle = "#EF4444";
  ovCtx.beginPath();
  ovCtx.moveTo(cx - 7, topY); ovCtx.lineTo(cx + 7, topY); ovCtx.lineTo(cx, topY + 10); ovCtx.closePath();
  ovCtx.fill();
  ovCtx.beginPath();
  ovCtx.moveTo(cx - 7, botY); ovCtx.lineTo(cx + 7, botY); ovCtx.lineTo(cx, botY - 10); ovCtx.closePath();
  ovCtx.fill();

  // デジタル数値の更新
  let idx = D.times.findIndex(t => t >= cursorT);
  let info = `Cursor: ${{cursorT.toFixed(3)}} s`;
  if (idx !== -1 && D.delta[idx] !== null) {{
    info += `  |  F3: ${{D.f3[idx]}} Hz   F2: ${{D.f2[idx]}} Hz   Δ: ${{D.delta[idx]}} Hz`;
  }}
  readout.textContent = info;
}}

// --- 再生エンジン ---
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

// --- キーボードショートカット (Spaceキー) ---
window.addEventListener("keydown", (e) => {{
  if (e.code === "Space" && e.target === document.body) {{
    e.preventDefault();
    if (isPlaying) stopPlayback();
    else startPlayback();
  }}
}});

// --- インタラクティブ・マウス操作（カーソル直接ドラッグ ＆ 範囲ドラッグ） ---
let dragMode = "none"; // "cursor" (カーソル掴み) | "select" (範囲選択)

// カーソル直上（左右10px以内）にいるか判定
function isNearCursor(mouseX) {{
  const cx = tToX(cursorT);
  return Math.abs(mouseX - cx) <= 10;
}}

ovCanvas.addEventListener("mousemove", (e) => {{
  const rect = ovCanvas.getBoundingClientRect();
  const mouseX = e.clientX - rect.left;

  if (dragMode === "none") {{
    // ホバー時にカーソル直上なら左右矢印アイコン(↔)にする
    if (isNearCursor(mouseX)) {{
      ovCanvas.style.cursor = "ew-resize";
    }} else {{
      ovCanvas.style.cursor = "crosshair";
    }}
  }} else if (dragMode === "cursor") {{
    // 赤線カーソルそのものを直接ドラッグ移動中
    cursorT = xToT(mouseX);
    drawOverlay();
  }} else if (dragMode === "select") {{
    // 範囲ドラッグ中
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
    // 赤線を掴んだ場合
    dragMode = "cursor";
    ovCanvas.style.cursor = "ew-resize";
  }} else {{
    // 背景をクリック・ドラッグした場合
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
    // 単なるクリック（ドラッグ距離が極小）なら範囲選択を全体に戻し、クリック位置にカーソルを移動
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

if st.session_state.recorded:
    scope_data = extract_scope_data(st.session_state.wav_path)
    html_code = build_spike2_html(scope_data)
    components.html(html_code, height=760, scrolling=False)