"""
views_a1.py: A1 音素ホルマント・スコープ (R/L) Spike2風描画生成
"""
import json

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
    padding: 8px 14px; background: #0F172A;
    border: 1px solid #1E293B; border-radius: 8px 8px 0 0;
  }}
  button {{
    font-family: inherit; font-size: 13px; font-weight: bold;
    padding: 6px 12px; border-radius: 6px;
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
    font-family: inherit; font-size: 13px; padding: 5px 8px;
    border-radius: 6px; border: 1px solid #334155;
    background: #1E293B; color: #E2E8F0; font-weight: bold;
  }}
  .readout {{
    font-family: inherit; font-size: 13px; font-weight: 500; margin-left: auto;
    font-variant-numeric: tabular-nums; color: #E2E8F0; white-space: nowrap;
  }}
  #scope-wrapper {{
    position: relative; width: 100%; height: 530px;
    border: 1px solid #1E293B; border-top: none; background: #090D16;
    border-radius: 0 0 8px 8px;
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
  <label style="font-size: 13px; font-weight: 500; display: flex; align-items: center; gap: 6px; color: #94A3B8;">
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

const MARGIN = {{ left: 100, right: 30, top: 18, bottom: 40 }};
let W = 1000, H = 530;
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
  const gap = 20, h1 = 105;
  const remainH = totalH - h1 - (gap * 2);
  const h2 = remainH * 0.52, h3 = remainH * 0.48;
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
  bgCtx.font = "12px " + APP_FONT;
  const durationSpan = xMax - xMin;
  const stepT = durationSpan <= 0.6 ? 0.05 : (durationSpan <= 1.5 ? 0.1 : 0.2);
  const firstTick = Math.ceil(xMin / stepT) * stepT;

  panels.forEach((p) => {{
    for (let t = firstTick; t <= xMax + 1e-6; t += stepT) {{
      const x = tToX(t);
      bgCtx.strokeStyle = "rgba(148, 163, 184, 0.12)"; bgCtx.lineWidth = 1.0;
      bgCtx.beginPath(); bgCtx.moveTo(x, p.top); bgCtx.lineTo(x, p.bottom); bgCtx.stroke();
      bgCtx.strokeStyle = "#475569"; bgCtx.lineWidth = 1.5;
      bgCtx.beginPath(); bgCtx.moveTo(x, p.bottom); bgCtx.lineTo(x, p.bottom + 4); bgCtx.stroke();
      bgCtx.fillStyle = "#94A3B8"; bgCtx.textAlign = "center"; bgCtx.textBaseline = "top";
      bgCtx.fillText(t.toFixed(2), x, p.bottom + 5);
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
    bgCtx.strokeStyle = "rgba(148, 163, 184, 0.12)"; bgCtx.lineWidth = 1.0;
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

  // Ch2 凡例 (F3/F2)
  const legendX = W - MARGIN.right - 85;
  bgCtx.fillStyle = "rgba(9, 13, 22, 0.85)";
  bgCtx.fillRect(legendX - 8, p2.top + 6, 80, 44);
  bgCtx.strokeStyle = "#1E293B"; bgCtx.lineWidth = 1.0;
  bgCtx.strokeRect(legendX - 8, p2.top + 6, 80, 44);

  bgCtx.font = "bold 13px " + APP_FONT;
  bgCtx.strokeStyle = "#38BDF8"; bgCtx.lineWidth = 3.0;
  bgCtx.beginPath(); bgCtx.moveTo(legendX, p2.top + 18); bgCtx.lineTo(legendX + 18, p2.top + 18); bgCtx.stroke();
  bgCtx.fillStyle = "#38BDF8"; bgCtx.textAlign = "left"; bgCtx.textBaseline = "middle";
  bgCtx.fillText("F3", legendX + 24, p2.top + 18);
  bgCtx.strokeStyle = "#10B981"; bgCtx.lineWidth = 3.0;
  bgCtx.beginPath(); bgCtx.moveTo(legendX, p2.top + 36); bgCtx.lineTo(legendX + 18, p2.top + 36); bgCtx.stroke();
  bgCtx.fillStyle = "#10B981"; bgCtx.fillText("F2", legendX + 24, p2.top + 36);

  const dMin = 0, dMax = 2500;
  [0, 500, 1000, 1500, 2000, 2500].forEach(hz => {{
    const y = valToY(hz, dMin, dMax, p3);
    bgCtx.strokeStyle = "rgba(148, 163, 184, 0.12)"; bgCtx.lineWidth = 1.0;
    bgCtx.beginPath(); bgCtx.moveTo(MARGIN.left, y); bgCtx.lineTo(W - MARGIN.right, y); bgCtx.stroke();
    bgCtx.strokeStyle = "#475569"; bgCtx.lineWidth = 1.5;
    bgCtx.beginPath(); bgCtx.moveTo(MARGIN.left - 5, y); bgCtx.lineTo(MARGIN.left, y); bgCtx.stroke();
    bgCtx.fillStyle = "#94A3B8"; bgCtx.textAlign = "right"; bgCtx.textBaseline = "middle";
    bgCtx.fillText(hz.toString(), MARGIN.left - 8, y);
  }});

  // Ch3 R STRIKE ZONE (<1000Hz)
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

  // Ch3 凡例
  const leg3X = W - MARGIN.right - 140;
  bgCtx.fillStyle = "rgba(9, 13, 22, 0.85)"; bgCtx.fillRect(leg3X - 8, p3.top + 6, 136, 26);
  bgCtx.strokeStyle = "#1E293B"; bgCtx.lineWidth = 1.0;
  bgCtx.strokeRect(leg3X - 8, p3.top + 6, 136, 26);
  bgCtx.font = "bold 13px " + APP_FONT;
  bgCtx.strokeStyle = "#F59E0B"; bgCtx.lineWidth = 3.0;
  bgCtx.beginPath(); bgCtx.moveTo(leg3X, p3.top + 18); bgCtx.lineTo(leg3X + 18, p3.top + 18); bgCtx.stroke();
  bgCtx.fillStyle = "#F59E0B"; bgCtx.textAlign = "left"; bgCtx.textBaseline = "middle";
  bgCtx.fillText("Δ(F3 - F2)", leg3X + 24, p3.top + 18);

  panels.forEach(p => {{
    bgCtx.strokeStyle = "#334155"; bgCtx.lineWidth = 1.5;
    bgCtx.beginPath(); bgCtx.moveTo(MARGIN.left, p.top); bgCtx.lineTo(MARGIN.left, p.bottom); bgCtx.lineTo(W - MARGIN.right, p.bottom); bgCtx.stroke();
  }});

  bgCtx.font = "bold 14px " + APP_FONT; bgCtx.fillStyle = "#CBD5E1";
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