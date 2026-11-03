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
        padding-top: 1.8rem !important;
        padding-bottom: 0.5rem !important;
        padding-left: 1.5rem !important;
        padding-right: 1.5rem !important;
    }
    h1 {
        font-size: 1.6rem !important;
        font-weight: 700 !important;
        margin-bottom: 0.5rem !important;
        color: #F8FAFC !important;
        line-height: 1.3 !important;
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
</style>
""", unsafe_allow_html=True)

st.title("NinjaPhonetics: Biofeedback Scope")

if "current_audio_bytes" not in st.session_state:
    st.session_state.current_audio_bytes = None
if "current_label" not in st.session_state:
    st.session_state.current_label = ""
if "last_recorded_bytes" not in st.session_state:
    st.session_state.last_recorded_bytes = None
if "auto_play" not in st.session_state:
    st.session_state.auto_play = False

# --- お手本音声プリセット設定 ---
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

# マイク入力
audio_input = st.audio_input("または自分の声を録音して解析")
if not preset_clicked and audio_input is not None:
    new_bytes = audio_input.read()
    if new_bytes != st.session_state.last_recorded_bytes:
        st.session_state.last_recorded_bytes = new_bytes
        st.session_state.current_audio_bytes = new_bytes
        st.session_state.current_label = "あなたの発音録音"
        st.session_state.auto_play = False


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

    # 末尾の -ly 等が切れないよう、後の余白を十分に確保（+0.25s）
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
    user-select: none;
    overflow: hidden;
  }}
  .toolbar {{
    display: flex; align-items: center; gap: 10px;
    padding: 10px 14px; background: #0F172A;
    border: 1px solid #1E293B; border-radius: 8px 8px 0 0;
  }}
  button {{
    font-family: inherit;
    font-size: 14px; font-weight: bold;
    padding: 8px 14px; border-radius: 6px;
    border: 1px solid #334155; cursor: pointer;
    background: #1E293B; color: #E2E8F0;
    transition: all 0.15s;
    white-space: nowrap;
  }}
  button:hover {{ background: #334155; color: #FFFFFF; }}
  
  button.btn-play-ready {{
    background: #16A34A; color: #FFFFFF; border-color: #15803D;
  }}
  button.btn-play-ready:hover {{
    background: #15803D;
  }}
  button.btn-play-active {{
    background: #DC2626; color: #FFFFFF; border-color: #B91C1C;
  }}
  
  button.toggle-active {{
    background: #334155; color: #F8FAFC; border-color: #64748B;
  }}

  select {{
    font-family: inherit;
    font-size: 14px; padding: 6px 10px;
    border-radius: 6px; border: 1px solid #334155;
    background: #1E293B; color: #E2E8F0;
    font-weight: bold;
  }}
  .readout {{
    font-family: inherit;
    font-size: 14px; font-weight: 500; margin-left: auto;
    font-variant-numeric: tabular-nums; color: #E2E8F0;
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
  <button id="btn-dl-plot" title="プロット画像をPNGで保存">💾 Plot (.png)</button>
  <button id="btn-dl-audio" title="音声ファイルをWAVで保存">🔊 Audio (.wav)</button>
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

let xMin = D.x_min;
let xMax = D.x_max;
let selStart = xMin;
let selEnd = xMax;
let cursorT = (xMin + xMax) / 2;
let showSpec = true;

let specCanvas = null;
if (D.spec_w > 0 && D.spec_h > 0 && D.spec_b64) {{
  specCanvas = document.createElement('canvas');
  specCanvas.width = D.spec_w;
  specCanvas.height = D.spec_h;
  const sCtx = specCanvas.getContext('2d');
  const imgData = sCtx.createImageData(D.spec_w, D.spec_h);
  const raw = atob(D.spec_b64);
  for (let i = 0; i < raw.length; i++) {{
    const val = raw.charCodeAt(i);
    const p = i * 4;
    const gray = Math.min(255, 12 + val * 0.85);
    imgData.data[p]     = gray;
    imgData.data[p + 1] = gray;
    imgData.data[p + 2] = gray;
    imgData.data[p + 3] = 255;
  }}
  sCtx.putImageData(imgData, 0, 0);
}}

// --- ピッチ維持再生のための HTMLAudioElement 構築 ---
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
  // 音程（ピッチ）を変えずに再生速度だけを変える設定
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
  const gap = 24;
  const h1 = 120;
  const remainH = totalH - h1 - (gap * 2);
  const h2 = remainH * 0.50;
  const h3 = remainH * 0.50;

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
  const p1 = panels[0];
  const p2 = panels[1];
  const p3 = panels[2];

  if (showSpec && specCanvas) {{
    bgCtx.save();
    bgCtx.beginPath();
    bgCtx.rect(MARGIN.left, p2.top, plotW, p2.height);
    bgCtx.clip();
    bgCtx.imageSmoothingEnabled = true;
    bgCtx.imageSmoothingQuality = "high";
    bgCtx.drawImage(specCanvas, MARGIN.left, p2.top, plotW, p2.height);
    bgCtx.restore();
  }}

  const APP_FONT = '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif';
  bgCtx.font = "13px " + APP_FONT;
  bgCtx.fillStyle = "#94A3B8";

  const durationSpan = xMax - xMin;
  const stepT = durationSpan <= 0.6 ? 0.05 : (durationSpan <= 1.5 ? 0.1 : 0.2);
  const firstTick = Math.ceil(xMin / stepT) * stepT;

  panels.forEach((p) => {{
    for (let t = firstTick; t <= xMax + 1e-6; t += stepT) {{
      const x = tToX(t);

      bgCtx.strokeStyle = "rgba(148, 163, 184, 0.14)";
      bgCtx.lineWidth = 1.0;
      bgCtx.beginPath();
      bgCtx.moveTo(x, p.top);
      bgCtx.lineTo(x, p.bottom);
      bgCtx.stroke();

      bgCtx.strokeStyle = "#475569";
      bgCtx.lineWidth = 1.5;
      bgCtx.beginPath();
      bgCtx.moveTo(x, p.bottom);
      bgCtx.lineTo(x, p.bottom + 5);
      bgCtx.stroke();

      bgCtx.fillStyle = "#94A3B8";
      bgCtx.textAlign = "center";
      bgCtx.textBaseline = "top";
      bgCtx.fillText(t.toFixed(2), x, p.bottom + 6);
    }}
  }});

  // --- 第1段: 生波形 ---
  const waveMid = (p1.top + p1.bottom) / 2;
  bgCtx.strokeStyle = "#1E293B";
  bgCtx.beginPath(); bgCtx.moveTo(MARGIN.left, waveMid); bgCtx.lineTo(W - MARGIN.right, waveMid); bgCtx.stroke();

  const maxAmp = Math.max(...D.wave_y.map(Math.abs), 0.05);
  [-1.0, 0.0, 1.0].forEach(relVal => {{
    const absVal = relVal * maxAmp;
    const y = valToY(absVal, -maxAmp * 1.1, maxAmp * 1.1, p1);
    bgCtx.strokeStyle = "#334155"; bgCtx.lineWidth = 1.5;
    bgCtx.beginPath(); bgCtx.moveTo(MARGIN.left - 5, y); bgCtx.lineTo(MARGIN.left, y); bgCtx.stroke();
    bgCtx.fillStyle = "#94A3B8";
    bgCtx.textAlign = "right"; bgCtx.textBaseline = "middle";
    bgCtx.fillText(relVal.toFixed(1), MARGIN.left - 8, y);
  }});

  bgCtx.save();
  bgCtx.beginPath(); bgCtx.rect(MARGIN.left, p1.top, plotW, p1.height); bgCtx.clip();
  bgCtx.strokeStyle = "#94A3B8";
  bgCtx.lineWidth = 1.5;
  bgCtx.beginPath();
  for (let i = 0; i < D.wave_t.length; i++) {{
    const x = tToX(D.wave_t[i]);
    const y = valToY(D.wave_y[i], -maxAmp * 1.1, maxAmp * 1.1, p1);
    if (i === 0) bgCtx.moveTo(x, y); else bgCtx.lineTo(x, y);
  }}
  bgCtx.stroke();
  bgCtx.restore();

  // --- 第2段: F2 and F3 (500 ~ 4000Hz) ---
  const fMin = 500, fMax = 4000;
  [1000, 1500, 2000, 2500, 3000, 3500, 4000].forEach(hz => {{
    const y = valToY(hz, fMin, fMax, p2);
    bgCtx.strokeStyle = "rgba(148, 163, 184, 0.14)"; bgCtx.lineWidth = 1.0;
    bgCtx.beginPath(); bgCtx.moveTo(MARGIN.left, y); bgCtx.lineTo(W - MARGIN.right, y); bgCtx.stroke();
    bgCtx.strokeStyle = "#475569"; bgCtx.lineWidth = 1.5;
    bgCtx.beginPath(); bgCtx.moveTo(MARGIN.left - 5, y); bgCtx.lineTo(MARGIN.left, y); bgCtx.stroke();
    bgCtx.fillStyle = "#94A3B8";
    bgCtx.textAlign = "right"; bgCtx.textBaseline = "middle";
    bgCtx.fillText(hz.toString(), MARGIN.left - 8, y);
  }});

  bgCtx.save();
  bgCtx.beginPath(); bgCtx.rect(MARGIN.left, p2.top, plotW, p2.height); bgCtx.clip();
  drawSeries(D.times, D.f3, fMin, fMax, p2, "#38BDF8", 3.0);
  drawSeries(D.times, D.f2, fMin, fMax, p2, "#10B981", 3.0);
  bgCtx.restore();

  // 中段 凡例
  const legendX = W - MARGIN.right - 82;
  const legendY1 = p2.top + 18;
  const legendY2 = p2.top + 38;
  bgCtx.fillStyle = "rgba(9, 13, 22, 0.85)";
  bgCtx.fillRect(legendX - 8, p2.top + 6, 76, 44);

  bgCtx.font = "bold 14px " + APP_FONT;
  bgCtx.strokeStyle = "#38BDF8"; bgCtx.lineWidth = 3.0;
  bgCtx.beginPath(); bgCtx.moveTo(legendX, legendY1); bgCtx.lineTo(legendX + 18, legendY1); bgCtx.stroke();
  bgCtx.fillStyle = "#38BDF8"; bgCtx.textAlign = "left"; bgCtx.textBaseline = "middle";
  bgCtx.fillText("F3", legendX + 24, legendY1);

  bgCtx.strokeStyle = "#10B981"; bgCtx.lineWidth = 3.0;
  bgCtx.beginPath(); bgCtx.moveTo(legendX, legendY2); bgCtx.lineTo(legendX + 18, legendY2); bgCtx.stroke();
  bgCtx.fillStyle = "#10B981";
  bgCtx.fillText("F2", legendX + 24, legendY2);

  // --- 第3段: Δ(F3 - F2) ---
  const dMin = 0, dMax = 2500;
  [0, 500, 1000, 1500, 2000, 2500].forEach(hz => {{
    const y = valToY(hz, dMin, dMax, p3);
    bgCtx.strokeStyle = "rgba(148, 163, 184, 0.14)"; bgCtx.lineWidth = 1.0;
    bgCtx.beginPath(); bgCtx.moveTo(MARGIN.left, y); bgCtx.lineTo(W - MARGIN.right, y); bgCtx.stroke();
    bgCtx.strokeStyle = "#475569"; bgCtx.lineWidth = 1.5;
    bgCtx.beginPath(); bgCtx.moveTo(MARGIN.left - 5, y); bgCtx.lineTo(MARGIN.left, y); bgCtx.stroke();
    bgCtx.fillStyle = "#94A3B8";
    bgCtx.textAlign = "right"; bgCtx.textBaseline = "middle";
    bgCtx.fillText(hz.toString(), MARGIN.left - 8, y);
  }});

  const yTarget = valToY(1000, dMin, dMax, p3);
  bgCtx.save();
  bgCtx.strokeStyle = "#EF4444"; bgCtx.lineWidth = 1.5; bgCtx.setLineDash([6, 5]);
  bgCtx.beginPath(); bgCtx.moveTo(MARGIN.left, yTarget); bgCtx.lineTo(W - MARGIN.right, yTarget); bgCtx.stroke();
  bgCtx.restore();
  bgCtx.font = "13px " + APP_FONT;
  bgCtx.fillStyle = "#EF4444"; bgCtx.textAlign = "right"; bgCtx.textBaseline = "bottom";
  bgCtx.fillText("R Target (<1000Hz)", W - MARGIN.right - 6, yTarget - 3);

  bgCtx.save();
  bgCtx.beginPath(); bgCtx.rect(MARGIN.left, p3.top, plotW, p3.height); bgCtx.clip();
  drawSeries(D.times, D.delta, dMin, dMax, p3, "#F59E0B", 3.2);
  bgCtx.restore();

  // 下段 凡例
  const leg3X = W - MARGIN.right - 145;
  const leg3Y = p3.top + 18;
  bgCtx.fillStyle = "rgba(9, 13, 22, 0.85)";
  bgCtx.fillRect(leg3X - 8, p3.top + 6, 140, 26);

  bgCtx.font = "bold 13px " + APP_FONT;
  bgCtx.strokeStyle = "#F59E0B"; bgCtx.lineWidth = 3.0;
  bgCtx.beginPath(); bgCtx.moveTo(leg3X, leg3Y); bgCtx.lineTo(leg3X + 18, leg3Y); bgCtx.stroke();
  bgCtx.fillStyle = "#F59E0B"; bgCtx.textAlign = "left"; bgCtx.textBaseline = "middle";
  bgCtx.fillText("Δ(F3 - F2)", leg3X + 24, leg3Y);

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
  bgCtx.font = "bold 15px " + APP_FONT;
  bgCtx.fillStyle = "#CBD5E1";
  drawRotatedLabel("Waveform", 28, (p1.top + p1.bottom) / 2);
  drawRotatedLabel("F2 and F3 (Hz)", 28, (p2.top + p2.bottom) / 2);
  drawRotatedLabel("Δ(F3 - F2)", 28, (p3.top + p3.bottom) / 2);

  // X軸タイトル
  bgCtx.fillStyle = "#94A3B8";
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

  if (Math.abs(selEnd - selStart) > 0.005) {{
    const sx = tToX(Math.min(selStart, selEnd));
    const ex = tToX(Math.max(selStart, selEnd));

    ovCtx.fillStyle = "rgba(255, 255, 255, 0.08)";
    ovCtx.fillRect(sx, topY, ex - sx, botY - topY);

    ovCtx.strokeStyle = "rgba(255, 255, 255, 0.4)";
    ovCtx.lineWidth = 1.0;
    ovCtx.beginPath();
    ovCtx.moveTo(sx, topY); ovCtx.lineTo(sx, botY);
    ovCtx.moveTo(ex, topY); ovCtx.lineTo(ex, botY);
    ovCtx.stroke();
  }}

  const cx = tToX(cursorT);
  ovCtx.strokeStyle = "#EF4444";
  ovCtx.lineWidth = 2.0;
  ovCtx.beginPath();
  ovCtx.moveTo(cx, topY);
  ovCtx.lineTo(cx, botY);
  ovCtx.stroke();

  ovCtx.fillStyle = "#EF4444";
  ovCtx.beginPath();
  ovCtx.moveTo(cx - 6, topY); ovCtx.lineTo(cx + 6, topY); ovCtx.lineTo(cx, topY + 8); ovCtx.closePath();
  ovCtx.fill();
  ovCtx.beginPath();
  ovCtx.moveTo(cx - 6, botY); ovCtx.lineTo(cx + 6, botY); ovCtx.lineTo(cx, botY - 8); ovCtx.closePath();
  ovCtx.fill();

  let idx = D.times.findIndex(t => t >= cursorT);
  let html = `<span style="color:#94A3B8;">Cursor:</span> ${{cursorT.toFixed(3)}} s`;
  if (idx !== -1 && D.delta[idx] !== null) {{
    html += `  |  <span style="color:#38BDF8;">F3:</span> ${{D.f3[idx]}} Hz   <span style="color:#10B981;">F2:</span> ${{D.f2[idx]}} Hz   <span style="color:#F59E0B;">Δ:</span> ${{D.delta[idx]}} Hz`;
  }}
  readout.innerHTML = html;
}}

btnSpec.addEventListener("click", () => {{
  showSpec = !showSpec;
  if (showSpec) {{
    btnSpec.textContent = "Spectrogram: ON";
    btnSpec.classList.add("toggle-active");
  }} else {{
    btnSpec.textContent = "Spectrogram: OFF";
    btnSpec.classList.remove("toggle-active");
  }}
  drawBackground();
}});

// --- ピッチ維持対応の再生処理 ---
function startPlayback() {{
  if (!audioEl.src) return;
  stopPlayback();

  const rate = parseFloat(selRate.value);
  audioEl.playbackRate = rate;
  audioEl.preservesPitch = true;
  audioEl.mozPreservesPitch = true;
  audioEl.webkitPreservesPitch = true;

  const s = Math.min(selStart, selEnd);
  const e = Math.max(selStart, selEnd);
  const playFrom = (cursorT >= s && cursorT < e - 0.02) ? cursorT : s;

  audioEl.currentTime = playFrom;
  isPlaying = true;
  btnPlay.textContent = "⏹ Stop (Space)";
  btnPlay.className = "btn-play-active";

  audioEl.play().catch(err => {{
    console.warn("Autoplay blocked or interrupted:", err);
    stopPlayback();
  }});

  function tick() {{
    if (!isPlaying) return;
    cursorT = audioEl.currentTime;
    if (cursorT >= e || audioEl.ended) {{
      audioEl.pause();
      isPlaying = false;
      cursorT = e;
      btnPlay.textContent = "▶ Play (Space)";
      btnPlay.className = "btn-play-ready";
      drawOverlay();
      return;
    }}
    drawOverlay();
    animId = requestAnimationFrame(tick);
  }}
  animId = requestAnimationFrame(tick);
}}

function stopPlayback() {{
  audioEl.pause();
  if (animId) cancelAnimationFrame(animId);
  isPlaying = false;
  btnPlay.textContent = "▶ Play (Space)";
  btnPlay.className = "btn-play-ready";
}}

// 再生中に速度変更ドロップダウンを切り替えた場合も即座に反映
selRate.addEventListener("change", () => {{
  const rate = parseFloat(selRate.value);
  audioEl.playbackRate = rate;
  audioEl.preservesPitch = true;
  audioEl.mozPreservesPitch = true;
  audioEl.webkitPreservesPitch = true;
}});

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

// --- プロット画像 (.png) のダウンロード ---
btnDlPlot.addEventListener("click", () => {{
  const expCanvas = document.createElement("canvas");
  expCanvas.width = bgCanvas.width;
  expCanvas.height = bgCanvas.height;
  const eCtx = expCanvas.getContext("2d");

  eCtx.drawImage(bgCanvas, 0, 0);
  eCtx.drawImage(ovCanvas, 0, 0);

  const link = document.createElement("a");
  link.download = `NinjaPhonetics_${{D.export_name}}.png`;
  link.href = expCanvas.toDataURL("image/png");
  link.click();
}});

// --- 音声ファイル (.wav) のダウンロード ---
btnDlAudio.addEventListener("click", () => {{
  if (!audioBlobUrl) return;
  const link = document.createElement("a");
  link.download = `NinjaPhonetics_${{D.export_name}}.wav`;
  link.href = audioBlobUrl;
  link.click();
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

// お手本ボタン押下時: 1秒待ってから先頭から自動再生
if (D.auto_play) {{
  setTimeout(() => {{
    cursorT = xMin;
    startPlayback();
  }}, 1000);
}}
</script>
</body>
</html>
"""

# スコープ描画と解説カード
if st.session_state.current_audio_bytes is not None:
    if st.session_state.current_label:
        # フォント約2倍・余白を大幅に拡大して視認性を強化
        st.markdown(
            f"<div style='color: #94A3B8; font-size: 1.85rem; font-weight: 700; margin-top: 18px; margin-bottom: 16px; line-height: 1.25;'>"
            f"● 表示中: <span style='color: #F8FAFC;'>{st.session_state.current_label}</span>"
            f"</div>",
            unsafe_allow_html=True
        )
    
    # ダウンロードファイル名用のクリーンな文字列生成
    clean_export_name = re.sub(r'[^a-zA-Z0-9_\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF]', '_', st.session_state.current_label)
    clean_export_name = re.sub(r'_+', '_', clean_export_name).strip('_')

    scope_data = extract_scope_data_from_bytes(st.session_state.current_audio_bytes)
    html_code = build_spike2_html(
        scope_data,
        auto_play=st.session_state.auto_play,
        export_name=clean_export_name or "sample"
    )
    # 一度自動再生トリガーを渡したらリセット
    st.session_state.auto_play = False

    components.html(html_code, height=640, scrolling=False)

    # 下部解説枠
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
    elif "あなたの発音" in st.session_state.current_label:
        st.markdown("""
        <div style="background: #0F172A; border: 1px solid #1E293B; border-left: 3px solid #64748B; border-radius: 6px; padding: 12px 18px; margin-top: 10px;">
            <span style="font-size: 1.05rem; font-weight: 600; color: #F8FAFC;">判定ガイド</span>
            <p style="font-size: 0.9rem; color: #94A3B8; margin: 4px 0 0 0; line-height: 1.5;">
                ・<b>/r/ の判定</b>: 語頭で黄色い線（Δ）が破線（1000Hz）を下回る深い谷を作っていれば成功です。<br>
                ・<b>/l/ の判定</b>: 青い線（F3）が高い位置（2500Hz以上）にとどまり、黄色い線が破線を割らなければクリアです。
            </p>
        </div>
        """, unsafe_allow_html=True)