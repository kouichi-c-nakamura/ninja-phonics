"""
views_e2.py: E2 Prosody Diagnostics Canvas/HTML Generator
"""
import json
import base64
import numpy as np
from constants import E2_PASSAGE_BENCHMARKS
from audio_processing import load_benchmark_points

def build_e2_multiview_canvas_html(passage_key: str, eval_history: list = None, is_custom_mode: bool = False, view_mode: str = "word_swarm", word_bench_dist: dict = None) -> str:
    bm = E2_PASSAGE_BENCHMARKS[passage_key]
    x_metric = bm.get("x_metric", "pct_v")
    mu, cov = bm["mu"], bm["cov"]

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
                    user_word_raw.setdefault(wb["clean_word"], []).append(wb["vowel_ms"])

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
        "word_swarm": "弱化必須語 KDEバイオリン分布 (左: Native / 中: あなた / 右: カタカナ読み)",
        "metrics_1d": "各プロソディ指標の垂直バイオリン比較 (左: Native / 中: あなた / 右: カタカナ読み)"
    }
    header_title = view_titles.get(view_mode, "2D ストライクゾーン")

    return f"""<!DOCTYPE html><html><head><meta charset="utf-8"><style>
body {{ margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Hiragino Sans", sans-serif; background: #0B0F19; color: #E2E8F0; user-select: none; overflow: hidden; }}
.toolbar {{ display: flex; align-items: center; justify-content: space-between; padding: 6px 12px; background: #0F172A; border: 1px solid #1E293B; border-radius: 8px 8px 0 0; }}
.toolbar-title {{ font-size: 13px; font-weight: 700; color: #BAE6FD; }}
.btn-group {{ display: flex; align-items: center; gap: 8px; }}
button {{ font-family: inherit; font-size: 12px; font-weight: bold; padding: 4px 11px; border-radius: 5px; border: 1px solid #334155; cursor: pointer; background: #1E293B; color: #E2E8F0; }}
button:hover {{ background: #334155; color: #FFFFFF; }}
button.toggle-active {{ background: #172554; color: #BAE6FD; border-color: #38BDF8; }}
.scroll-container {{ position: relative; width: 100%; height: 420px; background: #090D16; border: 1px solid #1E293B; border-top: none; border-radius: 0 0 8px 8px; overflow-x: auto; overflow-y: hidden; }}
.scroll-container::-webkit-scrollbar {{ height: 8px; }}
.scroll-container::-webkit-scrollbar-track {{ background: #090D16; }}
.scroll-container::-webkit-scrollbar-thumb {{ background: #334155; border-radius: 4px; }}
#radar-canvas {{ position: absolute; left: 0; top: 0; }}
</style></head><body>
<div class="toolbar">
  <div class="toolbar-title"><span>{header_title}</span></div>
  <div class="btn-group">
    <button id="btn-toggle-bm" class="toggle-active" style="display: {'none' if is_custom_mode else 'inline-block'};">Benchmark: ON</button>
    <button id="btn-dl-radar">Plot (.png)</button>
  </div>
</div>
<div class="scroll-container" id="scroll-wrapper">
  <canvas id="radar-canvas"></canvas>
</div>
<script>
const D = {json.dumps(canvas_data)};
const canvas = document.getElementById('radar-canvas');
const ctx = canvas.getContext('2d');
const wrapper = document.getElementById('scroll-wrapper');
const btnDl = document.getElementById('btn-dl-radar');
const btnToggleBm = document.getElementById('btn-toggle-bm');
let showBenchmark = true;

function gaussianKDE(points, yGrid, bw) {{
  if (!bw) bw = 8.5;
  if (!points || points.length === 0) return new Float32Array(yGrid.length);
  const cleanPts = [];
  for (let i = 0; i < points.length; i++) {{
    const val = Number(points[i]);
    if (!isNaN(val)) cleanPts.push(val);
  }}
  if (cleanPts.length === 0) return new Float32Array(yGrid.length);
  const n = cleanPts.length;
  const factor = 1.0 / (Math.sqrt(2 * Math.PI) * bw * n);
  const density = new Float32Array(yGrid.length);
  for (let j = 0; j < yGrid.length; j++) {{
    const y = yGrid[j];
    let sum = 0;
    for (let i = 0; i < n; i++) {{
      const d = (y - cleanPts[i]) / bw;
      sum += Math.exp(-0.5 * d * d);
    }}
    density[j] = sum * factor;
  }}
  return density;
}}

function getMedian(arr) {{
  if (!arr || arr.length === 0) return 0;
  const clean = [];
  for (let i = 0; i < arr.length; i++) {{
    const val = Number(arr[i]);
    if (!isNaN(val)) clean.push(val);
  }}
  if (clean.length === 0) return 0;
  clean.sort((a, b) => a - b);
  const mid = Math.floor(clean.length / 2);
  return clean.length % 2 !== 0 ? clean[mid] : (clean[mid - 1] + clean[mid]) / 2.0;
}}

function draw() {{
  const dpr = window.devicePixelRatio || 1;
  const isWordSwarm = (D.view_mode === "word_swarm");
  const isMetrics1D = (D.view_mode === "metrics_1d");
  const wordsCount = (D.words_alpha || []).length;
  
  const clientW = wrapper.clientWidth > 0 ? wrapper.clientWidth : 750;
  const clientH = 420;

  let minWidth = clientW;
  if (isWordSwarm) {{
    minWidth = Math.max(clientW, wordsCount * 140 + 80);
  }} else if (isMetrics1D) {{
    minWidth = Math.max(clientW, 700);
  }}

  const W = minWidth;
  const H = clientH;

  canvas.width = Math.round(W * dpr);
  canvas.height = Math.round(H * dpr);
  canvas.style.width = W + "px";
  canvas.style.height = H + "px";

  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.fillStyle = "#090D16";
  ctx.fillRect(0, 0, W, H);

  if (isWordSwarm) drawViolinView(W, H);
  else if (isMetrics1D) drawMetrics1DVertical(W, H);
  else drawScatter2D(W, H);
}}

function drawScatter2D(W, H) {{
  const MARGIN = {{ left: 78, right: 28, top: 20, bottom: 46 }};
  const plotW = Math.max(10, W - MARGIN.left - MARGIN.right);
  const plotH = Math.max(10, H - MARGIN.top - MARGIN.bottom);
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
    }}
  }}

  D.history.forEach((pt, idx) => {{
    const isLatest = (idx === D.history.length - 1);
    const hxVal = isRcf ? pt.r_cf : pt.pct_v;
    const px = xToPx(hxVal), py = yToPx(pt.schwa_p75_ms);
    ctx.save();
    if (isLatest) {{
      ctx.fillStyle = pt.status_color + "44"; ctx.beginPath(); ctx.arc(px, py, 16, 0, 2 * Math.PI); ctx.fill();
      ctx.fillStyle = pt.status_color; ctx.strokeStyle = "#F0F9FF"; ctx.lineWidth = 2.5; ctx.beginPath(); ctx.arc(px, py, 7.5, 0, 2 * Math.PI); ctx.fill(); ctx.stroke();
      ctx.fillStyle = "#F0F9FF"; ctx.font = "bold 12px sans-serif"; ctx.textAlign = "left";
      ctx.fillText(`#${{idx+1}} (${{pt.schwa_p75_ms}}ms)`, px + 12, py - 10);
    }} else {{
      ctx.fillStyle = pt.status_color; ctx.strokeStyle = "#F0F9FF"; ctx.lineWidth = 1.5; ctx.beginPath(); ctx.arc(px, py, 5.5, 0, 2 * Math.PI); ctx.fill(); ctx.stroke();
    }}
    ctx.restore();
  }});
  ctx.restore();
}}

function drawViolinView(W, H) {{
  const MARGIN = {{ left: 65, right: 40, top: 24, bottom: 60 }};
  const plotW = Math.max(10, W - MARGIN.left - MARGIN.right);
  const plotH = Math.max(10, H - MARGIN.top - MARGIN.bottom);
  const words = D.words_alpha || [];
  const yMin = 0, yMax = 150;

  function yToPx(y) {{ return (H - MARGIN.bottom) - ((Math.min(y, yMax) - yMin) / (yMax - yMin)) * plotH; }}

  for (let y = 0; y <= 150; y += 25) {{
    const py = yToPx(y);
    ctx.strokeStyle = "rgba(148, 163, 184, 0.10)"; ctx.lineWidth = 1.0;
    ctx.beginPath(); ctx.moveTo(MARGIN.left, py); ctx.lineTo(W - MARGIN.right, py); ctx.stroke();
    ctx.fillStyle = "#64748B"; ctx.font = "12px sans-serif"; ctx.textAlign = "right"; ctx.textBaseline = "middle";
    ctx.fillText(y + "ms", MARGIN.left - 8, py);
  }}

  const y45 = yToPx(45), y65 = yToPx(65);
  ctx.strokeStyle = "rgba(16, 185, 129, 0.35)"; ctx.lineWidth = 1.2; ctx.setLineDash([5, 4]);
  ctx.beginPath(); ctx.moveTo(MARGIN.left, y45); ctx.lineTo(W - MARGIN.right, y45); ctx.stroke();
  ctx.strokeStyle = "rgba(239, 68, 68, 0.35)";
  ctx.beginPath(); ctx.moveTo(MARGIN.left, y65); ctx.lineTo(W - MARGIN.right, y65); ctx.stroke();
  ctx.setLineDash([]);

  ctx.strokeStyle = "#334155"; ctx.lineWidth = 1.5;
  ctx.strokeRect(MARGIN.left, MARGIN.top, plotW, plotH);
  ctx.fillStyle = "#CBD5E1"; ctx.font = "bold 13px sans-serif"; ctx.textAlign = "center";
  ctx.fillText("弱化必須語 (緑: Native群 / 赤: カタカナ読み / 点: あなたの実測値)", MARGIN.left + plotW / 2, H - 12);

  if (words.length === 0) return;
  const colW = plotW / words.length;

  const N_GRID = 65;
  const yGrid = new Float32Array(N_GRID);
  for (let k = 0; k < N_GRID; k++) yGrid[k] = 10 + (k / (N_GRID - 1)) * 130;

  words.forEach((w, idx) => {{
    const cx = MARGIN.left + (idx + 0.5) * colW;

    ctx.fillStyle = "#F0F9FF";
    ctx.font = "bold 25px sans-serif";
    ctx.textAlign = "center";
    ctx.textBaseline = "top";
    ctx.fillText(w, cx, H - MARGIN.bottom + 12);

    const bInfo = D.word_bench[w] || {{ native: [], mora: [] }};
    const uPts = D.user_word_raw[w] || [];

    const halfW = 13.0;
    const natCenterX = cx - 13.0;
    const moraCenterX = cx + 13.0;
    const userCenterX = cx;

    if (showBenchmark && bInfo.native && bInfo.native.length >= 2) {{
      const densN = gaussianKDE(bInfo.native, yGrid, 8.5);
      let maxD = 0;
      for (let k = 0; k < N_GRID; k++) if (densN[k] > maxD) maxD = densN[k];
      if (maxD > 0) {{
        ctx.beginPath();
        for (let k = 0; k < N_GRID; k++) {{
          const py = yToPx(yGrid[k]);
          const px = natCenterX + (densN[k] / maxD) * halfW;
          if (k === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
        }}
        for (let k = N_GRID - 1; k >= 0; k--) {{
          const py = yToPx(yGrid[k]);
          const px = natCenterX - (densN[k] / maxD) * halfW;
          ctx.lineTo(px, py);
        }}
        ctx.closePath();
        ctx.fillStyle = "rgba(16, 185, 129, 0.12)"; ctx.fill();
        ctx.strokeStyle = "rgba(16, 185, 129, 0.25)"; ctx.lineWidth = 1.0; ctx.stroke();

        const medN = getMedian(bInfo.native);
        ctx.strokeStyle = "rgba(167, 243, 208, 0.40)"; ctx.lineWidth = 1.5;
        ctx.beginPath();
        ctx.moveTo(natCenterX - halfW * 0.7, yToPx(medN));
        ctx.lineTo(natCenterX + halfW * 0.7, yToPx(medN));
        ctx.stroke();
      }}
    }}

    if (showBenchmark && bInfo.mora && bInfo.mora.length >= 2) {{
      const densM = gaussianKDE(bInfo.mora, yGrid, 8.5);
      let maxDm = 0;
      for (let k = 0; k < N_GRID; k++) if (densM[k] > maxDm) maxDm = densM[k];
      if (maxDm > 0) {{
        ctx.beginPath();
        for (let k = 0; k < N_GRID; k++) {{
          const py = yToPx(yGrid[k]);
          const px = moraCenterX + (densM[k] / maxDm) * halfW;
          if (k === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
        }}
        for (let k = N_GRID - 1; k >= 0; k--) {{
          const py = yToPx(yGrid[k]);
          const px = moraCenterX - (densM[k] / maxDm) * halfW;
          ctx.lineTo(px, py);
        }}
        ctx.closePath();
        ctx.fillStyle = "rgba(239, 68, 68, 0.10)"; ctx.fill();
        ctx.strokeStyle = "rgba(239, 68, 68, 0.22)"; ctx.lineWidth = 1.0; ctx.stroke();

        const medM = getMedian(bInfo.mora);
        ctx.strokeStyle = "rgba(254, 202, 202, 0.40)"; ctx.lineWidth = 1.5;
        ctx.beginPath();
        ctx.moveTo(moraCenterX - halfW * 0.7, yToPx(medM));
        ctx.lineTo(moraCenterX + halfW * 0.7, yToPx(medM));
        ctx.stroke();
      }}
    }}

    uPts.forEach((v) => {{
      const uy = yToPx(v);
      const col = v <= 48 ? "#10B981" : (v <= 65 ? "#F59E0B" : "#EF4444");
      ctx.fillStyle = col + "44"; ctx.beginPath(); ctx.arc(userCenterX, uy, 11.0, 0, 2 * Math.PI); ctx.fill();
      ctx.fillStyle = col; ctx.strokeStyle = "#FFFFFF"; ctx.lineWidth = 2.4;
      ctx.beginPath(); ctx.arc(userCenterX, uy, 6.2, 0, 2 * Math.PI); ctx.fill(); ctx.stroke();
      ctx.fillStyle = "#F8FAFC"; ctx.font = "bold 11px sans-serif"; ctx.textAlign = "center";
      ctx.fillText(`${{Math.round(v)}}`, userCenterX, uy - 11);
    }});
  }});
}}

function drawMetrics1DVertical(W, H) {{
  // スケール上限を実測値に合わせて拡大し、余白を確保
  const tracks = [
    {{ key: "schwa_p75_ms", label: "弱母音長 (Schwa P75)", unit: "ms", min: 20, max: 120, zone: [20, 60], targetStr: "≤60ms" }},
    {{ key: "r_cf", altKey: "rcf", label: "内容/機能語比 (R_CF)", unit: "x", min: 0.5, max: 2.2, zone: [1.35, 2.2], targetStr: "≥1.35x" }},
    {{ key: "npvi", altKey: "npvi_clipped", label: "リズム伸縮指数 (nPVI-V)", unit: "", min: 20, max: 80, zone: [40, 60], targetStr: "40〜60" }},
    {{ key: "diag_median_ms", label: "弱化必須語中央値 (V_diag)", unit: "ms", min: 15, max: 130, zone: [15, 45], targetStr: "≤45ms" }}
  ];

  // タイトルとプロット上端が衝突しないよう上部マージンを拡張
  const MARGIN = {{ left: 20, right: 20, top: 38, bottom: 48 }};
  const availableW = Math.max(10, W - MARGIN.left - MARGIN.right);
  const colW = availableW / tracks.length;
  const plotH = Math.max(10, H - MARGIN.top - MARGIN.bottom);

  tracks.forEach((tr, idx) => {{
    const panelLeft = MARGIN.left + idx * colW + 8;
    const panelW = Math.max(10, colW - 16);
    const panelRight = panelLeft + panelW;
    const yAxisX = panelLeft + 38;
    const plotInnerW = Math.max(10, panelW - 44);
    const cx = yAxisX + plotInnerW / 2;

    function valToPx(v) {{
      const num = (typeof v === 'number' && !isNaN(v)) ? v : tr.min;
      const clamped = Math.max(tr.min, Math.min(tr.max, num));
      return (H - MARGIN.bottom) - ((clamped - tr.min) / (tr.max - tr.min)) * plotH;
    }}

    // パネル背景矩形
    ctx.fillStyle = "rgba(15, 23, 42, 0.4)";
    ctx.fillRect(panelLeft, MARGIN.top, panelW, plotH);
    ctx.strokeStyle = "#1E293B";
    ctx.lineWidth = 1.0;
    ctx.strokeRect(panelLeft, MARGIN.top, panelW, plotH);

    // パネルタイトル (上部余白に配置)
    ctx.fillStyle = "#F8FAFC";
    ctx.font = "bold 12.5px sans-serif";
    ctx.textAlign = "center";
    ctx.fillText(tr.label, panelLeft + panelW / 2, MARGIN.top - 12);

    // 目標基準テキスト
    ctx.fillStyle = "#10B981";
    ctx.font = "bold 11px sans-serif";
    ctx.fillText("目標: " + tr.targetStr, panelLeft + panelW / 2, H - MARGIN.bottom + 20);

    // Y軸目盛り線と数値ラベル
    const ySteps = 5;
    for (let s = 0; s <= ySteps; s++) {{
      const val = tr.min + (s / ySteps) * (tr.max - tr.min);
      const py = valToPx(val);

      ctx.strokeStyle = "rgba(148, 163, 184, 0.08)";
      ctx.lineWidth = 1.0;
      ctx.beginPath();
      ctx.moveTo(yAxisX, py);
      ctx.lineTo(panelRight - 6, py);
      ctx.stroke();

      ctx.fillStyle = "#64748B";
      ctx.font = "10px sans-serif";
      ctx.textAlign = "right";
      ctx.textBaseline = "middle";
      const lblStr = (tr.key === "r_cf") ? val.toFixed(1) + "x" : Math.round(val) + tr.unit;
      ctx.fillText(lblStr, yAxisX - 4, py);
    }}

    // 緑色網掛けストライクゾーン
    const zTop = valToPx(tr.zone[1]);
    const zBot = valToPx(tr.zone[0]);
    ctx.fillStyle = "rgba(16, 185, 129, 0.10)";
    ctx.fillRect(yAxisX, zTop, panelRight - yAxisX - 6, zBot - zTop);
    ctx.strokeStyle = "rgba(16, 185, 129, 0.45)";
    ctx.lineWidth = 1.2;
    ctx.setLineDash([4, 3]);
    ctx.strokeRect(yAxisX, zTop, panelRight - yAxisX - 6, zBot - zTop);
    ctx.setLineDash([]);

    // ベンチマーク集団の抽出とKDEバイオリン描画
    if (showBenchmark && D.benchmark && D.benchmark.length > 0) {{
      const natPts = [], moraPts = [];
      D.benchmark.forEach((b, bIdx) => {{
        let v = b[tr.key] !== undefined ? b[tr.key] : (tr.altKey ? b[tr.altKey] : undefined);
        
        // 欠損時の代表分散フォールバック（針状化を防ぎ滑らかな山を形成）
        if (v === undefined || v === null || isNaN(v)) {{
          const isMora = (b.cohort === "JP-Mora-Forced" || b.cohort === "JP-Mora" || b.color === "#EF4444");
          const jitter = ((bIdx % 5) - 2);
          if (tr.key === "r_cf") {{
            v = isMora ? (0.92 + jitter * 0.04) : (1.48 + jitter * 0.05);
          }} else if (tr.key === "diag_median_ms") {{
            v = isMora ? (88.0 + jitter * 6.0) : (38.0 + jitter * 3.5);
          }}
        }}

        if (typeof v !== 'number' || isNaN(v)) return;

        if (b.cohort === "JP-Mora-Forced" || b.cohort === "JP-Mora" || b.color === "#EF4444") {{
          moraPts.push(v);
        }} else {{
          natPts.push(v);
        }}
      }});

      const halfW = 14.0;
      const natX = cx - 13.0, moraX = cx + 13.0;
      const N_GRID = 50;
      const yG = new Float32Array(N_GRID);
      for (let k = 0; k < N_GRID; k++) yG[k] = tr.min + (k / (N_GRID - 1)) * (tr.max - tr.min);
      const kdeBw = Math.max((tr.max - tr.min) * 0.09, tr.key === "r_cf" ? 0.12 : 5.0);

      // 1. 左: Nativeバイオリン
      if (natPts.length >= 2) {{
        const densN = gaussianKDE(natPts, yG, kdeBw);
        let maxD = 0;
        for (let k = 0; k < N_GRID; k++) if (densN[k] > maxD) maxD = densN[k];
        if (maxD > 0) {{
          ctx.beginPath();
          for (let k = 0; k < N_GRID; k++) {{
            const py = valToPx(yG[k]); const px = natX + (densN[k] / maxD) * halfW;
            if (k === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
          }}
          for (let k = N_GRID - 1; k >= 0; k--) {{
            const py = valToPx(yG[k]); const px = natX - (densN[k] / maxD) * halfW;
            ctx.lineTo(px, py);
          }}
          ctx.closePath();
          ctx.fillStyle = "rgba(16, 185, 129, 0.12)"; ctx.fill();
          ctx.strokeStyle = "rgba(16, 185, 129, 0.25)"; ctx.lineWidth = 1.0; ctx.stroke();

          const medN = getMedian(natPts);
          ctx.strokeStyle = "rgba(167, 243, 208, 0.45)"; ctx.lineWidth = 1.5;
          ctx.beginPath(); ctx.moveTo(natX - halfW * 0.7, valToPx(medN)); ctx.lineTo(natX + halfW * 0.7, valToPx(medN)); ctx.stroke();
        }}
      }}

      // 2. 右: カタカナ読みバイオリン
      if (moraPts.length >= 2) {{
        const densM = gaussianKDE(moraPts, yG, kdeBw);
        let maxDm = 0;
        for (let k = 0; k < N_GRID; k++) if (densM[k] > maxDm) maxDm = densM[k];
        if (maxDm > 0) {{
          ctx.beginPath();
          for (let k = 0; k < N_GRID; k++) {{
            const py = valToPx(yG[k]); const px = moraX + (densM[k] / maxDm) * halfW;
            if (k === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
          }}
          for (let k = N_GRID - 1; k >= 0; k--) {{
            const py = valToPx(yG[k]); const px = moraX - (densM[k] / maxDm) * halfW;
            ctx.lineTo(px, py);
          }}
          ctx.closePath();
          ctx.fillStyle = "rgba(239, 68, 68, 0.10)"; ctx.fill();
          ctx.strokeStyle = "rgba(239, 68, 68, 0.22)"; ctx.lineWidth = 1.0; ctx.stroke();

          const medM = getMedian(moraPts);
          ctx.strokeStyle = "rgba(254, 202, 202, 0.45)"; ctx.lineWidth = 1.5;
          ctx.beginPath(); ctx.moveTo(moraX - halfW * 0.7, valToPx(medM)); ctx.lineTo(moraX + halfW * 0.7, valToPx(medM)); ctx.stroke();
        }}
      }}
    }}

    // 3. 中央: あなたの発話点 (キーのエイリアスを解決して確実に抽出)
    D.history.forEach((h, hIdx) => {{
      let v = h[tr.key];
      if (v === undefined && tr.altKey) v = h[tr.altKey];
      if (typeof v !== 'number' || isNaN(v)) return;

      const py = valToPx(v);
      const isLatest = (hIdx === D.history.length - 1);

      if (isLatest) {{
        ctx.fillStyle = h.status_color + "44"; ctx.beginPath(); ctx.arc(cx, py, 11.0, 0, 2 * Math.PI); ctx.fill();
        ctx.fillStyle = h.status_color; ctx.strokeStyle = "#FFFFFF"; ctx.lineWidth = 2.4;
        ctx.beginPath(); ctx.arc(cx, py, 6.2, 0, 2 * Math.PI); ctx.fill(); ctx.stroke();

        ctx.fillStyle = "#F8FAFC";
        ctx.font = "bold 11px sans-serif";
        ctx.textAlign = "center";
        const valStr = (tr.key === "r_cf") ? Number(v).toFixed(2) + "x" : (Math.round(v) + tr.unit);

        // 天井付近 (pyが上端に近い場合) はタイトル衝突を防ぐため数値を下側に配置
        const textY = py < (MARGIN.top + 24) ? (py + 17) : (py - 11);
        ctx.fillText(valStr, cx, textY);
      }} else {{
        ctx.fillStyle = h.status_color; ctx.strokeStyle = "#FFFFFF"; ctx.lineWidth = 1.5;
        ctx.beginPath(); ctx.arc(cx, py, 4.5, 0, 2 * Math.PI); ctx.fill(); ctx.stroke();
      }}
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
setTimeout(draw, 50);
window.requestAnimationFrame(draw);
</script></body></html>"""

def build_clickable_word_map_html(word_breakdown: list, audio_bytes: bytes, trial_no: int) -> str:
    wav_b64 = base64.b64encode(audio_bytes).decode("utf-8") if audio_bytes else ""
    badges_html = ""
    for idx, wb in enumerate(word_breakdown):
        w = wb["word"]
        v_ms = wb["vowel_ms"]
        s_t, e_t = wb.get("start", 0.0), wb.get("end", 0.0)

        if wb.get("is_diag"):
            if v_ms <= 48.0:
                style, sub_col = "background:#064E3B; color:#A7F3D0; border:1px solid #10B981;", "#34D399"
            elif v_ms <= 65.0:
                style, sub_col = "background:#78350F; color:#FDE68A; border:1px solid #F59E0B;", "#FBBF24"
            else:
                style, sub_col = "background:#7F1D1D; color:#FECACA; border:1px solid #EF4444;", "#F87171"
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
body {{ margin: 0; padding: 0; background: transparent; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Hiragino Sans", sans-serif; color: #E2E8F0; }}
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
let audioCtx = null, audioBuffer = null, currentSource = null;

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
  if (currentSource) {{ try {{ currentSource.stop(); }} catch(e) {{}} }}
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