import json
from pathlib import Path

import numpy as np
import pandas as pd

SYSTEM_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <title>Sloppy Feelings — System Evolution</title>
  <style>
    body {
      margin: 0;
      background: #0b1120;
      color: #e2e8f0;
      font-family: system-ui, -apple-system, sans-serif;
      display: flex;
      flex-direction: column;
      align-items: center;
      padding: 20px;
      gap: 12px;
    }
    canvas {
      border: 1px solid #334155;
      border-radius: 8px;
      background: #0f172a;
      cursor: crosshair;
    }
    .controls {
      display: flex;
      align-items: center;
      gap: 14px;
      flex-wrap: wrap;
      font-size: 13px;
      background: #1e293b;
      padding: 8px 16px;
      border-radius: 8px;
      border: 1px solid #334155;
    }
    button, select {
      background: #334155;
      color: #f8fafc;
      border: 1px solid #475569;
      border-radius: 4px;
      padding: 4px 10px;
      cursor: pointer;
    }
    #hoverInfo {
      font-size: 12px;
      color: #94a3b8;
      min-height: 18px;
      font-family: monospace;
    }
  </style>
</head>
<body>
  <canvas id="myCanvas" width="680" height="680"></canvas>

  <div class="controls">
    <button id="playBtn">Pause</button>
    <label>Frame: <span id="timeLabel">0</span> / <span id="maxFrameLabel">0</span>
      (<span id="secLabel">0.0</span>s)</label>
    <input id="timeSlider" type="range" min="0" max="1"
           step="0.01" value="0" style="width: 220px;" />
    <label>Speed:
      <select id="speedSelect">
        <option value="0.5">0.5x</option>
        <option value="1" selected>1x</option>
        <option value="2">2x</option>
        <option value="5">5x</option>
      </select>
    </label>
    <label><input id="trailToggle" type="checkbox" checked /> Trails</label>
  </div>
  <div id="hoverInfo">Hover over a particle to inspect its poem &amp; stanza</div>

  <script src="system_data.js"></script>
  <script>
  const payload = window.SYSTEM_DATA || { frames: [], bounds: { xMin: 0, xMax: 1, yMin: 0, yMax: 1, dt: 0.5 }, meta: { poemIds: [], poems: [], stanzas: [], nLines: [] } };
  const frames = payload.frames;
  const bounds = payload.bounds;
  const meta = payload.meta;
  const numFrames = frames.length;
  const numParticles = numFrames > 0 ? frames[0].length : 0;

  const canvas = document.getElementById("myCanvas");
  const ctx = canvas.getContext("2d");
  const W = canvas.width;
  const H = canvas.height;
  const PAD = 28;

  const playBtn = document.getElementById("playBtn");
  const timeSlider = document.getElementById("timeSlider");
  const timeLabel = document.getElementById("timeLabel");
  const maxFrameLabel = document.getElementById("maxFrameLabel");
  const secLabel = document.getElementById("secLabel");
  const speedSelect = document.getElementById("speedSelect");
  const trailToggle = document.getElementById("trailToggle");
  const hoverInfo = document.getElementById("hoverInfo");

  timeSlider.max = Math.max(0, numFrames - 1);
  maxFrameLabel.textContent = Math.max(0, numFrames - 1);

  let isPlaying = true;
  let currentTime = 0;
  let lastTimestamp = performance.now();
  let screenCoords = new Array(numParticles);

  function drawGridAndBounds() {
    ctx.strokeStyle = "rgba(148, 163, 184, 0.12)";
    ctx.lineWidth = 1;
    for (let i = 0; i <= 10; i++) {
      const x = PAD + (i / 10) * (W - 2 * PAD);
      const y = PAD + (i / 10) * (H - 2 * PAD);
      ctx.beginPath(); ctx.moveTo(x, PAD); ctx.lineTo(x, H - PAD); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(PAD, y); ctx.lineTo(W - PAD, y); ctx.stroke();
    }
    ctx.fillStyle = "rgba(148, 163, 184, 0.75)";
    ctx.font = "11px monospace";
    ctx.fillText(`X: [${bounds.xMin}, ${bounds.xMax}]`, PAD + 4, H - 8);
    ctx.fillText(`Y: [${bounds.yMin}, ${bounds.yMax}]`, PAD + 4, 18);
  }

  function renderFrame(t) {
    if (numFrames === 0) return;
    if (trailToggle.checked) {
      ctx.fillStyle = "rgba(15, 23, 42, 0.22)";
      ctx.fillRect(0, 0, W, H);
    } else {
      ctx.fillStyle = "#0f172a";
      ctx.fillRect(0, 0, W, H);
      drawGridAndBounds();
    }

    const t0 = Math.floor(t);
    const t1 = Math.min(t0 + 1, numFrames - 1);
    const alpha = t - t0;

    const frame0 = frames[t0];
    const frame1 = frames[t1];

    for (let i = 0; i < numParticles; i++) {
      const nx = frame0[i][0] * (1 - alpha) + frame1[i][0] * alpha;
      const ny = frame0[i][1] * (1 - alpha) + frame1[i][1] * alpha;

      const x = PAD + nx * (W - 2 * PAD);
      const y = (H - PAD) - ny * (H - 2 * PAD);
      screenCoords[i] = [x, y];

      const hue = (meta.poemIds[i] * 137.5) % 360;
      const radius = Math.min(2 + Math.sqrt(meta.nLines[i] || 1) * 0.6, 6);

      ctx.beginPath();
      ctx.arc(x, y, radius, 0, 2 * Math.PI);
      ctx.fillStyle = `hsla(${hue}, 80%, 65%, 0.88)`;
      ctx.fill();
    }
  }

  function animate(now) {
    const dt = (now - lastTimestamp) / 1000;
    lastTimestamp = now;

    if (isPlaying && numFrames > 1) {
      const speed = parseFloat(speedSelect.value);
      currentTime += (dt / bounds.dt) * speed;
      if (currentTime >= numFrames - 1) {
        currentTime = 0;
        ctx.fillStyle = "#0f172a";
        ctx.fillRect(0, 0, W, H);
      }
      timeSlider.value = currentTime;
      timeLabel.textContent = Math.floor(currentTime);
      secLabel.textContent = (currentTime * bounds.dt).toFixed(1);
    }

    renderFrame(currentTime);
    requestAnimationFrame(animate);
  }

  playBtn.addEventListener("click", () => {
    isPlaying = !isPlaying;
    playBtn.textContent = isPlaying ? "Pause" : "Play";
  });

  timeSlider.addEventListener("input", (e) => {
    currentTime = parseFloat(e.target.value);
    timeLabel.textContent = Math.floor(currentTime);
    secLabel.textContent = (currentTime * bounds.dt).toFixed(1);
    if (!trailToggle.checked) renderFrame(currentTime);
  });

  canvas.addEventListener("mousemove", (e) => {
    const rect = canvas.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    const my = e.clientY - rect.top;
    let bestIdx = -1;
    let bestDist2 = 100;
    for (let i = 0; i < numParticles; i++) {
      if (!screenCoords[i]) continue;
      const dx = screenCoords[i][0] - mx;
      const dy = screenCoords[i][1] - my;
      const d2 = dx * dx + dy * dy;
      if (d2 < bestDist2) {
        bestDist2 = d2;
        bestIdx = i;
      }
    }
    if (bestIdx >= 0) {
      hoverInfo.textContent = `particle #${bestIdx} | poem: ${meta.poems[bestIdx]} (stanza ${meta.stanzas[bestIdx]}, ${meta.nLines[bestIdx]} lines)`;
    } else {
      hoverInfo.textContent = "Hover over a particle to inspect its poem & stanza";
    }
  });

  ctx.fillStyle = "#0f172a";
  ctx.fillRect(0, 0, W, H);
  drawGridAndBounds();
  requestAnimationFrame(animate);
  </script>
</body>
</html>
"""


def normalize_trajectory(
    raw_data: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Normalize a (time, particles, 2) trajectory globally into [0, 1] per axis."""
    mins = raw_data.min(axis=(0, 1))
    maxs = raw_data.max(axis=(0, 1))
    span = np.where(maxs - mins == 0, 1.0, maxs - mins)
    norm_data = np.where(maxs - mins == 0, 0.5, (raw_data - mins) / span)
    return norm_data, mins, maxs


def write_system_html(
    system: np.ndarray,
    stanzas: pd.DataFrame,
    out_path: Path,
    dt: float = 0.05,
    stride: int = 10,
) -> Path:
    """Write `system_data.js` and the static `system.html` viewer in `out_path.parent`."""
    sliced = system[::stride]
    norm_data, mins, maxs = normalize_trajectory(sliced)

    payload = {
        "bounds": {
            "xMin": float(np.round(mins[0], 2)),
            "xMax": float(np.round(maxs[0], 2)),
            "yMin": float(np.round(mins[1], 2)),
            "yMax": float(np.round(maxs[1], 2)),
            "dt": float(dt * stride),
        },
        "meta": {
            "poemIds": pd.factorize(stanzas["poem"])[0].tolist(),
            "poems": stanzas["poem"].tolist(),
            "stanzas": stanzas["stanza"].tolist(),
            "nLines": stanzas["n_lines"].tolist(),
        },
        "frames": np.round(norm_data, 3).tolist(),
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    data_js_path = out_path.parent / "system_data.js"
    data_js_path.write_text(
        f"window.SYSTEM_DATA = {json.dumps(payload)};\n", encoding="utf-8"
    )
    out_path.write_text(SYSTEM_HTML_TEMPLATE, encoding="utf-8")
    return out_path
