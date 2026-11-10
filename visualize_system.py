import json
from pathlib import Path

import numpy as np
import pandas as pd

SYSTEM_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <title>Sloppy Feelings — Parchment &amp; Ink</title>
  <style>
    :root {
      --parchment-bg: #f3e9d2;
      --parchment-canvas: #fbf5e6;
      --parchment-panel: #eadcc3;
      --parchment-border: #c8b496;
      --ink-dark: #3b2d1f;
      --ink-muted: #6e5a44;
      --ink-accent: #8b5a2b;
    }
    body {
      margin: 0;
      background: radial-gradient(circle at center, #f6eedc 0%, #e5d5b7 100%);
      color: var(--ink-dark);
      font-family: "Georgia", "Palatino Linotype", "Book Antiqua", serif;
      display: flex;
      flex-direction: column;
      align-items: center;
      padding: 20px;
      gap: 12px;
      min-height: 100vh;
      box-sizing: border-box;
    }
    .canvas-wrap {
      position: relative;
      border-radius: 6px;
      box-shadow:
        0 8px 28px rgba(59, 45, 31, 0.18),
        inset 0 0 40px rgba(139, 90, 43, 0.12);
      border: 1.5px solid var(--parchment-border);
      background: var(--parchment-canvas);
      overflow: hidden;
    }
    canvas {
      display: block;
      cursor: crosshair;
    }
    .controls {
      display: flex;
      align-items: center;
      gap: 14px;
      flex-wrap: wrap;
      justify-content: center;
      max-width: 780px;
      font-size: 13px;
      background: var(--parchment-panel);
      padding: 10px 18px;
      border-radius: 6px;
      border: 1px solid var(--parchment-border);
      box-shadow: 0 2px 8px rgba(59, 45, 31, 0.08);
    }
    .control-group {
      display: flex;
      align-items: center;
      gap: 6px;
    }
    .divider {
      width: 1px;
      height: 20px;
      background: var(--parchment-border);
    }
    button, select {
      background: #fdf8ec;
      color: var(--ink-dark);
      border: 1px solid #b59e7d;
      border-radius: 4px;
      padding: 4px 11px;
      font-family: inherit;
      font-size: 12.5px;
      cursor: pointer;
      box-shadow: 0 1px 2px rgba(59, 45, 31, 0.08);
    }
    button:hover, select:hover {
      background: #f4ead5;
    }
    input[type="range"] {
      accent-color: var(--ink-accent);
      cursor: pointer;
    }
    input[type="checkbox"] {
      accent-color: var(--ink-accent);
      cursor: pointer;
    }
    .dial-val {
      display: inline-block;
      min-width: 28px;
      font-variant-numeric: tabular-nums;
      color: var(--ink-muted);
      font-size: 12px;
    }
    #hoverInfo {
      font-size: 12.5px;
      color: var(--ink-muted);
      min-height: 18px;
      font-style: italic;
    }
  </style>
</head>
<body>
  <div class="canvas-wrap">
    <canvas id="myCanvas" width="720" height="720"></canvas>
  </div>

  <div class="controls">
    <div class="control-group">
      <button id="playBtn">Pause</button>
      <label>Frame: <span id="timeLabel">0</span>/<span id="maxFrameLabel">0</span>
        (<span id="secLabel">0.0</span>s)</label>
      <input id="timeSlider" type="range" min="0" max="1"
             step="0.01" value="0" style="width: 170px;" />
      <label>Speed:
        <select id="speedSelect">
          <option value="0.5">0.5x</option>
          <option value="1" selected>1x</option>
          <option value="2">2x</option>
          <option value="5">5x</option>
        </select>
      </label>
    </div>

    <div class="divider"></div>

    <div class="control-group">
      <label title="Draw each particle's trajectory through previous time steps as a curve">
        <input id="particleCurveToggle" type="checkbox" checked /> Particle History Curves
      </label>
      <label title="Connect stanzas (0 → 1 → 2...) of the same poem as a curve">
        <input id="poemCurveToggle" type="checkbox" checked /> Poem Stanza Curves
      </label>
    </div>

    <div class="divider"></div>

    <div class="control-group">
      <label for="bezierDial">Bezier Smooth:</label>
      <input id="bezierDial" type="range" min="0" max="1.2" step="0.05" value="0.85" style="width: 95px;" />
      <span id="bezierVal" class="dial-val">0.85</span>
    </div>

    <div class="control-group">
      <label for="noiseDial">Curve Noise:</label>
      <input id="noiseDial" type="range" min="0" max="1" step="0.05" value="0.35" style="width: 95px;" />
      <span id="noiseVal" class="dial-val">0.35</span>
    </div>

    <div class="control-group">
      <label for="blotDial">Ink Blot Size:</label>
      <input id="blotDial" type="range" min="0.4" max="2.2" step="0.1" value="1.0" style="width: 80px;" />
      <span id="blotVal" class="dial-val">1.0</span>
    </div>

    <div class="control-group">
      <label for="tailFadeDial">Tail Fade:</label>
      <input id="tailFadeDial" type="range" min="0" max="1" step="0.05" value="0.50" style="width: 85px;" />
      <span id="tailFadeVal" class="dial-val">0.50</span>
    </div>
  </div>
  <div id="hoverInfo">Hover over an ink blot to inspect its poem &amp; stanza</div>

  <script src="system_data.js"></script>
  <script>
  const payload = window.SYSTEM_DATA || {
    frames: [],
    bounds: { xMin: 0, xMax: 1, yMin: 0, yMax: 1, dt: 0.5 },
    meta: { poemIds: [], poems: [], stanzas: [], nLines: [] }
  };
  const frames = payload.frames;
  const bounds = payload.bounds;
  const meta = payload.meta;
  const numFrames = frames.length;
  const numParticles = numFrames > 0 ? frames[0].length : 0;

  const canvas = document.getElementById("myCanvas");
  const ctx = canvas.getContext("2d");
  const W = canvas.width;
  const H = canvas.height;
  const PAD = 34;

  const playBtn = document.getElementById("playBtn");
  const timeSlider = document.getElementById("timeSlider");
  const timeLabel = document.getElementById("timeLabel");
  const maxFrameLabel = document.getElementById("maxFrameLabel");
  const secLabel = document.getElementById("secLabel");
  const speedSelect = document.getElementById("speedSelect");
  const particleCurveToggle = document.getElementById("particleCurveToggle");
  const poemCurveToggle = document.getElementById("poemCurveToggle");
  const bezierDial = document.getElementById("bezierDial");
  const bezierVal = document.getElementById("bezierVal");
  const noiseDial = document.getElementById("noiseDial");
  const noiseVal = document.getElementById("noiseVal");
  const blotDial = document.getElementById("blotDial");
  const blotVal = document.getElementById("blotVal");
  const tailFadeDial = document.getElementById("tailFadeDial");
  const tailFadeVal = document.getElementById("tailFadeVal");
  const hoverInfo = document.getElementById("hoverInfo");

  timeSlider.max = Math.max(0, numFrames - 1);
  maxFrameLabel.textContent = Math.max(0, numFrames - 1);

  // Deterministic PRNG (Mulberry32)
  function mulberry32(a) {
    return function() {
      let t = (a += 0x6D2B79F5);
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  // Coherent 1D wave noise for hand-drawn ink wobble
  function coherentNoise(s, seed) {
    return (
      0.55 * Math.sin(s * 2.3 + seed * 1.7) +
      0.30 * Math.cos(s * 5.7 - seed * 3.1) +
      0.15 * Math.sin(s * 11.3 + seed * 0.9)
    );
  }

  // Pre-render static parchment texture onto an offscreen canvas for fast blitting
  const parchmentCanvas = document.createElement("canvas");
  parchmentCanvas.width = W;
  parchmentCanvas.height = H;
  const pCtx = parchmentCanvas.getContext("2d");

  function buildParchmentBackground() {
    // Base warm parchment wash
    const rad = pCtx.createRadialGradient(W * 0.48, H * 0.48, W * 0.1, W * 0.5, H * 0.5, W * 0.72);
    rad.addColorStop(0, "#FDF6E2");
    rad.addColorStop(0.7, "#F7EBD0");
    rad.addColorStop(1, "#EAD7B7");
    pCtx.fillStyle = rad;
    pCtx.fillRect(0, 0, W, H);

    // Subtle age mottling / water stains
    const rng = mulberry32(101);
    for (let i = 0; i < 45; i++) {
      const cx = rng() * W;
      const cy = rng() * H;
      const r = 30 + rng() * 110;
      const g = pCtx.createRadialGradient(cx, cy, 2, cx, cy, r);
      const tone = rng() > 0.5 ? "180, 145, 95" : "215, 190, 145";
      g.addColorStop(0, `rgba(${tone}, 0.045)`);
      g.addColorStop(1, `rgba(${tone}, 0)`);
      pCtx.fillStyle = g;
      pCtx.beginPath();
      pCtx.arc(cx, cy, r, 0, Math.PI * 2);
      pCtx.fill();
    }

    // Fine paper fibers & specks
    for (let i = 0; i < 1800; i++) {
      const x = rng() * W;
      const y = rng() * H;
      const alpha = 0.02 + rng() * 0.055;
      pCtx.fillStyle = rng() > 0.35 ? `rgba(120, 90, 55, ${alpha})` : `rgba(255, 250, 235, ${alpha * 1.4})`;
      pCtx.fillRect(x, y, rng() * 1.8, rng() * 1.8);
    }

    // Hand-ruled faint sepia margin lines & coordinate ticks
    pCtx.strokeStyle = "rgba(130, 100, 68, 0.14)";
    pCtx.lineWidth = 0.8;
    for (let i = 0; i <= 10; i++) {
      const x = PAD + (i / 10) * (W - 2 * PAD);
      const y = PAD + (i / 10) * (H - 2 * PAD);
      pCtx.beginPath(); pCtx.moveTo(x, PAD); pCtx.lineTo(x, H - PAD); pCtx.stroke();
      pCtx.beginPath(); pCtx.moveTo(PAD, y); pCtx.lineTo(W - PAD, y); pCtx.stroke();
    }

    pCtx.fillStyle = "rgba(105, 80, 52, 0.65)";
    pCtx.font = "italic 11px Georgia, serif";
    pCtx.fillText(`x ∈ [${bounds.xMin}, ${bounds.xMax}]`, PAD + 4, H - 10);
    pCtx.fillText(`y ∈ [${bounds.yMin}, ${bounds.yMax}]`, PAD + 4, 20);
  }
  buildParchmentBackground();

  // Soft watercolor / ink-blot palette inspired by the hand-drawn reference
  const INK_PALETTE = [
    { h: 38,  s: 82, l: 64 }, // warm ochre / golden yellow
    { h: 356, s: 74, l: 64 }, // coral / vermilion red
    { h: 102, s: 70, l: 65 }, // spring leaf green
    { h: 225, s: 76, l: 66 }, // periwinkle / cobalt ink
    { h: 28,  s: 76, l: 74 }, // soft peach / sienna wash
    { h: 348, s: 68, l: 74 }, // rose pink blot
    { h: 145, s: 65, l: 74 }, // mint / celadon ink
    { h: 262, s: 64, l: 68 }, // lavender / violet ink
    { h: 16,  s: 78, l: 62 }, // terracotta
    { h: 188, s: 62, l: 60 }, // teal wash
  ];

  function getParticleColor(poemId) {
    const base = INK_PALETTE[poemId % INK_PALETTE.length];
    const shift = ((poemId * 29) % 17) - 8;
    const h = (base.h + shift + 360) % 360;
    return {
      fill: `hsla(${h}, ${base.s}%, ${base.l}%, 0.78)`,
      rim: `hsla(${h}, ${base.s + 6}%, ${Math.max(32, base.l - 16)}%, 0.45)`,
      stroke: `hsla(${h}, ${base.s + 8}%, ${Math.max(28, base.l - 12)}%, 0.72)`,
      speck: `hsla(${h}, ${base.s + 10}%, ${Math.max(30, base.l - 10)}%, 0.65)`,
      rgba: (alpha) => `hsla(${h}, ${base.s + 8}%, ${Math.max(28, base.l - 12)}%, ${alpha})`,
    };
  }

  // Group particle indices by poem in stanza sequence (0 -> 1 -> 2...)
  const poemGroups = new Map();
  for (let i = 0; i < numParticles; i++) {
    const pid = meta.poemIds[i];
    if (!poemGroups.has(pid)) poemGroups.set(pid, []);
    poemGroups.get(pid).push(i);
  }
  for (const indices of poemGroups.values()) {
    indices.sort((a, b) => (meta.stanzas[a] || 0) - (meta.stanzas[b] || 0));
  }

  // Precompute each particle's hand-drawn ink-blot shape & micro-splatters
  const blotShapes = new Array(numParticles);
  const NUM_LOBE_PTS = 12;
  for (let i = 0; i < numParticles; i++) {
    const rng = mulberry32(1000 + i * 73);
    const baseR = Math.min(4.2 + Math.sqrt(meta.nLines[i] || 1) * 1.15, 11.5);
    const p1 = rng() * Math.PI * 2;
    const p2 = rng() * Math.PI * 2;
    const p3 = rng() * Math.PI * 2;
    const a1 = 0.08 + rng() * 0.14;
    const a2 = 0.05 + rng() * 0.11;
    const a3 = 0.03 + rng() * 0.07;
    const stretchX = 0.88 + rng() * 0.26;
    const stretchY = 0.88 + rng() * 0.26;
    const rot = rng() * Math.PI * 2;

    const unitPts = [];
    for (let k = 0; k < NUM_LOBE_PTS; k++) {
      const theta = (k / NUM_LOBE_PTS) * Math.PI * 2;
      const rMod = 1 + a1 * Math.sin(2 * theta + p1) + a2 * Math.cos(3 * theta + p2) + a3 * Math.sin(4 * theta + p3);
      const lx = Math.cos(theta) * rMod * stretchX;
      const ly = Math.sin(theta) * rMod * stretchY;
      const rx = lx * Math.cos(rot) - ly * Math.sin(rot);
      const ry = lx * Math.sin(rot) + ly * Math.cos(rot);
      unitPts.push([rx, ry]);
    }

    // Occasional tiny ink卫星/speck near the blot
    const specks = [];
    const nSpecks = rng() < 0.35 ? (rng() < 0.12 ? 2 : 1) : 0;
    for (let s = 0; s < nSpecks; s++) {
      const ang = rng() * Math.PI * 2;
      const dist = 1.35 + rng() * 1.4;
      specks.push({
        dx: Math.cos(ang) * dist,
        dy: Math.sin(ang) * dist,
        r: 0.6 + rng() * 0.9,
      });
    }

    blotShapes[i] = {
      baseR,
      unitPts,
      specks,
      color: getParticleColor(meta.poemIds[i]),
      seed: rng() * 100,
    };
  }

  // Draw a closed smooth ink blot centered at (cx, cy)
  function drawInkBlot(cx, cy, shape, scale) {
    const R = shape.baseR * scale;
    const pts = shape.unitPts;
    const n = pts.length;

    ctx.beginPath();
    const startMidX = cx + ((pts[n - 1][0] + pts[0][0]) * 0.5) * R;
    const startMidY = cy + ((pts[n - 1][1] + pts[0][1]) * 0.5) * R;
    ctx.moveTo(startMidX, startMidY);

    for (let k = 0; k < n; k++) {
      const pCurr = pts[k];
      const pNext = pts[(k + 1) % n];
      const cpx = cx + pCurr[0] * R;
      const cpy = cy + pCurr[1] * R;
      const midX = cx + ((pCurr[0] + pNext[0]) * 0.5) * R;
      const midY = cy + ((pCurr[1] + pNext[1]) * 0.5) * R;
      ctx.quadraticCurveTo(cpx, cpy, midX, midY);
    }
    ctx.closePath();
    ctx.fillStyle = shape.color.fill;
    ctx.fill();

    // Subtle organic pooled-ink edge
    ctx.lineWidth = 0.85;
    ctx.strokeStyle = shape.color.rim;
    ctx.stroke();

    // Draw tiny ink specks around the blot
    for (const sp of shape.specks) {
      ctx.beginPath();
      ctx.arc(cx + sp.dx * R, cy + sp.dy * R, sp.r * Math.sqrt(scale), 0, Math.PI * 2);
      ctx.fillStyle = shape.color.speck;
      ctx.fill();
    }
  }

  /**
   * Draw a sequence of 2D points using Catmull-Rom -> Cubic Bezier smoothing
   * with hand-drawn multi-stroke wobble, optional tail fade, and ink specks along the curve.
   */
  function drawHandDrawnBezierCurve(
    pts,
    colorObj,
    smoothing,
    noiseLevel,
    seedBase,
    lineWidth = 1.6,
    tailFade = 0.0
  ) {
    if (!pts || pts.length < 2) return;

    // Two overlapping strokes mimic the hand-drawn double-looped ink pen look in the reference
    const numPasses = noiseLevel > 0.02 ? 2 : 1;
    const numSegs = pts.length - 1;
    ctx.lineCap = "round";
    ctx.lineJoin = "round";

    for (let pass = 0; pass < numPasses; pass++) {
      const passSeed = seedBase + pass * 19.37;
      const wobbleScale = noiseLevel * (pass === 0 ? 5.5 : 8.0);
      const passWidth = pass === 0 ? lineWidth : lineWidth * 0.72;
      const baseAlpha = pass === 0 ? 0.86 : 0.68;

      // Perturb anchor points slightly per pass so the two strokes weave around each other
      const noisyPts = pts.map((p, idx) => {
        if (wobbleScale === 0) return [p[0], p[1]];
        // Keep endpoints slightly tighter so curves still meet their ink blots cleanly
        const endTaper = (idx === 0 || idx === pts.length - 1) ? 0.35 : 1.0;
        const nx = coherentNoise(idx * 1.4, passSeed) * wobbleScale * endTaper;
        const ny = coherentNoise(idx * 1.4 + 7.1, passSeed + 3.3) * wobbleScale * endTaper;
        return [p[0] + nx, p[1] + ny];
      });

      if (tailFade <= 0.001) {
        ctx.beginPath();
        ctx.moveTo(noisyPts[0][0], noisyPts[0][1]);

        if (noisyPts.length === 2 && smoothing < 0.05) {
          ctx.lineTo(noisyPts[1][0], noisyPts[1][1]);
        } else {
          for (let i = 0; i < numSegs; i++) {
            const p0 = noisyPts[Math.max(0, i - 1)];
            const p1 = noisyPts[i];
            const p2 = noisyPts[i + 1];
            const p3 = noisyPts[Math.min(noisyPts.length - 1, i + 2)];

            const tension = smoothing / 3.0;
            let cp1x = p1[0] + (p2[0] - p0[0]) * tension;
            let cp1y = p1[1] + (p2[1] - p0[1]) * tension;
            let cp2x = p2[0] - (p3[0] - p1[0]) * tension;
            let cp2y = p2[1] - (p3[1] - p1[1]) * tension;

            if (wobbleScale > 0) {
              const dx = p2[0] - p1[0];
              const dy = p2[1] - p1[1];
              const segLen = Math.hypot(dx, dy);
              const perpScale = Math.min(segLen * 0.28, wobbleScale * 1.45);
              const n1 = coherentNoise(i * 2.1 + 0.5, passSeed + 11.0);
              const n2 = coherentNoise(i * 2.1 + 1.3, passSeed + 23.0);
              if (segLen > 1e-3) {
                const px = -dy / segLen;
                const py = dx / segLen;
                cp1x += px * n1 * perpScale;
                cp1y += py * n1 * perpScale;
                cp2x += px * n2 * perpScale;
                cp2y += py * n2 * perpScale;
              }
            }

            ctx.bezierCurveTo(cp1x, cp1y, cp2x, cp2y, p2[0], p2[1]);
          }
        }

        ctx.strokeStyle = colorObj.rgba(baseAlpha);
        ctx.lineWidth = passWidth;
        ctx.stroke();
      } else {
        // Per-segment tail fade from oldest segment (i=0) to newest head segment (i=numSegs-1)
        const fadeExp = tailFade * 4.0;
        for (let i = 0; i < numSegs; i++) {
          const progress = (i + 1) / numSegs;
          const segAlpha = baseAlpha * Math.pow(progress, fadeExp);
          if (segAlpha < 0.025) continue;

          const p0 = noisyPts[Math.max(0, i - 1)];
          const p1 = noisyPts[i];
          const p2 = noisyPts[i + 1];
          const p3 = noisyPts[Math.min(noisyPts.length - 1, i + 2)];

          const tension = smoothing / 3.0;
          let cp1x = p1[0] + (p2[0] - p0[0]) * tension;
          let cp1y = p1[1] + (p2[1] - p0[1]) * tension;
          let cp2x = p2[0] - (p3[0] - p1[0]) * tension;
          let cp2y = p2[1] - (p3[1] - p1[1]) * tension;

          if (wobbleScale > 0) {
            const dx = p2[0] - p1[0];
            const dy = p2[1] - p1[1];
            const segLen = Math.hypot(dx, dy);
            const perpScale = Math.min(segLen * 0.28, wobbleScale * 1.45);
            const n1 = coherentNoise(i * 2.1 + 0.5, passSeed + 11.0);
            const n2 = coherentNoise(i * 2.1 + 1.3, passSeed + 23.0);
            if (segLen > 1e-3) {
              const px = -dy / segLen;
              const py = dx / segLen;
              cp1x += px * n1 * perpScale;
              cp1y += py * n1 * perpScale;
              cp2x += px * n2 * perpScale;
              cp2y += py * n2 * perpScale;
            }
          }

          ctx.beginPath();
          ctx.moveTo(p1[0], p1[1]);
          if (numSegs === 1 && smoothing < 0.05) {
            ctx.lineTo(p2[0], p2[1]);
          } else {
            ctx.bezierCurveTo(cp1x, cp1y, cp2x, cp2y, p2[0], p2[1]);
          }
          ctx.strokeStyle = colorObj.rgba(segAlpha);
          ctx.lineWidth = passWidth * (0.45 + 0.55 * Math.pow(progress, tailFade * 1.2));
          ctx.stroke();
        }
      }
    }

    // Sprinkle subtle stray ink specks near the curve when Curve Noise is active
    if (noiseLevel > 0.05 && pts.length >= 2) {
      const speckCount = Math.min(4, Math.floor(pts.length * noiseLevel * 0.45));
      for (let s = 0; s < speckCount; s++) {
        const frac = (s + 0.5) / speckCount;
        const speckAlpha = 0.78 * (tailFade > 0.001 ? Math.pow(frac, tailFade * 4.0) : 1.0);
        if (speckAlpha < 0.03) continue;
        const idx = Math.min(pts.length - 1, Math.floor(frac * pts.length));
        const basePt = pts[idx];
        const ox = coherentNoise(s * 3.7, seedBase + 51.2) * (8 + noiseLevel * 22);
        const oy = coherentNoise(s * 3.7 + 4.4, seedBase + 67.8) * (8 + noiseLevel * 22);
        const r = 0.7 + Math.abs(coherentNoise(s, seedBase)) * 1.1;
        ctx.beginPath();
        ctx.arc(basePt[0] + ox, basePt[1] + oy, r, 0, Math.PI * 2);
        ctx.fillStyle = colorObj.rgba(speckAlpha);
        ctx.fill();
      }
    }
  }

  let isPlaying = true;
  let currentTime = 0;
  let lastTimestamp = performance.now();
  let screenCoords = new Array(numParticles);

  function toScreen(nx, ny) {
    return [
      PAD + nx * (W - 2 * PAD),
      (H - PAD) - ny * (H - 2 * PAD),
    ];
  }

  function renderFrame(t) {
    if (numFrames === 0) return;

    // Blit clean parchment background every frame; previous time values are drawn as curves
    ctx.drawImage(parchmentCanvas, 0, 0);

    const t0 = Math.floor(t);
    const t1 = Math.min(t0 + 1, numFrames - 1);
    const alpha = t - t0;

    const frame0 = frames[t0];
    const frame1 = frames[t1];

    const smoothing = parseFloat(bezierDial.value);
    const noiseLevel = parseFloat(noiseDial.value);
    const blotScale = parseFloat(blotDial.value);
    const tailFade = parseFloat(tailFadeDial.value);

    // Compute current interpolated screen coordinates for all particles
    for (let i = 0; i < numParticles; i++) {
      const nx = frame0[i][0] * (1 - alpha) + frame1[i][0] * alpha;
      const ny = frame0[i][1] * (1 - alpha) + frame1[i][1] * alpha;
      screenCoords[i] = toScreen(nx, ny);
    }

    // 1. Particle History Curves: trajectory of each particle from frame 0 through currentTime
    if (particleCurveToggle.checked && t >= 0.05) {
      // Subsample historical keyframes so Bezier smoothing creates graceful sweeping arcs
      const keyframeStep = Math.max(1, Math.floor(t0 / 12));
      for (let i = 0; i < numParticles; i++) {
        const historyPts = [];
        for (let f = 0; f <= t0; f += keyframeStep) {
          historyPts.push(toScreen(frames[f][i][0], frames[f][i][1]));
        }
        // Always include the exact current interpolated position at the tip
        const cur = screenCoords[i];
        const last = historyPts[historyPts.length - 1];
        if (!last || Math.hypot(cur[0] - last[0], cur[1] - last[1]) > 0.5) {
          historyPts.push(cur);
        }
        drawHandDrawnBezierCurve(
          historyPts,
          blotShapes[i].color,
          smoothing,
          noiseLevel,
          blotShapes[i].seed,
          1.35,
          tailFade
        );
      }
    }

    // 2. Poem Stanza Curves: connect stanzas (0 -> 1 -> 2...) of each poem
    if (poemCurveToggle.checked) {
      for (const [poemId, indices] of poemGroups.entries()) {
        if (indices.length < 2) continue;
        const stanzaPts = indices.map((idx) => screenCoords[idx]);
        const colorObj = getParticleColor(poemId);
        drawHandDrawnBezierCurve(
          stanzaPts,
          colorObj,
          smoothing,
          noiseLevel,
          poemId * 13.7 + 7.0,
          1.75
        );
      }
    }

    // 3. Draw hand-drawn ink-blot particles on top
    for (let i = 0; i < numParticles; i++) {
      const [x, y] = screenCoords[i];
      drawInkBlot(x, y, blotShapes[i], blotScale);
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
    renderFrame(currentTime);
  });

  bezierDial.addEventListener("input", () => {
    bezierVal.textContent = parseFloat(bezierDial.value).toFixed(2);
    if (!isPlaying) renderFrame(currentTime);
  });

  noiseDial.addEventListener("input", () => {
    noiseVal.textContent = parseFloat(noiseDial.value).toFixed(2);
    if (!isPlaying) renderFrame(currentTime);
  });

  blotDial.addEventListener("input", () => {
    blotVal.textContent = parseFloat(blotDial.value).toFixed(1);
    if (!isPlaying) renderFrame(currentTime);
  });

  tailFadeDial.addEventListener("input", () => {
    tailFadeVal.textContent = parseFloat(tailFadeDial.value).toFixed(2);
    if (!isPlaying) renderFrame(currentTime);
  });

  particleCurveToggle.addEventListener("change", () => {
    if (!isPlaying) renderFrame(currentTime);
  });

  poemCurveToggle.addEventListener("change", () => {
    if (!isPlaying) renderFrame(currentTime);
  });

  canvas.addEventListener("mousemove", (e) => {
    const rect = canvas.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    const my = e.clientY - rect.top;
    let bestIdx = -1;
    let bestDist2 = 144;
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
      hoverInfo.textContent = `ink blot #${bestIdx} — poem: ${meta.poems[bestIdx]} (stanza ${meta.stanzas[bestIdx]}, ${meta.nLines[bestIdx]} lines)`;
    } else {
      hoverInfo.textContent = "Hover over an ink blot to inspect its poem & stanza";
    }
  });

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
