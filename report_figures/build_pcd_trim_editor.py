"""
Build a self-contained HTML "trim bench" for hand-removing points from a point
cloud, meant to feed grasp_tilt_pointcloud.py.

It bakes every point of the source .pcd (quantised x/y/z + rgb) into one HTML
file. Open the file as a Claude Artifact and:

  * switch between the three axis-aligned views (Top XY / Front XZ / Side YZ),
  * lasso regions to delete them (points inside the polygon, in that view's
    two in-plane axes, are dropped),
  * copy the resulting REMOVE_CUTS block and paste it into
    grasp_tilt_pointcloud.py.

The cuts are pure geometry (polygons in world metres), so they are applied to
the FULL-resolution cloud by the figure script -- the editor only needs a
faithful preview.

Usage:
    python3 report_figures/build_pcd_trim_editor.py \
        [--pcd experiments/map/raw.pcd] \
        [--out /path/to/pcd_trim_bench.html]
"""

import argparse
import base64
import json
import os
import struct

import numpy as np

DEFAULT_PCD = "/root/Crate_truss_pick-and-place/experiments/map/raw.pcd"
DEFAULT_OUT = (
    "/tmp/claude-0/-root/12dd01bc-b94c-40e1-b3f8-3e756cf5c25d/"
    "scratchpad/pcd_trim_bench.html"
)


def load_pcd_ascii(path):
    with open(path) as f:
        lines = f.readlines()
    fields = None
    data_start = None
    for i, line in enumerate(lines):
        if line.startswith("FIELDS"):
            fields = line.split()[1:]
        elif line.startswith("DATA"):
            data_start = i + 1
            break
    data = np.loadtxt(lines[data_start:], dtype=np.float64)
    xyz = data[:, :3]
    colors = np.full((len(xyz), 3), 180, dtype=np.uint8)
    if fields is not None and "rgb" in fields:
        rgb_raw = data[:, fields.index("rgb")].astype(np.uint32)
        colors = np.stack(
            [(rgb_raw >> 16) & 255, (rgb_raw >> 8) & 255, rgb_raw & 255], axis=1
        ).astype(np.uint8)
    return xyz, colors


def quantise(xyz):
    """Per-axis linear quantisation to uint16 over the axis's own range."""
    lo = xyz.min(axis=0)
    hi = xyz.max(axis=0)
    span = np.where(hi > lo, hi - lo, 1.0)
    q = np.round((xyz - lo) / span * 65535.0).astype(np.uint16)
    return q, lo, span


def build_blob(q, colors):
    n = len(q)
    parts = [
        q[:, 0].astype("<u2").tobytes(),
        q[:, 1].astype("<u2").tobytes(),
        q[:, 2].astype("<u2").tobytes(),
        colors[:, 0].tobytes(),
        colors[:, 1].tobytes(),
        colors[:, 2].tobytes(),
    ]
    return base64.b64encode(b"".join(parts)).decode("ascii"), n


def percentile_bounds(xyz, lo_p=0.5, hi_p=99.5):
    return {
        "x": [np.percentile(xyz[:, 0], lo_p), np.percentile(xyz[:, 0], hi_p)],
        "y": [np.percentile(xyz[:, 1], lo_p), np.percentile(xyz[:, 1], hi_p)],
        "z": [np.percentile(xyz[:, 2], lo_p), np.percentile(xyz[:, 2], hi_p)],
    }


PAGE = r"""<title>Truss Cloud Trim Bench</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Archivo:wght@500;600;700&family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;500;600&display=swap">
<style>
:root {
  --bg: #eef0ec;
  --panel: #ffffff;
  --panel-2: #f6f7f4;
  --ink: #1a1e19;
  --ink-dim: #5c635a;
  --ink-faint: #8a9186;
  --line: #d9ddd4;
  --line-strong: #c3c9bc;
  --accent: #2f9e8f;         /* confirm / primary */
  --accent-ink: #16463f;
  --cut: #c65b1e;            /* lasso / destructive */
  --cut-soft: rgba(198,91,30,0.14);
  --removed: #d06a3a;
  --viewport: #0d1012;      /* stays dark in both themes -- instrument choice */
  --viewport-grid: rgba(150,170,170,0.16);
  --viewport-hud: rgba(220,235,232,0.72);
  --focus: #2f9e8f;
  --radius: 7px;
  --font-ui: "IBM Plex Sans", system-ui, sans-serif;
  --font-display: "Archivo", var(--font-ui);
  --font-mono: "IBM Plex Mono", ui-monospace, "SFMono-Regular", monospace;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --bg: #14171a;
    --panel: #1b1f22;
    --panel-2: #23282b;
    --ink: #e7eae4;
    --ink-dim: #a3aaa0;
    --ink-faint: #737b71;
    --line: #2e3438;
    --line-strong: #3c444a;
    --accent: #3fb5a4;
    --accent-ink: #bdeee6;
    --cut: #e07a3f;
    --cut-soft: rgba(224,122,63,0.18);
    --removed: #e07a3f;
  }
}
:root[data-theme="dark"] {
  --bg: #14171a;
  --panel: #1b1f22;
  --panel-2: #23282b;
  --ink: #e7eae4;
  --ink-dim: #a3aaa0;
  --ink-faint: #737b71;
  --line: #2e3438;
  --line-strong: #3c444a;
  --accent: #3fb5a4;
  --accent-ink: #bdeee6;
  --cut: #e07a3f;
  --cut-soft: rgba(224,122,63,0.18);
  --removed: #e07a3f;
}

* { box-sizing: border-box; }
html, body { height: 100%; }
body {
  margin: 0;
  background: var(--bg);
  color: var(--ink);
  font-family: var(--font-ui);
  font-size: 14px;
  line-height: 1.45;
  -webkit-font-smoothing: antialiased;
}
.app {
  display: grid;
  grid-template-columns: 312px 1fr 340px;
  grid-template-rows: 100vh;
  gap: 0;
}
.rail, .side {
  overflow-y: auto;
  background: var(--panel);
  display: flex;
  flex-direction: column;
}
.rail { border-right: 1px solid var(--line); }
.side { border-left: 1px solid var(--line); }
.stage {
  position: relative;
  min-width: 0;
  background: var(--viewport);
}
canvas#view { display: block; width: 100%; height: 100%; touch-action: none; cursor: crosshair; }
.stage[data-mode="pan"] canvas#view { cursor: grab; }
.stage[data-mode="pan"].dragging canvas#view { cursor: grabbing; }

.block { padding: 16px 18px; border-bottom: 1px solid var(--line); }
.block:last-child { border-bottom: 0; }

.brand { display: flex; align-items: baseline; gap: 9px; flex-wrap: wrap; }
.brand h1 {
  font-family: var(--font-display);
  font-weight: 700;
  font-size: 15px;
  letter-spacing: 0.02em;
  margin: 0;
  text-transform: uppercase;
}
.brand .tag {
  font-family: var(--font-mono);
  font-size: 11px;
  color: var(--ink-faint);
}
.lede { margin: 8px 0 0; color: var(--ink-dim); font-size: 12.5px; }

.label {
  font-family: var(--font-mono);
  font-size: 10.5px;
  letter-spacing: 0.11em;
  text-transform: uppercase;
  color: var(--ink-faint);
  margin: 0 0 9px;
}

.seg { display: flex; gap: 4px; background: var(--panel-2); border: 1px solid var(--line); border-radius: var(--radius); padding: 4px; }
.seg button {
  flex: 1;
  appearance: none;
  border: 0;
  background: transparent;
  color: var(--ink-dim);
  font-family: var(--font-ui);
  font-size: 12px;
  font-weight: 600;
  padding: 7px 6px;
  border-radius: 5px;
  cursor: pointer;
  line-height: 1.2;
}
.seg button .sub { display: block; font-family: var(--font-mono); font-weight: 400; font-size: 10px; color: var(--ink-faint); margin-top: 2px; }
.seg button[aria-pressed="true"] { background: var(--panel); color: var(--ink); box-shadow: 0 1px 2px rgba(0,0,0,0.12); }
.seg button[aria-pressed="true"] .sub { color: var(--accent); }
.seg.mode button[aria-pressed="true"][data-val="lasso"] { color: var(--cut); }
.seg.mode button[aria-pressed="true"][data-val="lasso"] .sub { color: var(--cut); }

.rows { display: flex; flex-direction: column; gap: 13px; }
.ctl { display: flex; flex-direction: column; gap: 6px; }
.ctl .row { display: flex; align-items: center; justify-content: space-between; gap: 10px; }
.ctl output { font-family: var(--font-mono); font-size: 12px; color: var(--ink-dim); }
input[type="range"] { width: 100%; accent-color: var(--accent); }

.switches { display: flex; flex-direction: column; gap: 9px; }
.switch { display: flex; align-items: center; gap: 9px; cursor: pointer; font-size: 12.5px; color: var(--ink-dim); }
.switch input { accent-color: var(--accent); width: 15px; height: 15px; }

.btn {
  appearance: none;
  font-family: var(--font-ui);
  font-weight: 600;
  font-size: 12.5px;
  padding: 8px 12px;
  border-radius: 6px;
  border: 1px solid var(--line-strong);
  background: var(--panel-2);
  color: var(--ink);
  cursor: pointer;
}
.btn:hover { border-color: var(--ink-faint); }
.btn:disabled { opacity: 0.45; cursor: not-allowed; }
.btn.wide { width: 100%; }
.btn.primary { background: var(--accent); border-color: var(--accent); color: #fff; }
.btn.danger { color: var(--cut); border-color: color-mix(in srgb, var(--cut) 45%, var(--line-strong)); }
.btn-row { display: flex; gap: 8px; }
.btn-row .btn { flex: 1; }

.stat-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 1px; background: var(--line); border: 1px solid var(--line); border-radius: var(--radius); overflow: hidden; }
.stat { background: var(--panel); padding: 10px 12px; }
.stat .n { font-family: var(--font-mono); font-size: 16px; font-weight: 500; letter-spacing: -0.01em; }
.stat .k { font-family: var(--font-mono); font-size: 9.5px; letter-spacing: 0.1em; text-transform: uppercase; color: var(--ink-faint); margin-top: 2px; }
.stat.cut .n { color: var(--cut); }

.hud {
  position: absolute;
  top: 12px; left: 14px;
  font-family: var(--font-mono);
  font-size: 11.5px;
  color: var(--viewport-hud);
  pointer-events: none;
  display: flex;
  flex-direction: column;
  gap: 3px;
  text-shadow: 0 1px 3px rgba(0,0,0,0.55);
}
.hud .k { color: rgba(220,235,232,0.5); }
.hint {
  position: absolute;
  bottom: 12px; left: 14px; right: 14px;
  font-family: var(--font-mono);
  font-size: 11px;
  color: var(--viewport-hud);
  pointer-events: none;
  text-shadow: 0 1px 3px rgba(0,0,0,0.55);
}
.hint b { color: #fff; font-weight: 500; }

.cutlist { display: flex; flex-direction: column; gap: 6px; }
.cut-row {
  display: flex; align-items: center; gap: 9px;
  padding: 8px 10px;
  border: 1px solid var(--line);
  border-radius: 6px;
  background: var(--panel-2);
  font-size: 12px;
}
.cut-row:hover { border-color: var(--cut); }
.cut-row .badge {
  font-family: var(--font-mono);
  font-size: 10px;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  padding: 2px 6px;
  border-radius: 4px;
  background: var(--cut-soft);
  color: var(--cut);
}
.cut-row .meta { color: var(--ink-dim); font-family: var(--font-mono); font-size: 11px; }
.cut-row .grow { flex: 1; }
.cut-row button {
  appearance: none; border: 0; background: transparent;
  color: var(--ink-faint); cursor: pointer; font-size: 15px; line-height: 1;
  padding: 2px 4px;
}
.cut-row button:hover { color: var(--cut); }
.empty { color: var(--ink-faint); font-size: 12px; font-style: italic; padding: 6px 2px; }

textarea {
  width: 100%;
  min-height: 132px;
  resize: vertical;
  font-family: var(--font-mono);
  font-size: 11px;
  line-height: 1.5;
  color: var(--ink);
  background: var(--panel-2);
  border: 1px solid var(--line);
  border-radius: 6px;
  padding: 10px;
}
.io-note { font-size: 11.5px; color: var(--ink-faint); margin: 7px 0 0; }
.copied { color: var(--accent); font-weight: 600; }

:focus-visible { outline: 2px solid var(--focus); outline-offset: 1px; }

@media (max-width: 1080px) {
  .app { grid-template-columns: 1fr; grid-template-rows: auto 62vh auto; }
  .rail { border-right: 0; border-bottom: 1px solid var(--line); }
  .side { border-left: 0; border-top: 1px solid var(--line); }
}
@media (prefers-reduced-motion: reduce) { * { transition: none !important; } }
</style>

<div class="app">
  <aside class="rail">
    <div class="block">
      <div class="brand">
        <h1>Trim Bench</h1>
        <span class="tag">__PCDNAME__</span>
      </div>
      <p class="lede">Lasso away clutter from the scanned truss, then hand the cut list to <span style="font-family:var(--font-mono);font-size:11.5px">grasp_tilt_pointcloud.py</span>. Cuts are polygons in world metres and apply to the full-resolution cloud.</p>
    </div>

    <div class="block">
      <p class="label">Projection</p>
      <div class="seg view" role="group" aria-label="View plane">
        <button data-val="xy" aria-pressed="true">Top<span class="sub">x · y</span></button>
        <button data-val="xz" aria-pressed="false">Front<span class="sub">x · z</span></button>
        <button data-val="yz" aria-pressed="false">Side<span class="sub">y · z</span></button>
      </div>
    </div>

    <div class="block">
      <p class="label">Tool</p>
      <div class="seg mode" role="group" aria-label="Pointer tool">
        <button data-val="pan" aria-pressed="true">Pan / Zoom<span class="sub">P</span></button>
        <button data-val="lasso" aria-pressed="false">Lasso cut<span class="sub">L</span></button>
      </div>
      <div class="btn-row" style="margin-top:10px">
        <button class="btn" id="finishCut" disabled>Finish cut ⏎</button>
        <button class="btn" id="cancelCut" disabled>Cancel esc</button>
      </div>
    </div>

    <div class="block rows">
      <div class="ctl">
        <div class="row"><p class="label" style="margin:0">Point size</p><output id="sizeOut">1 px</output></div>
        <input type="range" id="sizeR" min="1" max="5" step="1" value="1">
      </div>
      <div class="ctl">
        <p class="label" style="margin:0 0 2px">Colour</p>
        <div class="seg color" role="group" aria-label="Point colour">
          <button data-val="rgb" aria-pressed="true">Scan</button>
          <button data-val="depth" aria-pressed="false">Depth</button>
          <button data-val="flat" aria-pressed="false">Flat</button>
        </div>
      </div>
      <div class="switches">
        <label class="switch"><input type="checkbox" id="showRemoved" checked> Show removed points (dim)</label>
        <label class="switch"><input type="checkbox" id="fitAll"> Frame all points (incl. far spray)</label>
      </div>
      <button class="btn wide" id="reset">Reset view to fit</button>
    </div>

    <div class="block">
      <p class="label">Tally</p>
      <div class="stat-grid">
        <div class="stat"><div class="n" id="nTotal">–</div><div class="k">total pts</div></div>
        <div class="stat"><div class="n" id="nKept">–</div><div class="k">kept</div></div>
        <div class="stat cut"><div class="n" id="nCut">0</div><div class="k">removed</div></div>
        <div class="stat cut"><div class="n" id="pCut">0%</div><div class="k">removed</div></div>
      </div>
    </div>
  </aside>

  <main class="stage" id="stage" data-mode="pan">
    <canvas id="view"></canvas>
    <div class="hud">
      <div><span class="k">view </span><span id="hudView">TOP · x,y</span></div>
      <div><span class="k">cursor </span><span id="hudPos">–</span></div>
      <div><span class="k">scale </span><span id="hudScale">–</span></div>
    </div>
    <div class="hint" id="hint">Drag to pan · scroll to zoom · press <b>L</b> to start a lasso cut</div>
  </main>

  <aside class="side">
    <div class="block">
      <p class="label">Cuts <span id="cutCount" style="color:var(--cut)"></span></p>
      <div class="cutlist" id="cutlist"></div>
      <div class="btn-row" style="margin-top:10px">
        <button class="btn danger" id="undo" disabled>Undo last</button>
        <button class="btn danger" id="clearAll" disabled>Clear all</button>
      </div>
    </div>

    <div class="block">
      <p class="label">Export → paste into the figure script</p>
      <textarea id="exportBox" readonly spellcheck="false"></textarea>
      <div class="btn-row" style="margin-top:8px">
        <button class="btn primary" id="copyBtn">Copy REMOVE_CUTS</button>
      </div>
      <p class="io-note" id="copyNote">Paste this block near the top of <span style="font-family:var(--font-mono)">grasp_tilt_pointcloud.py</span>.</p>
    </div>

    <div class="block">
      <p class="label">Resume a previous session</p>
      <textarea id="importBox" spellcheck="false" placeholder="Paste a REMOVE_CUTS block (or a bare JSON array) here to reload it..."></textarea>
      <div class="btn-row" style="margin-top:8px">
        <button class="btn wide" id="importBtn">Load cuts</button>
      </div>
      <p class="io-note" id="importNote"></p>
    </div>
  </aside>
</div>

<script id="cloud-data" type="application/octet-stream">__BLOB__</script>
<script id="cloud-meta" type="application/json">__META__</script>
<script>
(function () {
  "use strict";
  var meta = JSON.parse(document.getElementById("cloud-meta").textContent);
  var N = meta.n;

  // ---- decode baked cloud -------------------------------------------------
  var b64 = document.getElementById("cloud-data").textContent.trim();
  var raw = atob(b64);
  var bytes = new Uint8Array(raw.length);
  for (var i = 0; i < raw.length; i++) bytes[i] = raw.charCodeAt(i);
  var off = 0;
  var qx = new Uint16Array(bytes.buffer, off, N); off += N * 2;
  var qy = new Uint16Array(bytes.buffer, off, N); off += N * 2;
  var qz = new Uint16Array(bytes.buffer, off, N); off += N * 2;
  var cr = new Uint8Array(bytes.buffer, off, N); off += N;
  var cg = new Uint8Array(bytes.buffer, off, N); off += N;
  var cb = new Uint8Array(bytes.buffer, off, N); off += N;

  var lo = meta.lo, span = meta.span;         // dequant: world = lo + q/65535*span
  var wx = new Float32Array(N), wy = new Float32Array(N), wz = new Float32Array(N);
  for (i = 0; i < N; i++) {
    wx[i] = lo[0] + qx[i] / 65535 * span[0];
    wy[i] = lo[1] + qy[i] / 65535 * span[1];
    wz[i] = lo[2] + qz[i] / 65535 * span[2];
  }
  var axisArr = { x: wx, y: wy, z: wz };
  var VIEWS = {
    xy: { u: "x", v: "y", label: "TOP · x,y" },
    xz: { u: "x", v: "z", label: "FRONT · x,z" },
    yz: { u: "y", v: "z", label: "SIDE · y,z" }
  };

  // ---- state ------------------------------------------------------------
  var state = {
    view: "xy",
    mode: "pan",
    size: 1,
    color: "rgb",
    showRemoved: true,
    fitAll: false,
    cam: { cx: 0, cy: 0, scale: 1 },
    cuts: [],                 // {view, poly:[[u,v],...]}
    draft: null,              // in-progress poly (world coords)
    hoverCut: -1
  };
  var removed = new Uint8Array(N);   // 1 => inside some cut for its view

  var stage = document.getElementById("stage");
  var canvas = document.getElementById("view");
  var ctx = canvas.getContext("2d");
  var W = 0, H = 0, dpr = 1, img = null, buf = null;

  function bgU32() {
    // viewport bg #0d1012 -> little-endian 0xAABBGGRR
    return (255 << 24) | (0x12 << 16) | (0x10 << 8) | 0x0d;
  }

  function resize() {
    dpr = Math.min(window.devicePixelRatio || 1, 2);
    var r = stage.getBoundingClientRect();
    W = Math.max(1, Math.round(r.width * dpr));
    H = Math.max(1, Math.round(r.height * dpr));
    canvas.width = W; canvas.height = H;
    canvas.style.width = r.width + "px";
    canvas.style.height = r.height + "px";
    img = ctx.createImageData(W, H);
    buf = new Uint32Array(img.data.buffer);
    render();
  }

  function axIdx(a) { return a === "x" ? 0 : a === "y" ? 1 : 2; }

  function boundsFor(viewKey, all) {
    var v = VIEWS[viewKey];
    if (all) {
      return {
        umin: lo[axIdx(v.u)], umax: lo[axIdx(v.u)] + span[axIdx(v.u)],
        vmin: lo[axIdx(v.v)], vmax: lo[axIdx(v.v)] + span[axIdx(v.v)]
      };
    }
    var pb = meta.pbounds;   // percentile bounds per axis
    return {
      umin: pb[v.u][0], umax: pb[v.u][1],
      vmin: pb[v.v][0], vmax: pb[v.v][1]
    };
  }

  function fitView() {
    var b = boundsFor(state.view, state.fitAll);
    var cu = (b.umin + b.umax) / 2, cv = (b.vmin + b.vmax) / 2;
    var du = (b.umax - b.umin) || 1, dv = (b.vmax - b.vmin) || 1;
    var pad = 0.92;
    var s = Math.min(W / du, H / dv) * pad;
    state.cam = { cx: cu, cy: cv, scale: s };
  }

  // world <-> screen (screen in device px)
  function toScreen(u, v) {
    return [
      (u - state.cam.cx) * state.cam.scale + W / 2,
      H / 2 - (v - state.cam.cy) * state.cam.scale
    ];
  }
  function toWorld(sx, sy) {
    return [
      (sx - W / 2) / state.cam.scale + state.cam.cx,
      state.cam.cy - (sy - H / 2) / state.cam.scale
    ];
  }

  // ---- point-in-polygon (ray cast) ------------------------------------
  function pointInPoly(px, py, poly) {
    var inside = false, n = poly.length, j = n - 1;
    for (var k = 0; k < n; k++) {
      var xi = poly[k][0], yi = poly[k][1];
      var xj = poly[j][0], yj = poly[j][1];
      if (((yi > py) !== (yj > py)) &&
          (px < (xj - xi) * (py - yi) / (yj - yi) + xi)) inside = !inside;
      j = k;
    }
    return inside;
  }

  function recomputeRemoved() {
    removed.fill(0);
    for (var c = 0; c < state.cuts.length; c++) {
      var cut = state.cuts[c];
      var v = VIEWS[cut.view];
      var ua = axisArr[v.u], va = axisArr[v.v];
      var poly = cut.poly;
      // bbox fast-reject
      var minx = Infinity, miny = Infinity, maxx = -Infinity, maxy = -Infinity;
      for (var p = 0; p < poly.length; p++) {
        if (poly[p][0] < minx) minx = poly[p][0];
        if (poly[p][0] > maxx) maxx = poly[p][0];
        if (poly[p][1] < miny) miny = poly[p][1];
        if (poly[p][1] > maxy) maxy = poly[p][1];
      }
      for (var idx = 0; idx < N; idx++) {
        if (removed[idx]) continue;
        var pu = ua[idx], pv = va[idx];
        if (pu < minx || pu > maxx || pv < miny || pv > maxy) continue;
        if (pointInPoly(pu, pv, poly)) removed[idx] = 1;
      }
    }
    updateTally();
    syncExport();
  }

  // ---- render ---------------------------------------------------------
  var rafPending = false;
  function render() {
    if (rafPending) return;
    rafPending = true;
    requestAnimationFrame(function () { rafPending = false; draw(); });
  }

  function draw() {
    if (!buf) return;
    var bg = bgU32();
    buf.fill(bg);

    var v = VIEWS[state.view];
    var ua = axisArr[v.u], va = axisArr[v.v];
    var cx = state.cam.cx, cy = state.cam.cy, sc = state.cam.scale;
    var halfW = W / 2, halfH = H / 2;
    var sz = state.size;
    var showRem = state.showRemoved;
    var mode = state.color;

    // depth ramp over the out-of-plane axis
    var thirdKey = ({ xy: "z", xz: "y", yz: "x" })[state.view];
    var ta = axisArr[thirdKey];
    var tmin = meta.pbounds[thirdKey][0], tmax = meta.pbounds[thirdKey][1];
    var trange = (tmax - tmin) || 1;

    var remCol = (255 << 24) | (0x3a << 16) | (0x6a << 8) | 0xd0; // dim orange
    var flatCol = (255 << 24) | (0xc8 << 16) | (0xd4 << 8) | 0x7f; // soft cyan

    for (var i = 0; i < N; i++) {
      var rem = removed[i];
      if (rem && !showRem) continue;
      var sx = (ua[i] - cx) * sc + halfW;
      if (sx < 0 || sx >= W) continue;
      var sy = halfH - (va[i] - cy) * sc;
      if (sy < 0 || sy >= H) continue;
      var col;
      if (rem) {
        col = remCol;
      } else if (mode === "rgb") {
        col = (255 << 24) | (cb[i] << 16) | (cg[i] << 8) | cr[i];
      } else if (mode === "flat") {
        col = flatCol;
      } else {
        var t = (ta[i] - tmin) / trange;
        if (t < 0) t = 0; else if (t > 1) t = 1;
        // teal -> sand ramp
        var rr = Math.round(40 + t * 200);
        var gg = Math.round(150 + t * 60);
        var bbc = Math.round(150 - t * 70);
        col = (255 << 24) | (bbc << 16) | (gg << 8) | rr;
      }
      var px = sx | 0, py = sy | 0;
      if (sz === 1) {
        buf[py * W + px] = col;
      } else {
        var h = sz >> 1;
        var x0 = px - h, x1 = px + (sz - 1 - h);
        var y0 = py - h, y1 = py + (sz - 1 - h);
        if (x0 < 0) x0 = 0; if (y0 < 0) y0 = 0;
        if (x1 >= W) x1 = W - 1; if (y1 >= H) y1 = H - 1;
        for (var yy = y0; yy <= y1; yy++) {
          var rowo = yy * W;
          for (var xx = x0; xx <= x1; xx++) buf[rowo + xx] = col;
        }
      }
    }
    ctx.putImageData(img, 0, 0);

    drawGrid();
    drawCuts();
    drawDraft();
  }

  function niceStep(rangeWorld, targetPx, scale) {
    var raw = targetPx / scale;               // world units per target spacing
    var mag = Math.pow(10, Math.floor(Math.log10(raw)));
    var norm = raw / mag;
    var step = norm < 1.5 ? 1 : norm < 3.5 ? 2 : norm < 7.5 ? 5 : 10;
    return step * mag;
  }

  function drawGrid() {
    var v = VIEWS[state.view];
    ctx.save();
    ctx.lineWidth = 1;
    ctx.strokeStyle = "rgba(150,170,170,0.14)";
    ctx.fillStyle = "rgba(210,228,224,0.6)";
    ctx.font = "11px 'IBM Plex Mono', monospace";
    var tl = toWorld(0, 0), br = toWorld(W, H);
    var step = niceStep(br[0] - tl[0], 96 * dpr, state.cam.scale);

    var u0 = Math.ceil(tl[0] / step) * step;
    for (var u = u0; u <= br[0]; u += step) {
      var s = toScreen(u, 0)[0];
      ctx.beginPath(); ctx.moveTo(s, 0); ctx.lineTo(s, H); ctx.stroke();
      ctx.fillText(u.toFixed(2), s + 4, H - 6);
    }
    var v0 = Math.ceil(br[1] / step) * step;
    for (var vv = v0; vv <= tl[1]; vv += step) {
      var sy = toScreen(0, vv)[1];
      ctx.beginPath(); ctx.moveTo(0, sy); ctx.lineTo(W, sy); ctx.stroke();
      ctx.fillText(vv.toFixed(2), 6, sy - 4);
    }
    // origin
    var o = toScreen(0, 0);
    ctx.strokeStyle = "rgba(150,170,170,0.4)";
    ctx.beginPath(); ctx.moveTo(o[0] - 6, o[1]); ctx.lineTo(o[0] + 6, o[1]);
    ctx.moveTo(o[0], o[1] - 6); ctx.lineTo(o[0], o[1] + 6); ctx.stroke();
    ctx.restore();
  }

  function polyPath(poly) {
    ctx.beginPath();
    for (var k = 0; k < poly.length; k++) {
      var s = toScreen(poly[k][0], poly[k][1]);
      if (k === 0) ctx.moveTo(s[0], s[1]); else ctx.lineTo(s[0], s[1]);
    }
  }

  function drawCuts() {
    for (var c = 0; c < state.cuts.length; c++) {
      if (state.cuts[c].view !== state.view) continue;
      var hot = c === state.hoverCut;
      ctx.save();
      polyPath(state.cuts[c].poly);
      ctx.closePath();
      ctx.fillStyle = hot ? "rgba(224,122,63,0.22)" : "rgba(224,122,63,0.10)";
      ctx.fill();
      ctx.lineWidth = hot ? 2.5 : 1.5;
      ctx.strokeStyle = "rgba(230,130,70,0.95)";
      ctx.setLineDash(hot ? [] : [5, 4]);
      ctx.stroke();
      ctx.restore();
    }
  }

  function drawDraft() {
    if (!state.draft || !state.draft.length) return;
    ctx.save();
    polyPath(state.draft);
    ctx.lineWidth = 2;
    ctx.strokeStyle = "#f0a86e";
    ctx.stroke();
    // vertices
    for (var k = 0; k < state.draft.length; k++) {
      var s = toScreen(state.draft[k][0], state.draft[k][1]);
      ctx.beginPath();
      ctx.arc(s[0], s[1], k === 0 ? 5 : 3.2, 0, 7);
      ctx.fillStyle = k === 0 ? "#fff" : "#f0a86e";
      ctx.fill();
    }
    if (mouse.inside) {
      var last = toScreen(state.draft[state.draft.length - 1][0], state.draft[state.draft.length - 1][1]);
      ctx.beginPath();
      ctx.moveTo(last[0], last[1]);
      ctx.lineTo(mouse.x, mouse.y);
      ctx.setLineDash([4, 4]);
      ctx.strokeStyle = "rgba(240,168,110,0.7)";
      ctx.stroke();
    }
    ctx.restore();
  }

  // ---- interaction --------------------------------------------------
  var mouse = { x: 0, y: 0, inside: false };
  var drag = null;

  function evtPos(e) {
    var r = canvas.getBoundingClientRect();
    return [(e.clientX - r.left) * dpr, (e.clientY - r.top) * dpr];
  }

  canvas.addEventListener("pointerdown", function (e) {
    canvas.setPointerCapture(e.pointerId);
    var p = evtPos(e);
    if (state.mode === "lasso") {
      var w = toWorld(p[0], p[1]);
      if (!state.draft) state.draft = [];
      // click near first vertex closes
      if (state.draft.length >= 3) {
        var s0 = toScreen(state.draft[0][0], state.draft[0][1]);
        if (Math.hypot(s0[0] - p[0], s0[1] - p[1]) < 12 * dpr) { finishCut(); return; }
      }
      state.draft.push([w[0], w[1]]);
      updateCutButtons();
      render();
    } else {
      drag = { x: p[0], y: p[1], cx: state.cam.cx, cy: state.cam.cy };
      stage.classList.add("dragging");
    }
  });

  canvas.addEventListener("pointermove", function (e) {
    var p = evtPos(e);
    mouse.x = p[0]; mouse.y = p[1]; mouse.inside = true;
    var w = toWorld(p[0], p[1]);
    var v = VIEWS[state.view];
    document.getElementById("hudPos").textContent =
      v.u + " " + w[0].toFixed(3) + "   " + v.v + " " + w[1].toFixed(3);
    if (drag) {
      state.cam.cx = drag.cx - (p[0] - drag.x) / state.cam.scale;
      state.cam.cy = drag.cy + (p[1] - drag.y) / state.cam.scale;
      render();
    } else if (state.mode === "lasso" && state.draft) {
      render();
    }
  });

  function endDrag(e) {
    if (drag) { drag = null; stage.classList.remove("dragging"); }
    try { canvas.releasePointerCapture(e.pointerId); } catch (x) {}
  }
  canvas.addEventListener("pointerup", endDrag);
  canvas.addEventListener("pointercancel", endDrag);
  canvas.addEventListener("pointerleave", function () { mouse.inside = false; render(); });

  canvas.addEventListener("wheel", function (e) {
    e.preventDefault();
    var p = evtPos(e);
    var before = toWorld(p[0], p[1]);
    var f = Math.exp(-e.deltaY * 0.0016);
    state.cam.scale *= f;
    state.cam.scale = Math.max(1, Math.min(state.cam.scale, 5e5));
    var after = toWorld(p[0], p[1]);
    state.cam.cx += before[0] - after[0];
    state.cam.cy += before[1] - after[1];
    updateHudScale();
    render();
  }, { passive: false });

  // ---- cuts lifecycle --------------------------------------------
  function finishCut() {
    if (!state.draft || state.draft.length < 3) return;
    state.cuts.push({ view: state.view, poly: state.draft.map(function (q) { return [q[0], q[1]]; }) });
    state.draft = null;
    updateCutButtons();
    renderCutList();
    recomputeRemoved();
    render();
  }
  function cancelCut() {
    state.draft = null;
    updateCutButtons();
    render();
  }
  document.getElementById("finishCut").addEventListener("click", finishCut);
  document.getElementById("cancelCut").addEventListener("click", cancelCut);

  function updateCutButtons() {
    var has = !!(state.draft && state.draft.length);
    document.getElementById("finishCut").disabled = !(state.draft && state.draft.length >= 3);
    document.getElementById("cancelCut").disabled = !has;
  }

  document.getElementById("undo").addEventListener("click", function () {
    state.cuts.pop();
    renderCutList(); recomputeRemoved(); render();
  });
  document.getElementById("clearAll").addEventListener("click", function () {
    if (!state.cuts.length) return;
    if (!confirm("Remove all " + state.cuts.length + " cuts?")) return;
    state.cuts = [];
    renderCutList(); recomputeRemoved(); render();
  });

  function renderCutList() {
    var box = document.getElementById("cutlist");
    box.innerHTML = "";
    if (!state.cuts.length) {
      box.innerHTML = '<div class="empty">No cuts yet. Pick a view, press L, click a polygon around the clutter, then ⏎.</div>';
    }
    state.cuts.forEach(function (cut, idx) {
      var row = document.createElement("div");
      row.className = "cut-row";
      row.innerHTML =
        '<span class="badge">' + cut.view + '</span>' +
        '<span class="meta">#' + (idx + 1) + '</span>' +
        '<span class="grow meta">' + cut.poly.length + ' pts</span>' +
        '<button title="delete cut" aria-label="delete cut ' + (idx + 1) + '">&times;</button>';
      row.addEventListener("mouseenter", function () { state.hoverCut = idx; render(); });
      row.addEventListener("mouseleave", function () { state.hoverCut = -1; render(); });
      row.querySelector("button").addEventListener("click", function () {
        state.cuts.splice(idx, 1);
        renderCutList(); recomputeRemoved(); render();
      });
      box.appendChild(row);
    });
    document.getElementById("undo").disabled = !state.cuts.length;
    document.getElementById("clearAll").disabled = !state.cuts.length;
    document.getElementById("cutCount").textContent = state.cuts.length ? "· " + state.cuts.length : "";
  }

  function updateTally() {
    var rc = 0;
    for (var i = 0; i < N; i++) rc += removed[i];
    document.getElementById("nTotal").textContent = N.toLocaleString();
    document.getElementById("nKept").textContent = (N - rc).toLocaleString();
    document.getElementById("nCut").textContent = rc.toLocaleString();
    document.getElementById("pCut").textContent = (rc / N * 100).toFixed(1) + "%";
  }

  // ---- export / import ------------------------------------------
  function round5(x) { return Math.round(x * 1e5) / 1e5; }
  function syncExport() {
    var body = state.cuts.map(function (c) {
      var pts = c.poly.map(function (p) { return "[" + round5(p[0]) + ", " + round5(p[1]) + "]"; }).join(", ");
      return '    {"view": "' + c.view + '", "poly": [' + pts + "]},";
    }).join("\n");
    var txt =
      "# Trim Bench cuts for " + meta.pcdname + " -- polygons in world metres.\n" +
      "#   view \"xy\" tests (x, y)   \"xz\" tests (x, z)   \"yz\" tests (y, z)\n" +
      "# A point is dropped if it lies inside ANY polygon for that view.\n" +
      "REMOVE_CUTS = [\n" + (body ? body + "\n" : "") + "]\n";
    document.getElementById("exportBox").value = txt;
  }

  document.getElementById("copyBtn").addEventListener("click", function () {
    var ta = document.getElementById("exportBox");
    ta.select();
    var ok = false;
    try { ok = document.execCommand("copy"); } catch (e) {}
    if (navigator.clipboard) {
      navigator.clipboard.writeText(ta.value).then(function () {
        flash("copyNote", "Copied ✓");
      }, function () { if (!ok) flash("copyNote", "Press ⌘/Ctrl+C — text is selected"); });
    }
    if (ok) flash("copyNote", "Copied ✓");
    window.getSelection().removeAllRanges();
  });

  function flash(id, msg) {
    var el = document.getElementById(id);
    var prev = el.innerHTML;
    el.innerHTML = '<span class="copied">' + msg + "</span>";
    setTimeout(function () { el.innerHTML = prev; }, 1800);
  }

  document.getElementById("importBtn").addEventListener("click", function () {
    var raw = document.getElementById("importBox").value;
    var a = raw.indexOf("["), b = raw.lastIndexOf("]");
    if (a < 0 || b < a) { flash("importNote", "Could not find a [ ... ] array."); return; }
    var arr;
    try { arr = JSON.parse(raw.slice(a, b + 1)); }
    catch (e) { flash("importNote", "Not valid JSON: " + e.message); return; }
    if (!Array.isArray(arr)) { flash("importNote", "Expected a JSON array."); return; }
    var clean = [];
    for (var i = 0; i < arr.length; i++) {
      var c = arr[i];
      if (c && VIEWS[c.view] && Array.isArray(c.poly) && c.poly.length >= 3) {
        clean.push({ view: c.view, poly: c.poly.map(function (p) { return [+p[0], +p[1]]; }) });
      }
    }
    if (!clean.length) { flash("importNote", "No usable cuts in that block."); return; }
    state.cuts = clean;
    renderCutList(); recomputeRemoved(); render();
    flash("importNote", "Loaded " + clean.length + " cut" + (clean.length > 1 ? "s" : "") + ".");
  });

  // ---- controls wiring ----------------------------------------
  function wireSeg(sel, key, after) {
    var group = document.querySelector(sel);
    group.querySelectorAll("button").forEach(function (btn) {
      btn.addEventListener("click", function () {
        group.querySelectorAll("button").forEach(function (b) { b.setAttribute("aria-pressed", "false"); });
        btn.setAttribute("aria-pressed", "true");
        state[key] = btn.dataset.val;
        if (after) after();
      });
    });
  }
  wireSeg(".seg.view", "view", function () {
    document.getElementById("hudView").textContent = VIEWS[state.view].label;
    state.draft = null; updateCutButtons();
    fitView(); updateHudScale(); renderCutList(); render();
  });
  wireSeg(".seg.mode", "mode", function () {
    stage.dataset.mode = state.mode;
    if (state.mode === "pan") { state.draft = null; updateCutButtons(); }
    document.getElementById("hint").innerHTML = state.mode === "lasso"
      ? "Click to drop lasso points · click the white dot or press <b>⏎</b> to close · <b>esc</b> cancels · <b>⌫</b> drops last point"
      : "Drag to pan · scroll to zoom · press <b>L</b> to start a lasso cut";
    render();
  });
  wireSeg(".seg.color", "color", render);

  document.getElementById("sizeR").addEventListener("input", function (e) {
    state.size = +e.target.value;
    document.getElementById("sizeOut").textContent = state.size + " px";
    render();
  });
  document.getElementById("showRemoved").addEventListener("change", function (e) {
    state.showRemoved = e.target.checked; render();
  });
  document.getElementById("fitAll").addEventListener("change", function (e) {
    state.fitAll = e.target.checked; fitView(); updateHudScale(); render();
  });
  document.getElementById("reset").addEventListener("click", function () {
    fitView(); updateHudScale(); render();
  });

  function updateHudScale() {
    var pxPerCm = state.cam.scale / dpr / 100;
    document.getElementById("hudScale").textContent = pxPerCm.toFixed(1) + " px/cm";
  }

  window.addEventListener("keydown", function (e) {
    if (e.target.tagName === "TEXTAREA") return;
    var k = e.key.toLowerCase();
    if (k === "l") { setMode("lasso"); }
    else if (k === "p") { setMode("pan"); }
    else if (k === "enter") { if (state.draft) finishCut(); }
    else if (k === "escape") { if (state.draft) cancelCut(); }
    else if (k === "backspace") {
      if (state.draft && state.draft.length) { state.draft.pop(); updateCutButtons(); render(); e.preventDefault(); }
    } else if (k === "z" && (e.ctrlKey || e.metaKey)) {
      if (state.cuts.length) { state.cuts.pop(); renderCutList(); recomputeRemoved(); render(); }
    } else if (["1", "2", "3"].indexOf(k) >= 0) {
      setView(["xy", "xz", "yz"][+k - 1]);
    }
  });
  function setMode(m) {
    document.querySelector('.seg.mode button[data-val="' + m + '"]').click();
  }
  function setView(v) {
    document.querySelector('.seg.view button[data-val="' + v + '"]').click();
  }

  // ---- boot -------------------------------------------------
  window.addEventListener("resize", resize);
  document.getElementById("hudView").textContent = VIEWS.xy.label;
  renderCutList();
  updateTally();
  syncExport();
  resize();
  fitView();
  updateHudScale();
  render();
})();
</script>
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pcd", default=DEFAULT_PCD)
    ap.add_argument("--out", default=DEFAULT_OUT)
    args = ap.parse_args()

    xyz, colors = load_pcd_ascii(args.pcd)
    q, lo, span = quantise(xyz)
    blob, n = build_blob(q, colors)
    pbounds = percentile_bounds(xyz)

    meta = {
        "n": n,
        "lo": [float(x) for x in lo],
        "span": [float(x) for x in span],
        "pbounds": pbounds,
        "pcdname": os.path.basename(args.pcd),
    }

    html = (
        PAGE.replace("__BLOB__", blob)
        .replace("__META__", json.dumps(meta))
        .replace("__PCDNAME__", os.path.basename(args.pcd))
    )
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        f.write(html)
    print(f"points baked : {n:,}")
    print(f"blob (b64)   : {len(blob) / 1e6:.2f} MB")
    print(f"html written : {args.out}  ({os.path.getsize(args.out) / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
