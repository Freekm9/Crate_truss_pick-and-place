# Planner path replay page

Interactive 3D replay (three.js) of the saved planner-quality benchmark paths: time slider over the
retimed execution time, per grasp point all planners side by side, gripper pose now / at the goal /
at the grasp, arm of the followed planner, progress chart and per-planner metrics table.

Published at https://claude.ai/artifact/2wJ2a8cC9wL5r3GtjzWowg (private; share from the page).

- `build.py` reads `experiments/planner_benchmark/quality_20261020` (10 cm fingers) and
  `quality_finger5cm` (simplified + raw variants) and writes `site/data/` (index.json, one file per
  6 goals, truss clouds per scan; ~48 MB, git-ignored, regenerate with `python3 build.py`).
- `site/index.html` is the page; it loads `data/index.json` and then only the selected goal's file.
- Open locally: `cd site && python3 -m http.server` and browse to http://localhost:8000.
