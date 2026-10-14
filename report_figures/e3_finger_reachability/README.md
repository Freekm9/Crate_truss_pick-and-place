# E3 – Reachability vs finger extension (simulation)

Regenerate with `python3 make_e3_figures.py`. It reads the per-pose results in
`experiments/finger_reachability/full_20261014/`. Every number below is also in
`numbers.txt` and `table_e3.csv`.

| File | Use in the report |
|---|---|
| `table_e3.tex` | Full results table (`\label{tab:e3_finger_reachability}`, needs `booktabs`) |
| `reachability_vs_finger.pdf` | Main figure: reachability against finger extension |
| `reachability_by_wall.pdf` | Where the difference is: reachability against distance to the wall |

---

## Method (text for the thesis)

E3 measures how many realistic grasp poses each grasp generator can reach, as a function of
the finger extension. Finger extension is the length added beyond the stock Franka fingertip.

**Grasp points.** Nine point clouds of a filled crate were captured with the wrist camera from
the robot's capture pose. Between captures the trusses were rearranged, including trusses
along the walls, in the corners and in lower layers. The stems were segmented from each
cloud. Grasp points were placed every 2 cm along each truss's main stem, at least 2 cm from
the stem ends, at the stem's height. Each grasp point has a yaw: the gripper's closing axis
is perpendicular to the local stem direction, as in the pick pipeline. To cover errors in
the operator's stem direction, every point was also evaluated at yaw offsets of ±15° and
±30°. This gave 358 grasp points on 52 trusses, and 1790 grasp poses per finger extension and
generator. 138 of the points lie within 5 cm of a crate wall. No point lies closer than 1 cm,
because the tomatoes keep the stem away from the wall.

**Reachability check.** For each pose, each generator chooses an orientation exactly as the
pick pipeline does:
- The **top-down** generator (Van den Bent's 4-DOF baseline) tries straight down, then the
  same orientation rolled 180° about the approach axis.
- The **tilted** generator (ours) tries those two first, and then tilts of up to ±80° about
  two axes, smallest tilt first.

An orientation counts as reachable only if the whole pick sequence has collision-free IK
solutions: pre-grasp, grasp, back-out and lift. The straight-line motions between them must
also be executable.

This was evaluated with three obstacle models:
- **Walls:** the crate walls only.
- **Corridor:** the crate contents added as a height-map obstacle, built from the same
  cloud, with only a 4 cm-radius corridor along the approach left free.
- **Gripper:** the same height map, with the whole gripper volume along the approach left
  free.

As in the pipeline, an orientation whose pre-grasp collides with the contents is not
replaced by another tilt.

The sweep covered extensions of 0, 5, 9, 10, 13 and 20 cm, each in MoveIt with the matching
robot model. Confidence intervals are 95% cluster-bootstrap intervals that resample trusses,
because poses on the same stem are not independent. The paired difference between the
generators was tested with an exact McNemar test on the same poses.

## Results (text for the thesis)

With the stock fingers (0 cm), the top-down generator reached 35.3% of the grasp poses
[24, 47], and the tilted generator 91.2% [88, 94] (gripper model; Table
`tab:e3_finger_reachability`, Figure `reachability_vs_finger`). On the 1790 paired poses,
1000 were reachable only with a tilt and none only top-down (McNemar p < 10⁻³⁰⁰).

The difference shrank quickly with longer fingers:

| Extension | Tilted | Top-down | Poses reachable only by tilted |
|---|---|---|---|
| 0 cm | 91.2% | 35.3% | 1000 |
| 5 cm | 99.3% | 94.2% | 91 (p < 10⁻²⁷) |
| 9–13 cm | 99.7% | 99.6% | at most 3 (p ≥ 0.25) |

At no extension did the top-down generator reach a pose that the tilted generator could not.
The shortest extension that reached at least 95% of the poses was **5 cm for the tilted
generator and 9 cm for top-down**.

**Wall distance** (Figure `reachability_by_wall`). The poses that only tilting reaches lie
close to the walls.
- At 0 cm, top-down reached 14–30% of the poses within 5 cm of a wall, and tilted 40–94%.
- At 5 cm, top-down still lost poses up to 4 cm from the wall: 63% at 1–2 cm, 67% at 2–3 cm.
  Tilted lost poses only within 2 cm: 80% at 1–2 cm, 95% at 2–3 cm.
- From 9 cm on, both generators reached every pose more than 2 cm from a wall. The only
  remaining difference was at 1–2 cm: 86% tilted versus 77% top-down, on 35 poses.
- At 20 cm, top-down lost a few poses again between 2 and 4 cm from the wall (95–98%). The
  cause was not analysed; a likely explanation is that the longer fingers reach the wall
  at the pre-grasp height.

**Tilts used.** With 0 cm fingers, the tilted generator tilted 1000 of its 1790 poses; the
median summed tilt was 40°. At 5 cm it tilted only 91 poses, by a median of 20°, and from
9 cm on at most 12.

**Obstacle model.** The crate contents mattered only for the tilted generator with stock
fingers. With the corridor-only model, its reachability fell from 91.2% to 58.1%: a tilted
approach brings the hand down beside the stem, into the space of the neighbouring trusses.
From 5 cm on, all three obstacle models gave the same result within 0.4 percentage points.
The real-robot experiment (E4) uses the gripper model.

**Interpretation.** Tilting is a substitute for finger length. The tilted generator reaches
95% of realistic grasp poses with half the extension that top-down needs (5 vs 9 cm), and
with stock fingers it roughly triples reachability. At the 10 cm extension used on the
robot, both generators reach practically every pose in simulation. A difference between
them in E4 is therefore not expected from reachability, but at most from the poses within
2 cm of a wall and from execution: grasp quality and disturbance of the pile.

## Figure captions

**reachability_vs_finger.pdf** – Fraction of the 1790 grasp poses (358 points × 5 yaws) for
which the generator finds a feasible pick sequence, against the finger extension beyond the
stock fingertip. Crate walls and crate contents (gripper model) are obstacles. Error bars:
95% cluster-bootstrap CI over the 52 trusses. Dashed line: 95%. Dotted line: the extension
used on the robot.

**reachability_by_wall.pdf** – Reachable fraction by distance from the grasp point to the
nearest crate wall, for 0, 5 and 10 cm extensions (gripper model). Poses per bin: 1–2 cm 35,
2–3 cm 105, 3–4 cm 305, 4–5 cm 245, 5–7 cm 525, 7–10 cm 405, >10 cm 170.

## Limitations (for the discussion)

- **Kinematic reachability only.** E3 counts poses with a collision-free, executable pick
  sequence in simulation. Whether the grasp then holds, and how much it disturbs the pile,
  is measured in E4. E4 also logs the predicted reachability of every real attempt, which
  checks E3 against the robot.
- **Crate contents as a 2 cm height map.** The obstacle model of the contents is a height
  map with 2 cm cells. Overhangs and soft contact are not modelled.
- **Few points right at the walls.** Only 35 poses lie within 2 cm of a wall, from 7 points,
  so the 1–2 cm bin is uncertain.
- **One crate and one camera pose.** All scans used the same crate type and capture pose.
  Other crate dimensions or a different robot base position would change the numbers.
