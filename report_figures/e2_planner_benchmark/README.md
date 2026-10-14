# E2 – Planner selection (stage A, simulation)

Regenerate everything here with `python3 make_e2_figures.py`. It reads the full-grid results in
`experiments/planner_benchmark/20261013_134011/`. Every number below is also in `numbers.txt`
and `table_e2.csv`.

| File | Use in the report |
|---|---|
| `table_e2.tex` | Main results table (`\label{tab:e2_planner_benchmark}`, needs `booktabs`) |
| `planning_time.pdf` | Figure: planning time per planner |
| `path_ratio.pdf` | Figure: path ratio per planner |

---

## Method (text for the thesis)

The planners were compared in a planning-only benchmark in simulation (Gazebo + MoveIt,
Franka Panda, 9 cm finger extensions). The collision scene was the crate model plus a height
map of the crate contents, built from a real point cloud of a filled crate (scan
20261006_112906) as in the real pipeline. Grasp points were placed on a 5 cm grid over the
crate interior, 1.5 cm in from the walls, with four yaws (0°, 45°, 90°, 135°). For each grasp
point, the grasp generator chose a feasible orientation. Its pre-grasp pose was resolved to a
single joint goal: the collision-free IK solution nearest to the start configuration. Each
grasp gave two planning queries:

- **Capture pose to pre-grasp:** from the real robot's camera capture configuration to the
  pre-grasp goal, with the crate contents in the scene.
- **Lift to place:** from the configuration after lifting the truss to above the real place
  pose, without the crate contents, as in the real pipeline.

Because every planner plans to the same fixed joint goal, differences between planners come
only from the planner itself. This gave 212 unique start/goal pairs (106 per move). Each
planner planned every pair 5 times, so 1060 plans per planner, in shuffled planner order. The
settings matched the real pipeline: a 5 s planning time and 100 planning attempts. The
candidates were RRTConnect, BiTRRT, RRT* (OMPL), CHOMP and Pink (differential IK).

Metrics:
- **Success.** Reported with a Wilson 95% confidence interval.
- **Planning time.**
- **Path ratio.** The joint-space path length divided by the straight-line joint distance
  from start to goal; 1.0 is the shortest possible joint path.
- **End-effector ratio.** The TCP path length divided by the straight-line TCP distance.
- **Minimum clearance to the crate walls.**

The selection rule was fixed before the run:
1. Success within 2 percentage points of the best planner.
2. 95th-percentile planning time ≤ 2 s.
3. Lowest median path ratio, where planners within 0.02 count as tied.
4. Ties broken by the lower median planning time.

## Results (text for the thesis)

RRTConnect, BiTRRT, RRT* and CHOMP solved every query: 1060/1060 plans each, 95% CI
[99.3, 100]% per move (Table `tab:e2_planner_benchmark`). Pink solved 69.8% of the moves to
pre-grasp [65.8, 73.6] and all moves to the place pose. Its 160 failures were 130 plans that
MoveIt rejected for collision and 30 where the IK stalled before reaching the goal. Pink
therefore fails step 1 of the selection rule.

The planners differed most in planning time (`planning_time.pdf`):

| Planner | Median planning time | 95th percentile |
|---|---|---|
| RRTConnect | 0.041 s | 0.059 s |
| BiTRRT | 0.047 s | 0.073 s |
| CHOMP | 0.55 s | 0.68 s |
| RRT* | 5.03 s | 5.05 s |

RRT* is an anytime planner that keeps optimising for the whole time budget. It used the full
5 s on every query without finding a shorter path, and fails step 2.

Path quality did not separate the remaining planners (`path_ratio.pdf`). For RRTConnect,
BiTRRT and CHOMP the median and 95th-percentile path ratio were 1.000 for both moves: the
planned joint path was the straight line to the goal. Only 12 of 4240 plans from the four
complete planners had a path ratio above 1.03 or an end-effector ratio above 1.1, and all 12
came from a single start/goal pair. This was a top-down grasp at x = 0.50 m, y = −0.065 m,
yaw 90°, whose straight path passes about 1 cm from the crate.

On that pair, the sampling planners detoured in 4 of 5 repeats each:

| Planner | Max path ratio | Max end-effector ratio | Min clearance |
|---|---|---|---|
| RRTConnect | 1.39 | 1.45 | 9.2 mm |
| BiTRRT | 1.17 | 1.96 | 11.0 mm |
| RRT* | 1.55 | 4.42 | 1.8 mm |
| CHOMP | 1.002 | – | 10.9 mm |

CHOMP stayed on the straight path on that pair.

Steps 1 and 2 left RRTConnect, BiTRRT and CHOMP, tied on median path ratio (1.000). Step 4
selected **RRTConnect**, the fastest of the three. The difference is small in absolute terms:
per start/goal pair, BiTRRT took a median 6 ms longer (Wilcoxon p = 4.4·10⁻⁹) and CHOMP
502 ms longer (p = 1.5·10⁻³⁶). Both are negligible against a 30–45 s pick cycle. Success and
path ratio did not differ between RRTConnect and the other two (McNemar p = 1; Wilcoxon
p ≥ 0.3). RRTConnect was used for all further experiments.

## Figure captions

**planning_time.pdf** – Planning time per planner for the move from the capture pose to the
pre-grasp (left) and from the lift to above the place pose (right), over successful plans
(log scale). Each dot is one plan; boxes show the median and quartiles, whiskers 1.5 IQR.
Dashed line: the 2 s limit on the 95th-percentile planning time of the selection rule.

**path_ratio.pdf** – Joint path length relative to the straight-line joint distance to the
goal (1.0 = shortest joint path), over successful plans. Pink ends near, not exactly at, the
fixed joint goal (it tracks the pre-grasp pose), so its ratio is relative to a slightly
different goal and can fall below 1.

## Limitations (for the discussion)

- **One scan.** The benchmark used a single crate scan with only the top layer of trusses.
  On it, the tilted grasp generator chose a straight-down approach at every grid point, so
  all 212 pairs were shared by the top-down and tilted methods. The selection is therefore
  based on top-down approaches only. Tilted approaches near the walls are covered by the
  E3 scans.
- **Simulation only.** Planner selection used simulation; no separate real-robot planner
  comparison was run. The chosen planner was used on the real robot in E4, and its success
  and planning time there serve as hardware validation. A planner only changes the planned
  geometric path: execution (time parameterisation, controller, speed scaling) is the same
  for every planner.
- **Path ratio as a selection criterion.** Earlier real-robot logs showed path ratios up to
  2.2. All of them predate a fix to the crate-contents filter (fingers in the camera cloud
  became obstacle columns at the gripper). After the fix, real plans also had ratio 1.000.
- **Timing hardware.** Planning times were measured on one machine (Intel i7-12700H) with
  one benchmark at a time.
