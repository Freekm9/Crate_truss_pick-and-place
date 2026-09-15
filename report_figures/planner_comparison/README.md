# Planner comparison — thesis figures

`analyze_planners.py` reads the two logs from
[`experiments/planner_comparison_protocol.md`](../../experiments/planner_comparison_protocol.md)
and produces the figures/tables in this folder.

```
python3 analyze_planners.py                 # pooled analysis (default)
python3 analyze_planners.py --split-by-session   # sensitivity check, see caveat below
```

Requires `pandas` + `matplotlib` (no ROS needed — it only reads the two CSVs).

## Outputs

Each figure is saved as both `.png` (slides) and `.pdf` (vector, for LaTeX):

- **success_rates** — two panels: trial-level pick success (2026-08-21 session
  only — see "Data quirks" below) and movement-level planning+execution
  outcome (all sessions). These two tell *different* stories — see Findings.
- **planning_duration** — planning time per planner, log scale, with the 5s
  budget line marked.
- **path_length_and_waypoints** — path length and waypoint count, per planner.
- **execution_duration** — trajectory execution time per planner.
- **tradeoff_scatter** — planning time vs. path length, the classic
  speed/quality trade-off view.

`table_trial_success_rate.csv` and `table_movement_summary.csv` are the
underlying numbers, ready for `pandas.read_csv(...).to_latex()` or a direct
paste into a LaTeX table.

## Data quirks handled by the script

- **`RRTSTAR` vs `RRTstar`**: two rows on 2026-08-26 logged the planner name
  in caps (from a `--planners RRTSTAR` CLI typo, going by the timestamps).
  Merged into `RRTstar`.
- **`metrics.csv` schema drift**: partway through the 2026-08-26 session the
  logger started writing two video filenames (side-cam + gripper-cam) as
  extra comma-separated values, so some rows have 16 fields against a
  15-column header. `read_metrics_csv()` folds any extra trailing fields back
  into one `video_file` value.
- **`goal_reached` is not a success/failure signal** — it's only populated
  *after* `execution_success` is already `True`, so taken at face value it is
  ~100% for every planner and hides every failure. The movement-level outcome
  panel uses the `result` column (`success`/`failure`, combining planning and
  execution) instead.
- **`chomp` always logs `num_waypoints == 101`** — that's CHOMP's fixed
  trajectory discretization in this setup, not a meaningful "few waypoints,
  smooth path" result. Don't compare it to the sampling planners' waypoint
  counts at face value.
- A handful of rows have blank `planning_success`/`result` (3 rows) — these
  are trials where the movement call appears to have crashed/aborted before
  logging an outcome. They're dropped from the relevant aggregates rather
  than counted as failures.
- `test_pre_grasp` and `go_to_saved_pose` movement rows are excluded from the
  main comparison (matches `summarize_planner_experiment.py`'s scope of
  `grasp_pre_grasp` + `grasp_post_grasp` only) — `test_pre_grasp` looks like
  ad-hoc manual testing rather than a logged trial.

## Open question worth resolving before the results go in the thesis

The protocol doc explicitly warns: *"if [hand-eye calibration] is redone
between sessions, treat those sessions as separate datasets rather than
pooling them."* This analysis pools the 2026-08-20/21 session (RRTConnect,
BiTRRT, RRTstar) with the 2026-08-26 session (adds `chomp`, many more BiTRRT
trials). **Confirm whether calibration was redone between Aug 21 and Aug 26**
— if it was, re-run with `--split-by-session` and report the two sessions
separately, at least for `chomp` and the newer BiTRRT trials.

## Preliminary findings (pooled data, n as shown on each chart)

- **Trial-level pick success vs. movement-level outcome disagree**:
  BiTRRT has the *lowest* full-pick success rate (3/10, Aug-21 session) but
  the *highest* individual grasp-movement success rate of the three planners
  in that same window. This suggests BiTRRT's plan-and-execute step for a
  single movement usually works, but something downstream in the pick
  (gripper close, approach geometry, tolerance stacking across the two
  chained movements) fails disproportionately often — worth digging into
  qualitatively via the saved videos rather than blaming the planner outright.
- **RRTstar and chomp both spend far more time planning** (RRTstar routinely
  hits the 5s budget; chomp ~0.8s) **than RRTConnect/BiTRRT** (~0.1s), without
  a clearly shorter resulting path — the trade-off scatter shows no planner
  dominating on both axes at once.
- Small sample sizes for RRTConnect (n=6-12) and the Aug-21 trial log overall
  make the error bars wide — treat any single-planner ranking claim as
  provisional pending more trials, especially for RRTConnect and chomp.
