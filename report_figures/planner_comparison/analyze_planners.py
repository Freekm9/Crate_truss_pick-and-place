"""
Planner-comparison analysis for the thesis: loads the two logs produced by
run_planner_experiment.py (see experiments/planner_comparison_protocol.md),
cleans them up, prints summary tables, and saves a set of clean, consistent
figures (PNG for slides + PDF for LaTeX) to this directory.

Data quality notes handled here (see README.md in this folder for detail):
  - "RRTSTAR" / "RRTstar" casing is merged into one planner.
  - Partway through the 2026-08-26 session, metrics.csv started logging two
    video clips per row (side + gripper camera) as extra comma-separated
    values, so the file mixes 15-field and 16-field rows under one 15-column
    header. Loaded with Python's csv module and reconciled by hand below.
  - Main comparison uses only the two grasp-motion movements
    (grasp_pre_grasp, grasp_post_grasp) — same scope as
    summarize_planner_experiment.py — dropping go_to_saved_pose and the
    ad-hoc test_pre_grasp rows.
  - Rows with missing metrics (crashed/aborted calls) are dropped from the
    numeric aggregations automatically (pandas skipna).
  - The 2026-08-20/21 and 2026-08-26 sessions are pooled. The protocol notes
    this is only valid if hand-eye calibration was not redone between them —
    confirm that before trusting the pooled numbers; a --split-by-session
    flag is provided to check sensitivity to this assumption.

Usage:
    python3 analyze_planners.py [--split-by-session]
"""
import argparse
import csv
import os

import matplotlib
matplotlib.use("Agg")  # headless: never try to open a display window
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
EXPERIMENTS_DIR = os.path.normpath(os.path.join(HERE, "..", "..", "experiments"))
TRIAL_LOG = os.path.join(EXPERIMENTS_DIR, "planner_comparison", "trial_log.csv")
METRICS_LOG = os.path.join(EXPERIMENTS_DIR, "planning_metrics", "metrics.csv")

GRASP_MOVEMENTS = ["grasp_pre_grasp", "grasp_post_grasp"]

# Fixed categorical color order — one slot per planner, held constant across
# every figure so a planner's color never changes between charts.
# Hex values from the studio's validated categorical ramp (slots 1/2/3/7).
PLANNER_COLORS = {
    "RRTConnect": "#2a78d6",  # blue   — known-reliable baseline
    "BiTRRT":     "#eb6834",  # orange — project's current default
    "RRTstar":    "#1baf7a",  # aqua   — asymptotically-optimal option
    "chomp":      "#4a3aa7",  # violet — later addition
}
PLANNER_ORDER = ["RRTConnect", "BiTRRT", "RRTstar", "chomp"]

TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID_COLOR = "#c9c8c2"

plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "font.size": 11,
    "text.color": TEXT_PRIMARY,
    "axes.edgecolor": TEXT_SECONDARY,
    "axes.labelcolor": TEXT_PRIMARY,
    "axes.titlecolor": TEXT_PRIMARY,
    "xtick.color": TEXT_SECONDARY,
    "ytick.color": TEXT_SECONDARY,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": GRID_COLOR,
    "grid.alpha": 0.6,
    "grid.linewidth": 0.6,
    "axes.axisbelow": True,
    "font.family": "DejaVu Sans",
})


def wilson_ci(successes, n, z=1.96):
    """Wilson score interval for a binomial proportion — better-behaved than
    normal-approximation error bars for the small sample sizes here."""
    if n == 0:
        return 0.0, 0.0, 0.0
    p = successes / n
    denom = 1 + z ** 2 / n
    center = (p + z ** 2 / (2 * n)) / denom
    half = (z * np.sqrt((p * (1 - p) + z ** 2 / (4 * n)) / n)) / denom
    return p, max(0.0, center - half), min(1.0, center + half)


def read_metrics_csv(path):
    """metrics.csv has a fixed 15-column header, but partway through the
    2026-08-26 session the logger started appending a second video filename
    (gripper-cam clip) as an extra comma-separated value, so some rows have
    16 fields. Read with csv.reader and fold any extra trailing fields into
    video_file (joined with ';') so every row lines up with the header."""
    with open(path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        n_cols = len(header)
        rows = []
        for row in reader:
            if len(row) > n_cols:
                row = row[:n_cols - 1] + [";".join(row[n_cols - 1:])]
            rows.append(row)
    return pd.DataFrame(rows, columns=header)


def load_data():
    trials = pd.read_csv(TRIAL_LOG, parse_dates=["timestamp"])
    metrics = read_metrics_csv(METRICS_LOG)

    metrics["timestamp"] = pd.to_datetime(metrics["timestamp"])
    numeric_cols = ["planning_time_limit", "num_planning_attempts", "planning_duration_s",
                     "path_length", "num_waypoints", "execution_duration_s"]
    for col in numeric_cols:
        metrics[col] = pd.to_numeric(metrics[col], errors="coerce")
    bool_cols = ["is_retry", "planning_success", "execution_success", "goal_reached"]
    for col in bool_cols:
        metrics[col] = metrics[col].map({"True": True, "False": False, "": np.nan})

    for df in (trials, metrics):
        df["planner_id"] = df["planner_id"].replace({"RRTSTAR": "RRTstar"})

    return trials, metrics


def save_fig(fig, name):
    png_path = os.path.join(HERE, f"{name}.png")
    pdf_path = os.path.join(HERE, f"{name}.pdf")
    fig.savefig(png_path, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    print(f"  saved {png_path}")
    print(f"  saved {pdf_path}")


def style_axes(ax):
    ax.tick_params(length=0)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(GRID_COLOR)


def planners_present(df):
    return [p for p in PLANNER_ORDER if p in df["planner_id"].unique()]


# --------------------------------------------------------------------------
# Figure 1: success rates (trial-level pick success + movement-level goal-reached)
# --------------------------------------------------------------------------

def fig_success_rates(trials, grasp_moves):
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), sharey=True)

    # Panel A: trial-level pick success rate (only sessions with a trial log)
    ax = axes[0]
    planners = planners_present(trials)
    xs = np.arange(len(planners))
    for x, p in zip(xs, planners):
        sub = trials[trials["planner_id"] == p]
        n = len(sub)
        succ = (sub["service_result"] == "success").sum()
        rate, lo, hi = wilson_ci(succ, n)
        ax.bar(x, rate * 100, color=PLANNER_COLORS[p], width=0.6)
        ax.errorbar(x, rate * 100, yerr=[[max(0.0, rate * 100 - lo * 100)], [max(0.0, hi * 100 - rate * 100)]],
                     color=TEXT_PRIMARY, capsize=3, linewidth=1)
        ax.text(x, hi * 100 + 3, f"{succ}/{n}", ha="center", va="bottom",
                 fontsize=9, color=TEXT_SECONDARY)
    ax.set_xticks(xs)
    ax.set_xticklabels(planners)
    ax.set_ylabel("Success rate (%)")
    ax.set_title("Trial-level pick success\n(2026-08-21 session only)", fontsize=10.5)
    ax.set_ylim(0, 115)
    style_axes(ax)

    # Panel B: movement-level outcome rate (all sessions, grasp motions only).
    # Uses the `result` column (planning+execution combined), not
    # `goal_reached` — goal_reached is only ever recorded *after* execution
    # already succeeded, so it is ~100% by construction and hides failures.
    ax = axes[1]
    outcome_moves = grasp_moves[grasp_moves["result"].isin(["success", "failure"])]
    planners = planners_present(outcome_moves)
    xs = np.arange(len(planners))
    for x, p in zip(xs, planners):
        sub = outcome_moves[outcome_moves["planner_id"] == p]
        n = len(sub)
        succ = (sub["result"] == "success").sum()
        rate, lo, hi = wilson_ci(succ, n)
        ax.bar(x, rate * 100, color=PLANNER_COLORS[p], width=0.6)
        ax.errorbar(x, rate * 100, yerr=[[max(0.0, rate * 100 - lo * 100)], [max(0.0, hi * 100 - rate * 100)]],
                     color=TEXT_PRIMARY, capsize=3, linewidth=1)
        ax.text(x, hi * 100 + 3, f"{succ}/{n}", ha="center", va="bottom",
                 fontsize=9, color=TEXT_SECONDARY)
    ax.set_xticks(xs)
    ax.set_xticklabels(planners)
    ax.set_title("Movement-level outcome\n(all sessions, grasp motions)", fontsize=10.5)
    ax.set_ylim(0, 115)
    style_axes(ax)

    fig.suptitle("Success rate by planner", y=1.03, fontsize=13)
    fig.text(0.5, -0.04,
              "Error bars: 95% Wilson score interval. Labels: successes / n.",
              ha="center", fontsize=8.5, color=TEXT_SECONDARY)
    fig.tight_layout()
    save_fig(fig, "success_rates")
    plt.close(fig)


# --------------------------------------------------------------------------
# Figure 2: planning duration (log scale, since sampling-based planners
# finish in well under a second while RRTstar/chomp run close to the time cap)
# --------------------------------------------------------------------------

def fig_planning_duration(grasp_moves):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    planners = planners_present(grasp_moves)
    data = [grasp_moves.loc[grasp_moves["planner_id"] == p, "planning_duration_s"].dropna().values
            for p in planners]

    bp = ax.boxplot(data, positions=np.arange(len(planners)), widths=0.55,
                     patch_artist=True, showfliers=True,
                     flierprops=dict(marker="o", markersize=3, markerfacecolor=TEXT_SECONDARY,
                                      markeredgecolor="none", alpha=0.5),
                     medianprops=dict(color=TEXT_PRIMARY, linewidth=1.6),
                     whiskerprops=dict(color=TEXT_SECONDARY),
                     capprops=dict(color=TEXT_SECONDARY))
    for patch, p in zip(bp["boxes"], planners):
        patch.set_facecolor(PLANNER_COLORS[p])
        patch.set_alpha(0.65)
        patch.set_edgecolor(PLANNER_COLORS[p])

    time_limit = grasp_moves["planning_time_limit"].dropna().iloc[0] if grasp_moves["planning_time_limit"].notna().any() else None
    if time_limit:
        ax.axhline(time_limit, color=TEXT_SECONDARY, linestyle="--", linewidth=1)
        ax.text(len(planners) - 0.4, time_limit, f" {time_limit:g}s budget",
                 va="bottom", ha="right", fontsize=8.5, color=TEXT_SECONDARY)

    ax.set_yscale("log")
    ax.set_xticks(np.arange(len(planners)))
    ax.set_xticklabels(planners)
    ax.set_ylabel("Planning duration (s, log scale)")
    ax.set_title("Planning duration by planner\n(grasp_pre_grasp + grasp_post_grasp calls)")
    for x, p in zip(np.arange(len(planners)), planners):
        n = len(grasp_moves.loc[grasp_moves["planner_id"] == p, "planning_duration_s"].dropna())
        ax.text(x, ax.get_ylim()[0], f"n={n}", ha="center", va="bottom", fontsize=8, color=TEXT_SECONDARY)
    style_axes(ax)
    fig.tight_layout()
    save_fig(fig, "planning_duration")
    plt.close(fig)


# --------------------------------------------------------------------------
# Figure 3: path length + waypoint count (small multiples — different units,
# so two panels rather than a dual-axis chart)
# --------------------------------------------------------------------------

def fig_path_quality(grasp_moves):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    planners = planners_present(grasp_moves)

    for ax, col, ylabel, title in zip(
        axes,
        ["path_length", "num_waypoints"],
        ["Path length (rad, joint-space L2)", "Waypoint count"],
        ["Path length by planner", "Waypoint count by planner"],
    ):
        data = [grasp_moves.loc[grasp_moves["planner_id"] == p, col].dropna().values for p in planners]
        bp = ax.boxplot(data, positions=np.arange(len(planners)), widths=0.55,
                         patch_artist=True, showfliers=True,
                         flierprops=dict(marker="o", markersize=3, markerfacecolor=TEXT_SECONDARY,
                                          markeredgecolor="none", alpha=0.5),
                         medianprops=dict(color=TEXT_PRIMARY, linewidth=1.6),
                         whiskerprops=dict(color=TEXT_SECONDARY),
                         capprops=dict(color=TEXT_SECONDARY))
        for patch, p in zip(bp["boxes"], planners):
            patch.set_facecolor(PLANNER_COLORS[p])
            patch.set_alpha(0.65)
            patch.set_edgecolor(PLANNER_COLORS[p])
        ax.set_xticks(np.arange(len(planners)))
        ax.set_xticklabels(planners)
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=11)
        style_axes(ax)

    fig.text(0.5, -0.03,
              "chomp always reports exactly 101 waypoints (fixed trajectory discretization,\n"
              "not a planner behaving unusually) — its waypoint count is not comparable to the\n"
              "sampling-based planners' vertex counts.",
              ha="center", fontsize=8.5, color=TEXT_SECONDARY)
    fig.tight_layout()
    save_fig(fig, "path_length_and_waypoints")
    plt.close(fig)


# --------------------------------------------------------------------------
# Figure 4: execution duration
# --------------------------------------------------------------------------

def fig_execution_duration(grasp_moves):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    planners = planners_present(grasp_moves)
    data = [grasp_moves.loc[grasp_moves["planner_id"] == p, "execution_duration_s"].dropna().values
            for p in planners]
    bp = ax.boxplot(data, positions=np.arange(len(planners)), widths=0.55,
                     patch_artist=True, showfliers=True,
                     flierprops=dict(marker="o", markersize=3, markerfacecolor=TEXT_SECONDARY,
                                      markeredgecolor="none", alpha=0.5),
                     medianprops=dict(color=TEXT_PRIMARY, linewidth=1.6),
                     whiskerprops=dict(color=TEXT_SECONDARY),
                     capprops=dict(color=TEXT_SECONDARY))
    for patch, p in zip(bp["boxes"], planners):
        patch.set_facecolor(PLANNER_COLORS[p])
        patch.set_alpha(0.65)
        patch.set_edgecolor(PLANNER_COLORS[p])
    ax.set_xticks(np.arange(len(planners)))
    ax.set_xticklabels(planners)
    ax.set_ylabel("Execution duration (s)")
    ax.set_title("Trajectory execution duration by planner\n(grasp_pre_grasp + grasp_post_grasp calls)")
    style_axes(ax)
    fig.tight_layout()
    save_fig(fig, "execution_duration")
    plt.close(fig)


# --------------------------------------------------------------------------
# Figure 5: planning-time vs. path-length trade-off scatter
# --------------------------------------------------------------------------

def fig_tradeoff_scatter(grasp_moves):
    fig, ax = plt.subplots(figsize=(7, 5.5))
    planners = planners_present(grasp_moves)
    for p in planners:
        sub = grasp_moves[grasp_moves["planner_id"] == p].dropna(subset=["planning_duration_s", "path_length"])
        ax.scatter(sub["planning_duration_s"], sub["path_length"],
                    color=PLANNER_COLORS[p], label=p, alpha=0.65, s=26,
                    edgecolors="white", linewidths=0.4)
    ax.set_xscale("log")
    ax.set_xlabel("Planning duration (s, log scale)")
    ax.set_ylabel("Path length (rad, joint-space L2)")
    ax.set_title("Planning-time / path-quality trade-off\n(grasp_pre_grasp + grasp_post_grasp calls)")
    ax.legend(frameon=False, loc="upper left")
    style_axes(ax)
    fig.tight_layout()
    save_fig(fig, "tradeoff_scatter")
    plt.close(fig)


# --------------------------------------------------------------------------
# Summary tables (CSV, ready to paste into a LaTeX table via pandas.to_latex)
# --------------------------------------------------------------------------

def write_summary_tables(trials, grasp_moves):
    trial_summary = trials.groupby("planner_id").agg(
        n_trials=("service_result", "count"),
        n_success=("service_result", lambda s: (s == "success").sum()),
    )
    trial_summary["success_rate"] = (trial_summary["n_success"] / trial_summary["n_trials"]).round(3)
    trial_summary = trial_summary.reindex([p for p in PLANNER_ORDER if p in trial_summary.index])
    trial_summary.to_csv(os.path.join(HERE, "table_trial_success_rate.csv"))

    move_summary = grasp_moves.groupby(["planner_id", "movement"]).agg(
        n=("planning_success", "count"),
        planning_success_rate=("planning_success", "mean"),
        mean_planning_duration_s=("planning_duration_s", "mean"),
        median_planning_duration_s=("planning_duration_s", "median"),
        mean_path_length=("path_length", "mean"),
        mean_num_waypoints=("num_waypoints", "mean"),
        mean_execution_duration_s=("execution_duration_s", "mean"),
        goal_reached_rate=("goal_reached", "mean"),
    ).round(3)
    move_summary.to_csv(os.path.join(HERE, "table_movement_summary.csv"))

    print("\n=== Trial-level pick success rate, by planner (2026-08-21 session) ===")
    print(trial_summary.to_string())
    print("\n=== Per-movement planning/execution detail, by planner (all sessions) ===")
    print(move_summary.to_string())


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split-by-session", action="store_true",
                         help="Report the 2026-08-20/21 and 2026-08-26 sessions separately "
                              "instead of pooling them (sensitivity check for the calibration caveat).")
    args = parser.parse_args()

    trials, metrics = load_data()
    grasp_moves = metrics[metrics["movement"].isin(GRASP_MOVEMENTS)].copy()

    print(f"Trial log:   {TRIAL_LOG} ({len(trials)} trials)")
    print(f"Metrics log: {METRICS_LOG} ({len(metrics)} rows, {len(grasp_moves)} grasp-motion rows)\n")

    if args.split_by_session:
        metrics["session"] = np.where(metrics["timestamp"] < "2026-08-22", "2026-08-20/21", "2026-08-26")
        grasp_moves["session"] = np.where(grasp_moves["timestamp"] < "2026-08-22", "2026-08-20/21", "2026-08-26")
        for session, sub in grasp_moves.groupby("session"):
            print(f"\n--- session {session} ---")
            summary = sub.groupby(["planner_id", "movement"]).agg(
                n=("planning_success", "count"),
                planning_success_rate=("planning_success", "mean"),
                mean_planning_duration_s=("planning_duration_s", "mean"),
                mean_path_length=("path_length", "mean"),
                goal_reached_rate=("goal_reached", "mean"),
            ).round(3)
            print(summary.to_string())
        return

    write_summary_tables(trials, grasp_moves)

    print("\nGenerating figures...")
    fig_success_rates(trials, grasp_moves)
    fig_planning_duration(grasp_moves)
    fig_path_quality(grasp_moves)
    fig_execution_duration(grasp_moves)
    fig_tradeoff_scatter(grasp_moves)
    print("\nDone.")


if __name__ == "__main__":
    main()
