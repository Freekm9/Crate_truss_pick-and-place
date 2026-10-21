"""
Planner comparison on path quality: figures and tables for the thesis.

Usage: python3 make_figures.py <manifest.csv> [--out DIR]
The manifest (written by experiments/planner_benchmark/run_planner_quality.sh) lists
set (typical / edge / return), variant (simplified / raw / sweep / pose_goal) and the results CSV of each
stage; the saved paths are read from the paths_*.jsonl.gz next to each CSV.

Writes, to --out (default: next to this script):
  paths_examples.pdf/.png      TCP paths of every planner and repeat for example queries,
                               top and side view with the crate
  all_paths_<set>_<move>       every TCP path per planner (and raw OMPL output), transparent
  metrics_<set>.pdf/.png       per-planner distributions of the path-quality metrics
  raw_vs_simplified.pdf/.png   OMPL planners with and without MoveIt's post-processing
  rrtstar_budget.pdf/.png      RRT* path quality against its time budget (one line per goal set)
  goal_type.pdf/.png           pose goal (MoveIt samples the IK) vs. the nearest-IK joint goal
  time_split.pdf/.png          planning time and execution time per planner, side by side
  table_quality.tex/.csv       median (95th percentile) per criterion and planner
  stats_pairwise.csv           paired tests between the candidates (Wilcoxon, McNemar)
  flagged_paths.csv            erratic paths to inspect (TCP ratio > 1.5 or deviation > 10 cm)
  numbers.txt                  every number quoted in the text

Everything is over unique plans (plan_key x planner x repeat); Pink is a controller and
is left out of the main figures (--with-pink adds it).
"""
import argparse
import gzip
import itertools
import json
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.ticker
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', '..', 'src', 'learning_approach_to_robotic_grasping_of_vine_tomatoes',
                                'grasp', 'src', 'move_robot'))
import path_quality as pq  # noqa: E402

# Same hue per planner as the E2 figures; the Pilz references are neutral, told apart by line style
STYLE = {
    'RRTConnect': dict(label='RRT-Connect', color='#2a78d6', ls='-'),
    'BiTRRT': dict(label='BiTRRT', color='#eb6834', ls='-'),
    'RRTstar@1': dict(label='RRT* (1 s)', color='#6250d6', ls='-'),
    'RRTstar@5': dict(label='RRT* (5 s)', color='#1baf7a', ls='-'),
    'chomp': dict(label='CHOMP', color='#eda100', ls='-'),
    'pilz_ptp': dict(label='Joint line (PTP)', color='#6b6b6b', ls='-'),
    'pilz_lin': dict(label='Cartesian line (LIN)', color='#9a9a96', ls='--'),
    'pink': dict(label='Pink', color='#e87ba4', ls='-'),
}
MAIN = ['RRTConnect', 'BiTRRT', 'RRTstar@1', 'RRTstar@5', 'chomp', 'pilz_ptp', 'pilz_lin']
MOVES = {'pre_grasp': 'Capture pose to pre-grasp', 'transport': 'Lift to place',
         'return_lift': 'Lift to capture pose', 'return_lift_rot': 'Lift (rolled 180°) to capture pose',
         'return_place': 'Place to capture pose', 'return_place_rot': 'Place (rolled 180°) to capture pose'}
SETS = {'typical': 'typical goals', 'edge': 'wall-band and tilted goals', 'return': 'moves back to the capture pose'}
SET_COLORS = {'typical': '#1baf7a', 'edge': '#6250d6', 'return': '#eb6834'}
INK, MUTED, GRID = '#2b2b2b', '#6b6b6b', '#e4e4e0'
# Crate interior (samla_crate.scene) for the path plots: x, y range and rim height
CRATE = dict(x=(0.335, 0.585), y=(-0.08, 0.28), z=(0.005, 0.14))

# Criteria of the quality table: column, label, scale, lower is better
CRITERIA = [
    ('success', 'Success (\\%)', 100, False),
    ('planning_time_s', 'Planning time (s)', 1, True),
    ('exec_time_s', 'Execution time (s)', 1, True),
    ('path_ratio', 'Joint path ratio', 1, True),
    ('ee_ratio', 'TCP path ratio', 1, True),
    ('ee_max_dev_m', 'TCP deviation (mm)', 1000, True),
    ('ee_turn_rad', 'TCP turning (rad)', 1, True),
    ('joint_waste_rad', 'Wasted joint motion (rad)', 1, True),
    ('jerk_rms', 'RMS jerk (rad/s$^3$)', 1, True),
    ('min_clearance_mm', 'Wall clearance (mm)', 1, False),
    ('min_contents_clearance_mm', 'Truss clearance (mm)', 1, False),
    ('spread_mm', 'Repeat spread (mm)', 1, True),
]

plt.rcParams.update({'font.size': 9, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK, 'xtick.color': INK,
                     'ytick.color': INK, 'axes.spines.top': False, 'axes.spines.right': False})


def load(manifest):
    m = pd.read_csv(manifest)
    frames, paths = [], []
    for _, r in m.iterrows():
        d = pd.read_csv(r['results'])
        d['set'], d['variant'], d['run'] = r['set'], r['variant'], os.path.basename(r['results'])
        frames.append(d)
        p = r['results'].replace('results_', 'paths_').replace('.csv', '.jsonl.gz')
        if os.path.exists(p):
            try:  # a run still in progress ends in a truncated record
                with gzip.open(p, 'rt') as f:
                    for line in f:
                        rec = json.loads(line)
                        rec.update(set=r['set'], variant=r['variant'])
                        paths.append(rec)
            except (EOFError, ValueError):
                pass
    d = pd.concat(frames, ignore_index=True)
    d['success'] = d['success'].astype(str).str.lower() == 'true'
    # Unique plans: a start/goal pair shared by several goal sets is planned once
    u = d.drop_duplicates(['set', 'variant', 'plan_key', 'planner', 'repeat']).copy()
    # The pipeline's RRT* is RRTstar at 5 s; a plain 'RRTstar' run counts as RRTstar@5
    u.loc[u['planner'] == 'RRTstar', 'planner'] = 'RRTstar@5'
    for rec in paths:
        if rec['planner'] == 'RRTstar':
            rec['planner'] = 'RRTstar@5'
    spread = {}
    by_query = {}
    for rec in paths:
        by_query.setdefault((rec['set'], rec['variant'], rec['plan_key'], rec['planner']), []).append(rec)
    for key, rs in by_query.items():
        if len(rs) > 1:
            spread[key] = pq.path_spread([r['tcp'] for r in rs]) * 1000
    u['spread_mm'] = [spread.get((s, v, k, p), np.nan) for s, v, k, p in
                      zip(u['set'], u['variant'], u['plan_key'], u['planner'])]
    return u, by_query


def save(fig, out, name):
    fig.tight_layout()
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(out, f'{name}.{ext}'), dpi=200)
    plt.close(fig)


def legend_handles(planners):
    return [plt.Line2D([], [], color=STYLE[p]['color'], ls=STYLE[p]['ls'], lw=2, label=STYLE[p]['label'])
            for p in planners]


def fig_paths(u, by_query, out, planners, n_lines):
    """Example queries: the edge query where RRT-Connect's repeats spread most, a typical
    median query and a transport; top view (x-y) and side view (x-z)."""
    s = u[(u['variant'] == 'simplified') & u['success']]
    picks = []
    rc = s[(s['planner'] == 'RRTConnect') & (s['move'] == 'pre_grasp')]
    for set_name, how in (('edge', 'max'), ('typical', 'median')):
        g = rc[rc['set'] == set_name].groupby('plan_key')['spread_mm'].first().dropna()
        if len(g):
            key = g.idxmax() if how == 'max' else (g - g.median()).abs().idxmin()
            picks.append((set_name, key, f"{SETS[set_name].capitalize()}: "
                          f"{'largest' if how == 'max' else 'median'} RRT-Connect spread"))
    tr = s[(s['move'] == 'transport') & (s['planner'] == 'RRTConnect')]
    if len(tr):
        key = tr.groupby('plan_key')['ee_ratio'].median().sort_values().index[len(tr['plan_key'].unique()) // 2]
        picks.append((tr[tr['plan_key'] == key]['set'].iloc[0], key, 'Lift to place (median query)'))
    if not picks:
        return
    fig, axes = plt.subplots(2, len(picks), figsize=(2.6 * len(picks) + 0.4, 5.2), squeeze=False)
    for c, (set_name, key, title) in enumerate(picks):
        for r, (i, j, lab) in enumerate(((0, 1, 'y (m)'), (0, 2, 'z (m)'))):
            ax = axes[r, c]
            (x0, x1), (y0, y1) = CRATE['x'], CRATE['y' if j == 1 else 'z']
            ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec=MUTED, lw=1))
            for p in planners:
                for rec in by_query.get((set_name, 'simplified', key, p), [])[:n_lines]:
                    tcp = np.array(rec['tcp'])
                    ax.plot(tcp[:, i], tcp[:, j], color=STYLE[p]['color'], ls=STYLE[p]['ls'], lw=1.2, alpha=0.8)
                    ax.plot(*tcp[-1, [i, j]], 'o', color=INK, ms=3)
            ax.plot(*tcp[0, [i, j]], 's', color=INK, ms=4)
            ax.set_aspect('equal', adjustable='datalim')
            ax.set_xlabel('x (m)')
            ax.set_ylabel(lab)
            ax.grid(color=GRID, lw=0.6)
            if r == 0:
                ax.set_title(title, fontsize=8.5, color=INK)
    fig.legend(handles=legend_handles(planners), loc='lower center', ncol=min(len(planners), 4), frameon=False,
               fontsize=8)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(out, f'paths_examples.{ext}'), dpi=200)
    plt.close(fig)


def fig_all_paths(by_query, out, set_name, move, planners, alpha=0.06):
    """Every TCP path of every planner for one goal set and move, as thin transparent
    lines: one panel pair (top view x-y, side view x-z) per planner, same axes for all,
    so common routes show dark and single outliers light. The OMPL planners' raw output
    (no post-processing) gets its own panels when it was run."""
    panels = []
    for p in planners:
        for variant in ('simplified', 'raw'):
            recs = [r for (s, v, k, pl), rs in by_query.items() if s == set_name and v == variant and pl == p
                    and k.startswith(move + ':') for r in rs]
            if recs:
                panels.append((p, variant, recs))
    if not panels:
        return
    # Shared axes within the post-processed panels and within the raw panels (the raw
    # paths span ~1 m and would shrink everything else to a dot)
    lims = {}
    for variant in ('simplified', 'raw'):
        xyz = [np.array(r['tcp']) for _, v, rs in panels if v == variant for r in rs]
        if xyz:
            allxyz = np.concatenate(xyz)
            pad = 0.03
            lims[variant] = [(min(allxyz[:, i].min(), CRATE[c][0]) - pad, max(allxyz[:, i].max(), CRATE[c][1]) + pad)
                             for i, c in enumerate('xyz')]
    per_row = 2
    rows = int(np.ceil(len(panels) / per_row))
    fig, axes = plt.subplots(rows, 2 * per_row, figsize=(7.6, 2.0 * rows + 0.3), squeeze=False)
    for idx, (p, variant, recs) in enumerate(panels):
        r, c0 = divmod(idx, per_row)
        for k, (i, j) in enumerate(((0, 1), (0, 2))):
            ax = axes[r, 2 * c0 + k]
            segs = [np.array(rec['tcp'])[:, [i, j]] for rec in recs]
            ax.add_collection(LineCollection(segs, colors=STYLE[p]['color'], linewidths=0.6, alpha=alpha))
            (x0, x1), (y0, y1) = CRATE['x'], CRATE['y' if j == 1 else 'z']
            ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec=INK, lw=0.8))
            lim = lims[variant]
            ax.set_xlim(lim[0])
            ax.set_ylim(lim[j])
            ax.set_aspect('equal')
            ax.tick_params(labelsize=6.5)
            ax.set_xlabel('x (m)', fontsize=7)
            ax.set_ylabel('y (m)' if j == 1 else 'z (m)', fontsize=7)
            if k == 0:
                name = STYLE[p]['label'] + (' raw, wider axes' if variant == 'raw' else '')
                ax.set_title(f"{name}: {len(recs)} paths, top / side", fontsize=7.5, color=INK, loc='left')
    for idx in range(len(panels), rows * per_row):
        r, c0 = divmod(idx, per_row)
        axes[r, 2 * c0].axis('off')
        axes[r, 2 * c0 + 1].axis('off')
    save(fig, out, f'all_paths_{set_name}_{move}')


def strip_box(ax, data, planners, col, scale, rng):
    for i, p in enumerate(planners):
        v = data.loc[data['planner'] == p, col].dropna().values * scale
        if not len(v):
            continue
        ax.boxplot(v, positions=[i], widths=0.5, showfliers=False, medianprops=dict(color=INK, lw=1.5),
                   boxprops=dict(color=MUTED), whiskerprops=dict(color=MUTED), capprops=dict(color=MUTED))
        ax.scatter(i + rng.uniform(-0.18, 0.18, len(v)), v, s=5, color=STYLE[p]['color'], alpha=0.35, lw=0,
                   zorder=3)
    ax.set_xticks(range(len(planners)))
    ax.set_xticklabels([STYLE[p]['label'] for p in planners], rotation=40, ha='right', fontsize=7.5)
    ax.grid(axis='y', color=GRID, lw=0.6)


def fig_metrics(u, out, set_name, planners):
    s = u[(u['set'] == set_name) & (u['variant'] == 'simplified') & u['success']]
    if s.empty:
        return
    panels = [('ee_ratio', 'TCP path ratio', 1), ('ee_max_dev_m', 'TCP deviation (mm)', 1000),
              ('exec_time_s', 'Execution time (s)', 1), ('ee_turn_rad', 'TCP turning (rad)', 1),
              ('joint_waste_rad', 'Wasted joint motion (rad)', 1), ('spread_mm', 'Repeat spread (mm)', 1)]
    moves = [m for m in MOVES if (s['move'] == m).any()]
    fig, axes = plt.subplots(len(moves), len(panels), figsize=(1.75 * len(panels), 2.6 * len(moves)), squeeze=False)
    rng = np.random.default_rng(0)
    for r, move in enumerate(moves):
        for c, (col, lab, scale) in enumerate(panels):
            strip_box(axes[r, c], s[s['move'] == move], planners, col, scale, rng)
            axes[r, c].set_title(lab if r == 0 else '', fontsize=8.5, color=INK)
        axes[r, 0].set_ylabel(MOVES[move])
    save(fig, out, f'metrics_{set_name}')


def fig_raw(u, out):
    ompl = ['RRTConnect', 'BiTRRT', 'RRTstar@1', 'RRTstar@5']
    s = u[u['success'] & u['variant'].isin(['simplified', 'raw']) & u['planner'].isin(ompl) &
          (u['move'] == 'pre_grasp')]
    if s['variant'].nunique() < 2:
        return
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.0))
    for ax, (col, lab) in zip(axes, (('path_ratio', 'Joint path ratio'), ('ee_ratio', 'TCP path ratio'))):
        for i, p in enumerate(ompl):
            for k, variant in enumerate(('raw', 'simplified')):
                v = s[(s['planner'] == p) & (s['variant'] == variant)][col].dropna().values
                if not len(v):
                    continue
                pos = i + (k - 0.5) * 0.36
                ax.boxplot(v, positions=[pos], widths=0.3, showfliers=False, medianprops=dict(color=INK, lw=1.5),
                           boxprops=dict(color=STYLE[p]['color'] if variant == 'raw' else MUTED),
                           whiskerprops=dict(color=MUTED), capprops=dict(color=MUTED))
        ax.set_yscale('log')
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f'{v:g}'))
        ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.set_yticks([1, 1.5, 2, 3, 5, 10, 20])
        ax.axhline(1.0, color=MUTED, lw=0.8)
        ax.set_xticks(range(len(ompl)))
        ax.set_xticklabels([STYLE[p]['label'] for p in ompl], fontsize=8)
        ax.set_title(f'{lab}: raw (left) vs. post-processed (right)', fontsize=8.5, color=INK)
        ax.grid(axis='y', color=GRID, lw=0.6)
    save(fig, out, 'raw_vs_simplified')


def fig_budget(u, out):
    """RRT* against its time budget, on the pairs of each set's sweep (with the same pairs'
    1 s and 5 s plans of the main stage): one colour per set, solid median, dashed p95."""
    s = u[u['planner'].str.startswith('RRTstar@') & (u['move'] == 'pre_grasp')].copy()
    sets = [x for x in SETS if ((s['set'] == x) & (s['variant'] == 'sweep')).any()]
    if not sets:
        return
    s['budget'] = s['planner'].str.split('@').str[1].astype(float)
    fig, axes = plt.subplots(1, 3, figsize=(7.6, 2.6))
    for set_name in sets:
        ss = s[s['set'] == set_name]
        keys = set(ss[ss['variant'] == 'sweep']['plan_key'])
        ss = ss[ss['plan_key'].isin(keys) & (ss['variant'].isin(['sweep', 'simplified']))]
        color = SET_COLORS[set_name]
        succ = ss.groupby('budget')['success'].mean() * 100
        axes[0].plot(succ.index, succ.values, 'o-', color=color, lw=2, ms=5, label=SETS[set_name])
        for ax, col in zip(axes[1:], ('path_ratio', 'ee_ratio')):
            ok = ss[ss['success']].groupby('budget')[col]
            ax.plot(ok.median().index, ok.median().values, 'o-', color=color, lw=2, ms=5)
            ax.plot(ok.quantile(0.95).index, ok.quantile(0.95).values, 'o--', color=color, lw=1.2, ms=3, alpha=0.6)
    axes[0].set_ylabel('Success (%)')
    axes[0].legend(frameon=False, fontsize=7)
    axes[1].set_ylabel('Joint path ratio')
    axes[2].set_ylabel('TCP path ratio')
    axes[2].plot([], [], 'k-', label='median')
    axes[2].plot([], [], 'k--', alpha=0.6, label='p95')
    axes[2].legend(frameon=False, fontsize=7)
    for ax in axes:
        ax.set_xscale('log')
        ax.set_xlabel('RRT* time budget (s)')
        ax.grid(color=GRID, lw=0.6)
    save(fig, out, 'rrtstar_budget')


def fig_goal_type(u, out, planners, numbers):
    """Pose goal (variant pose_goal: MoveIt samples the IK goal, as plan_movement.py did
    before the nearest-IK goal) against the nearest-IK joint goal (simplified) on the same
    start/goal pairs: per move a row, per planner a pair of boxes (left nearest IK, right
    pose goal) of the joint path length, its ratio to the nearest-IK distance and the
    distance of the end configuration from the nearest-IK goal."""
    pose = u[u['variant'] == 'pose_goal']
    if pose.empty:
        return
    rows = []
    for set_name in SETS:
        for move in MOVES:
            pk = pose[(pose['set'] == set_name) & (pose['move'] == move)]
            if not pk.empty:
                rows.append((set_name, move, set(pk['plan_key'])))
    ps = [p for p in planners if p in set(pose['planner'])]
    panels = [('success', 'Success (%)'), ('joint_path_length', 'Joint path (rad)'),
              ('path_ratio', 'Ratio to nearest-IK distance'), ('goal_deviation_rad', 'End vs. nearest IK (rad)')]
    fig, axes = plt.subplots(len(rows), len(panels), figsize=(1.9 * len(panels), 2.4 * len(rows)), squeeze=False)
    for r, (set_name, move, keys) in enumerate(rows):
        g = u[(u['set'] == set_name) & (u['move'] == move) & u['plan_key'].isin(keys) &
              u['variant'].isin(['simplified', 'pose_goal'])]
        for c, (col, lab) in enumerate(panels):
            ax = axes[r, c]
            for i, p in enumerate(ps):
                for k, variant in enumerate(('simplified', 'pose_goal')):
                    gv = g[(g['planner'] == p) & (g['variant'] == variant)]
                    x = i + (k - 0.5) * 0.36
                    if col == 'success':
                        if len(gv):
                            ax.bar(x, gv['success'].mean() * 100, width=0.32, color=STYLE[p]['color'],
                                   alpha=1.0 if k else 0.45)
                        continue
                    v = gv.loc[gv['success'], col].dropna().values
                    if len(v):
                        ax.boxplot(v, positions=[x], widths=0.3, showfliers=True,
                                   flierprops=dict(marker='.', ms=3, mec=STYLE[p]['color']),
                                   medianprops=dict(color=INK, lw=1.5),
                                   boxprops=dict(color=STYLE[p]['color'] if k else MUTED),
                                   whiskerprops=dict(color=MUTED), capprops=dict(color=MUTED))
            ax.set_xticks(range(len(ps)))
            ax.set_xticklabels([STYLE[p]['label'] for p in ps], rotation=40, ha='right', fontsize=7)
            ax.grid(axis='y', color=GRID, lw=0.6)
            if r == 0:
                ax.set_title(lab, fontsize=8.5, color=INK)
        axes[r, 0].set_ylabel(f"{MOVES[move]}\n({SETS[set_name]})", fontsize=7.5)
        for p in ps:
            line = []
            for variant in ('simplified', 'pose_goal'):
                gv = g[(g['planner'] == p) & (g['variant'] == variant)]
                ok = gv[gv['success']]
                line.append(f"{'nearest-IK' if variant == 'simplified' else 'pose'}: success {gv['success'].mean() * 100:.1f}%, "
                            f"joint path median {ok['joint_path_length'].median():.2f} / p95 "
                            f"{ok['joint_path_length'].quantile(0.95):.2f} rad, "
                            f"> 3 rad {(ok['joint_path_length'] > 3).mean() * 100:.1f}%, "
                            f"end off nearest IK > 0.1 rad {(ok['goal_deviation_rad'] > 0.1).mean() * 100:.1f}%")
            numbers.append(f"goal type {set_name} {move} {p}: " + ' | '.join(line))
    fig.text(0.01, 0.005, 'Left (pale) box: nearest-IK joint goal; right box: pose goal', fontsize=7, color=MUTED)
    save(fig, out, 'goal_type')


def fig_time_split(u, out, planners):
    s = u[(u['variant'] == 'simplified') & u['success']]
    if s.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 2.8), sharey=True)
    rng = np.random.default_rng(1)
    for ax, (col, lab) in zip(axes, (('planning_time_s', 'Planning time (s, log)'),
                                     ('exec_time_s', 'Execution time (s, log)'))):
        for i, p in enumerate(planners):
            v = s.loc[s['planner'] == p, col].dropna().values
            if len(v):
                ax.scatter(v, i + rng.uniform(-0.2, 0.2, len(v)), s=5, color=STYLE[p]['color'], alpha=0.35, lw=0)
                ax.plot([np.median(v)] * 2, [i - 0.3, i + 0.3], color=INK, lw=2)
        ax.set_xscale('log')
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f'{v:g}'))
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        lo, hi = ax.get_xlim()
        ax.set_xticks([t for t in (0.003, 0.01, 0.03, 0.1, 0.3, 1, 2, 3, 5, 10, 20, 30) if lo <= t <= hi])
        ax.set_xlabel(lab)
        ax.grid(axis='x', color=GRID, lw=0.6)
    axes[0].set_yticks(range(len(planners)))
    axes[0].set_yticklabels([STYLE[p]['label'] for p in planners])
    axes[0].invert_yaxis()
    save(fig, out, 'time_split')


def table(u, out, planners, numbers):
    s = u[u['variant'] == 'simplified']
    rows = []
    for set_name in SETS:
        for move in MOVES:
            g = s[(s['set'] == set_name) & (s['move'] == move)]
            if g.empty:
                continue
            for col, lab, scale, lower in CRITERIA:
                row = dict(set=set_name, move=move, criterion=lab)
                for p in planners:
                    gp = g[g['planner'] == p]
                    if col == 'success':
                        row[p] = (gp['success'].mean() * scale, np.nan) if len(gp) else (np.nan, np.nan)
                    else:
                        v = gp.loc[gp['success'], col].dropna() * scale
                        q = 0.05 if not lower else 0.95  # the bad tail: high for costs, low for clearances
                        row[p] = (v.median(), v.quantile(q)) if len(v) else (np.nan, np.nan)
                rows.append(row)
    t = pd.DataFrame(rows)
    flat = t.copy()
    for p in planners:
        flat[p] = flat[p].apply(lambda x: x[0])
    flat.to_csv(os.path.join(out, 'table_quality.csv'), index=False)

    def fmt(v):
        return '--' if not np.isfinite(v) else f'{v:.0f}' if abs(v) >= 100 else f'{v:.2f}' if abs(v) < 10 else f'{v:.1f}'

    lines = [r'\begin{table*}[t]', r'\centering', r'\scriptsize',
             r'\caption{Path quality per planner (post-processed paths, as in the pipeline). Median, with the '
             r'unfavourable tail in brackets: the 95th percentile for costs, the 5th percentile for clearances. '
             r'Bold: best candidate per criterion; the Pilz lines are references, not candidates.}',
             r'\label{tab:planner_quality}', r'\setlength{\tabcolsep}{3pt}',
             r'\begin{tabular}{l' + 'r' * len(planners) + '}', r'\toprule',
             'Criterion & ' + ' & '.join(STYLE[p]['label'] for p in planners) + r' \\']
    candidates = [p for p in planners if not p.startswith('pilz')]
    for (set_name, move), g in t.groupby(['set', 'move'], sort=False):
        lines += [r'\midrule', rf'\multicolumn{{{len(planners) + 1}}}{{l}}{{\emph{{{MOVES[move]}, {SETS[set_name]}}}}} \\']
        for _, r in g.iterrows():
            lower = next(c[3] for c in CRITERIA if c[1] == r['criterion'])
            vals = {p: r[p][0] for p in candidates if np.isfinite(r[p][0])}
            best = (min if lower else max)(vals.values()) if vals else None
            cells = []
            for p in planners:
                med, tail = r[p]
                c = fmt(med) + ('' if not np.isfinite(tail) else f' [{fmt(tail)}]')
                cells.append(rf'\textbf{{{c}}}' if p in candidates and best is not None and np.isclose(med, best)
                             else c)
            lines.append(r['criterion'] + ' & ' + ' & '.join(cells) + r' \\')
            numbers.append(f"{set_name} {move} {r['criterion']}: " +
                           ', '.join(f"{p} {fmt(r[p][0])} [{fmt(r[p][1])}]" for p in planners))
    lines += [r'\bottomrule', r'\end{tabular}', r'\end{table*}']
    with open(os.path.join(out, 'table_quality.tex'), 'w') as f:
        f.write('\n'.join(lines) + '\n')


def stats(u, out, planners, numbers):
    """Paired tests between the candidate planners on the same queries (post-processed):
    Wilcoxon signed-rank on per-query medians for every criterion, McNemar (exact binomial
    on the discordant plans) for success. Writes stats_pairwise.csv."""
    from scipy.stats import binomtest, wilcoxon
    s = u[u['variant'] == 'simplified']
    candidates = [p for p in planners if not p.startswith('pilz') and p != 'pink']
    rows = []
    for (set_name, move), g in s.groupby(['set', 'move']):
        for a, b in itertools.combinations(candidates, 2):
            ga = g[g['planner'] == a].set_index(['plan_key', 'repeat'])
            gb = g[g['planner'] == b].set_index(['plan_key', 'repeat'])
            both = ga[['success']].join(gb[['success']], rsuffix='_b', how='inner')
            only_a = int((both['success'] & ~both['success_b']).sum())
            only_b = int((~both['success'] & both['success_b']).sum())
            rows.append(dict(set=set_name, move=move, a=a, b=b, criterion='success', n=len(both),
                             diff_median=(only_a - only_b) / max(len(both), 1),
                             p=binomtest(only_a, only_a + only_b).pvalue if only_a + only_b else 1.0))
            for col, lab, scale, _ in CRITERIA[1:]:
                ma = g[(g['planner'] == a) & g['success']].groupby('plan_key')[col].median()
                mb = g[(g['planner'] == b) & g['success']].groupby('plan_key')[col].median()
                pair = pd.concat([ma, mb], axis=1, keys=['a', 'b']).dropna()
                diff = (pair['b'] - pair['a']) * scale
                p = wilcoxon(diff).pvalue if len(diff) > 5 and (diff.abs() > 1e-9).any() else 1.0
                rows.append(dict(set=set_name, move=move, a=a, b=b, criterion=lab, n=len(pair),
                                 diff_median=diff.median() if len(diff) else np.nan, p=p))
    t = pd.DataFrame(rows)
    t.to_csv(os.path.join(out, 'stats_pairwise.csv'), index=False)
    sig = t[t['p'] < 0.05]
    numbers.append(f"pairwise tests (b - a, per-query medians): {len(sig)} of {len(t)} with p < 0.05; "
                   f"see stats_pairwise.csv")


def flagged(u, out, numbers):
    """Erratic paths for inspection: TCP path ratio > 1.5 or TCP deviation > 10 cm."""
    f = u[u['success'] & ((u['ee_ratio'] > 1.5) | (u['ee_max_dev_m'] > 0.10))]
    cols = ['set', 'variant', 'move', 'method', 'planner', 'plan_key', 'repeat', 'x', 'y', 'z', 'yaw_deg',
            'tilt_x_deg', 'tilt_y_deg', 'wall_dist_m', 'path_ratio', 'ee_ratio', 'ee_max_dev_m', 'spread_mm']
    f[[c for c in cols if c in f]].sort_values(['set', 'variant', 'planner']).to_csv(
        os.path.join(out, 'flagged_paths.csv'), index=False)
    n = u[u['success']].groupby(['set', 'variant', 'move', 'planner']).size()
    k = f.groupby(['set', 'variant', 'move', 'planner']).size()
    numbers.append("erratic paths (TCP ratio > 1.5 or deviation > 10 cm) / successful plans:\n" +
                   pd.concat([k, n], axis=1, keys=['flagged', 'plans']).fillna(0).astype(int).to_string())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('manifest')
    ap.add_argument('--out', default=HERE)
    ap.add_argument('--with-pink', action='store_true')
    ap.add_argument('--lines', type=int, default=5, help="Repeats drawn per planner in the path plots")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    u, by_query = load(args.manifest)
    planners = [p for p in MAIN + (['pink'] if args.with_pink else []) if p in set(u['planner'])]
    numbers = [f"unique plans per set/variant/move: "
               f"{u.groupby(['set', 'variant', 'move']).size().to_dict()}",
               f"unique start/goal pairs: {u.groupby(['set', 'move'])['plan_key'].nunique().to_dict()}"]
    fig_paths(u, by_query, args.out, planners, args.lines)
    for set_name in SETS:
        for move in MOVES:
            fig_all_paths(by_query, args.out, set_name, move, planners + (['pink'] if 'pink' not in planners and
                                                                          args.with_pink else []))
    for set_name in SETS:
        fig_metrics(u, args.out, set_name, planners)
    fig_raw(u, args.out)
    fig_budget(u, args.out)
    fig_goal_type(u, args.out, planners, numbers)
    fig_time_split(u, args.out, planners)
    table(u, args.out, planners, numbers)
    stats(u, args.out, planners, numbers)
    flagged(u, args.out, numbers)
    raw = u[(u['variant'] == 'raw') & u['success']].groupby(['set', 'move', 'planner'])[['path_ratio', 'ee_ratio']]
    numbers.append("raw (no simplification, 1 attempt) median path / ee ratio:\n" + raw.median().round(3).to_string())
    with open(os.path.join(args.out, 'numbers.txt'), 'w') as f:
        f.write('\n'.join(numbers) + '\n')
    print('\n'.join(numbers))


if __name__ == '__main__':
    main()
