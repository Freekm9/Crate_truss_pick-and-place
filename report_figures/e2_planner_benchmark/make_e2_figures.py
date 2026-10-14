"""
E2 stage A (planning-only planner benchmark): figures and tables for the report.

Reads the full-grid results of scan 20261006_112906 (see
experiments/planner_benchmark/README.md) and writes, next to this script:
  planning_time.pdf/.png   planning time per planner, pre-grasp and transport (log scale)
  path_ratio.pdf/.png      joint path length / direct joint distance per planner
  table_e2.tex             booktabs table (success with Wilson 95% CI, time, path ratio, clearance)
  table_e2.csv             the same numbers, unrounded
  numbers.txt              every number quoted in the results text

Rows of the results CSVs are per query; topdown and tilted queries that share a start/goal
pair (plan_key) are planned once, so everything here is over unique plans
(plan_key x planner x repeat).
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import binomtest, wilcoxon

HERE = os.path.dirname(os.path.abspath(__file__))
RUN = os.path.join(HERE, '..', '..', 'experiments', 'planner_benchmark', '20261013_134011')
RESULTS = ['results_20261013_134015_complete.csv', 'results_20261014_101318.csv']

PLANNERS = ['RRTConnect', 'BiTRRT', 'RRTstar', 'chomp', 'pink']
LABELS = {'RRTConnect': 'RRTConnect', 'BiTRRT': 'BiTRRT', 'RRTstar': 'RRT*', 'chomp': 'CHOMP', 'pink': 'Pink'}
COLORS = {'RRTConnect': '#2a78d6', 'BiTRRT': '#eb6834', 'RRTstar': '#1baf7a', 'chomp': '#eda100', 'pink': '#e87ba4'}
MOVES = {'pre_grasp': 'Capture pose to pre-grasp', 'transport': 'Lift to place'}
MAX_P95_S = 2.0  # selection rule step 2
INK, MUTED, GRID = '#2b2b2b', '#6b6b6b', '#e4e4e0'

plt.rcParams.update({'font.size': 9, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK, 'xtick.color': INK,
                     'ytick.color': INK, 'axes.spines.top': False, 'axes.spines.right': False})


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z ** 2 / n
    c = (p + z ** 2 / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / d
    return c - h, c + h


def load():
    r = pd.concat([pd.read_csv(os.path.join(RUN, f)) for f in RESULTS], ignore_index=True)
    r['success'] = r['success'].astype(str).str.lower().isin(['true', '1'])
    return r, r.drop_duplicates(['plan_key', 'planner', 'repeat'])


def strip(ax, data, value, ylabel, log=False):
    rng = np.random.default_rng(0)
    for i, p in enumerate(PLANNERS):
        v = data.loc[data.planner == p, value].dropna().to_numpy()
        ax.boxplot(v, positions=[i], widths=0.5, showfliers=False, medianprops=dict(color=INK, lw=1.5),
                   boxprops=dict(color=MUTED), whiskerprops=dict(color=MUTED), capprops=dict(color=MUTED))
        ax.scatter(i + rng.uniform(-0.18, 0.18, len(v)), v, s=6, color=COLORS[p], alpha=0.35, lw=0, zorder=3)
    ax.set_xticks(range(len(PLANNERS)), [LABELS[p] for p in PLANNERS], fontsize=8)
    ax.set_ylabel(ylabel)
    if log:
        ax.set_yscale('log')
    ax.grid(axis='y', color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def figures(r):
    ok = r[r.success]
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.0), sharey=True)
    for ax, (move, title) in zip(axes, MOVES.items()):
        strip(ax, ok[ok.move == move], 'planning_time_s', 'Planning time (s)' if move == 'pre_grasp' else '', log=True)
        ax.axhline(MAX_P95_S, color=MUTED, lw=1, ls='--')
        ax.text(len(PLANNERS) - 0.5, MAX_P95_S * 1.12, '2 s limit (p95)', color=MUTED, ha='right', fontsize=8)
        ax.set_title(title, fontsize=9, color=INK)
    fig.tight_layout()
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(HERE, f'planning_time.{ext}'), dpi=200)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.0), sharey=True)
    for ax, (move, title) in zip(axes, MOVES.items()):
        strip(ax, ok[ok.move == move], 'path_ratio', 'Joint path / direct joint distance' if move == 'pre_grasp' else '')
        ax.axhline(1.0, color=MUTED, lw=0.8)
        ax.set_title(title, fontsize=9, color=INK)
    fig.tight_layout()
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(HERE, f'path_ratio.{ext}'), dpi=200)
    plt.close(fig)


def table(r):
    rows = []
    for move in MOVES:
        for p in PLANNERS:
            d = r[(r.move == move) & (r.planner == p)]
            ok = d[d.success]
            k, n = int(d.success.sum()), len(d)
            lo, hi = wilson(k, n)
            rows.append(dict(move=move, planner=p, plans=n, pairs=d.plan_key.nunique(), successes=k,
                             success=k / n, ci_lo=lo, ci_hi=hi,
                             t_med_s=ok.planning_time_s.median(), t_p95_s=ok.planning_time_s.quantile(0.95),
                             ratio_med=ok.path_ratio.median(), ratio_p95=ok.path_ratio.quantile(0.95),
                             ratio_max=ok.path_ratio.max(), ee_ratio_max=ok.ee_ratio.max(),
                             clear_min_mm=ok.min_clearance_mm.min()))
    t = pd.DataFrame(rows)
    t.to_csv(os.path.join(HERE, 'table_e2.csv'), index=False)

    lines = [r'\begin{table}[htbp]', r'\centering',
             r'\caption{Planning-only planner benchmark (E2 stage A): 212 fixed start/goal pairs from one '
             r'crate scan, 5 repeats each, 9\,cm fingers. Success with Wilson 95\% CI; times and path ratios '
             r'over successful plans. Path ratio = joint path length / direct joint distance to the goal.}',
             r'\label{tab:e2_planner_benchmark}', r'\begin{tabular}{llrrrrr}', r'\toprule',
             r'Move & Planner & Success (\%) & $t_\mathrm{med}$ (s) & $t_{95}$ (s) & Ratio med & Ratio max \\',
             r'\midrule']
    for i, move in enumerate(MOVES):
        if i:
            lines.append(r'\midrule')
        for j, row in enumerate(t[t.move == move].itertuples()):
            name = MOVES[move] if j == 0 else ''
            succ = f'{100 * row.success:.1f} [{100 * row.ci_lo:.1f}, {100 * row.ci_hi:.1f}]'
            lines.append(f'{name} & {LABELS[row.planner]} & {succ} & {row.t_med_s:.3f} & {row.t_p95_s:.3f} & '
                         f'{row.ratio_med:.3f} & {row.ratio_max:.3f} \\\\')
    lines += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    with open(os.path.join(HERE, 'table_e2.tex'), 'w') as f:
        f.write('\n'.join(lines) + '\n')
    return t


def numbers(raw, r, t):
    out = []
    pairs = r.groupby('move').plan_key.nunique()
    out.append(f"unique start/goal pairs: {pairs.to_dict()}, repeats 5, plans per planner {len(r) // len(PLANNERS)}")
    out.append(f"pairs shared by topdown and tilted: {(raw.groupby('plan_key').method.nunique() == 2).sum()} "
               f"of {raw.plan_key.nunique()}; tilted rows with nonzero tilt: "
               f"{((raw.method == 'tilted') & ((raw.tilt_x_deg != 0) | (raw.tilt_y_deg != 0))).sum()}")
    for p in PLANNERS:
        d = r[r.planner == p]
        ok = d[d.success]
        out.append(f"{LABELS[p]}: success {d.success.mean():.4f} ({d.success.sum()}/{len(d)}), "
                   f"t_med {ok.planning_time_s.median():.3f} s, t_p95 {ok.planning_time_s.quantile(.95):.3f} s, "
                   f"ratio median {ok.path_ratio.median():.3f}")
    pg = r[(r.planner == 'pink') & (r.move == 'pre_grasp')]
    out.append(f"Pink pre-grasp failure reasons: {pg[~pg.success].pink_reason.value_counts().to_dict()}")

    ok = r[r.success & (r.planner != 'pink')]
    outl = ok[(ok.path_ratio > 1.03) | (ok.ee_ratio > 1.1)]
    out.append(f"outliers (ratio > 1.03 or ee ratio > 1.1), sampling + CHOMP: {len(outl)} of {len(ok)} plans, "
               f"pairs {sorted(outl.plan_key.unique())}, per planner {outl.planner.value_counts().to_dict()}")
    for p, g in outl.groupby('planner'):
        out.append(f"  {LABELS[p]}: {len(g)} plans, max ratio {g.path_ratio.max():.3f}, max ee ratio {g.ee_ratio.max():.3f}, "
                   f"min clearance {g.min_clearance_mm.min():.1f} mm, pairs {sorted(g.plan_key.unique())}")

    # Paired comparison with the chosen planner, per unique pair (mean over repeats)
    per = r[r.success].groupby(['plan_key', 'planner']).agg(t=('planning_time_s', 'median'),
                                                             ratio=('path_ratio', 'median')).unstack()
    for p in ['BiTRRT', 'chomp']:
        dt = (per['t'][p] - per['t']['RRTConnect']).dropna()
        out.append(f"planning time {LABELS[p]} - RRTConnect per pair: median {dt.median() * 1000:+.1f} ms, "
                   f"Wilcoxon p={wilcoxon(dt).pvalue:.2g} (n={len(dt)})")
    with open(os.path.join(HERE, 'numbers.txt'), 'w') as f:
        f.write('\n'.join(out) + '\n')
    print('\n'.join(out))


if __name__ == '__main__':
    raw, r = load()
    figures(r)
    t = table(r)
    numbers(raw, r, t)
    print(t.round(4).to_string(index=False))
