"""
E3 (offline reachability vs finger extension): figures and tables for the report.

Reads the per-pose check CSVs of the full sweep (9 whole-crate scans of 2026-10-14, see
experiments/finger_reachability/full_20261014) and writes, next to this script:
  reachability_vs_finger.pdf/.png  reachable fraction per finger extension, both generators,
                                   trusses as obstacle (gripper carve), cluster-bootstrap CI
  reachability_by_wall.pdf/.png    reachable fraction by distance to the nearest wall,
                                   at 0, 5 and 10 cm (the operating length)
  table_e3.tex / table_e3.csv      per finger extension and generator: walls only and with
                                   the trusses (both carve modes), McNemar on the paired poses
  numbers.txt                      every number quoted in the results text
"""

import glob
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import binomtest

HERE = os.path.dirname(os.path.abspath(__file__))
RUN = os.path.join(HERE, '..', '..', 'experiments', 'finger_reachability', 'full_20261014')
METHODS = {'topdown': 'Top-down (4-DOF)', 'tilted': 'Tilted (6/7-DOF)'}
COLORS = {'topdown': '#2a78d6', 'tilted': '#eb6834'}
MARKERS = {'topdown': 'o', 'tilted': 's'}
VARIANTS = {'walls': 'Walls only', 'cylinder': 'Trusses, corridor carve', 'gripper': 'Trusses, gripper carve'}
WALL_PANELS_CM = [0, 5, 10]
WALL_BINS_CM = [1, 2, 3, 4, 5, 7, 10, 100]  # no grasp point lies within 1 cm of a wall
WALL_LABELS = ['1-2', '2-3', '3-4', '4-5', '5-7', '7-10', '>10']
OPERATING_CM = 10
BOOTSTRAPS = 2000
INK, MUTED, GRID = '#2b2b2b', '#6b6b6b', '#e4e4e0'

plt.rcParams.update({'font.size': 9, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK, 'xtick.color': INK,
                     'ytick.color': INK, 'axes.spines.top': False, 'axes.spines.right': False})


def load():
    d = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(os.path.join(RUN, 'finger_*cm_*.csv')))],
                  ignore_index=True)
    d['finger_cm'] = (d.finger_ext_m * 100).round().astype(int)
    d['truss'] = d.scan.astype(str) + ':' + d.cluster.astype(str)
    for v in VARIANTS:
        d[v] = d[f'reachable_{v}'].astype(str).str.lower().isin(['true', '1'])
    return d


def cluster_ci(g, col, rng):
    """95% CI of the pose-level reachable fraction, resampling trusses (points on one stem
    are not independent), as finger_reachability.py report does."""
    per = g.groupby('truss')[col].agg(['sum', 'size'])
    s, n = per['sum'].to_numpy(), per['size'].to_numpy()
    idx = rng.integers(0, len(per), size=(BOOTSTRAPS, len(per)))
    boot = s[idx].sum(1) / n[idx].sum(1)
    return np.percentile(boot, [2.5, 97.5])


def summary(d):
    rng = np.random.default_rng(0)
    rows = []
    for (f, m), g in d.groupby(['finger_cm', 'method']):
        row = dict(finger_cm=f, method=m, poses=len(g), points=g.point_id.nunique() if 'point_id' in g else '',
                   trusses=g.truss.nunique())
        for v in VARIANTS:
            lo, hi = cluster_ci(g, v, rng)
            row.update({v: g[v].mean(), f'{v}_lo': lo, f'{v}_hi': hi})
        rows.append(row)
    return pd.DataFrame(rows)


def paired(d, variant):
    key = ['scan', 'cluster', 'point_id', 'yaw_offset_deg', 'finger_cm']
    w = d.pivot_table(index=key, columns='method', values=variant, aggfunc='first').dropna()
    out = []
    for f, g in w.groupby(level='finger_cm'):
        b = int((g.tilted & ~g.topdown).sum())
        c = int((~g.tilted & g.topdown).sum())
        p = binomtest(b, b + c, 0.5).pvalue if b + c else 1.0
        out.append(dict(finger_cm=f, tilted_only=b, topdown_only=c, p=p))
    return pd.DataFrame(out)


def fig_vs_finger(s):
    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    for m in METHODS:
        g = s[s.method == m].sort_values('finger_cm')
        ax.errorbar(g.finger_cm, 100 * g.gripper, yerr=[100 * (g.gripper - g.gripper_lo), 100 * (g.gripper_hi - g.gripper)],
                    color=COLORS[m], marker=MARKERS[m], ms=5, lw=2, capsize=3, ls='-' if m == 'topdown' else '--',
                    label=METHODS[m], markeredgecolor='white', markeredgewidth=0.8)
    ax.axvline(OPERATING_CM, color=MUTED, lw=0.8, ls=':')
    ax.text(OPERATING_CM + 0.3, 3, 'fingers used\n(10 cm)', color=MUTED, fontsize=7.5, va='bottom')
    ax.axhline(95, color=MUTED, lw=0.8, ls='--')
    ax.text(20, 91, '95%', color=MUTED, fontsize=7.5, ha='right', va='top')
    ax.set_xticks([0, 5, 9, 13, 20])
    ax.set_xlabel('Finger extension beyond the stock fingertip (cm)')
    ax.set_ylabel('Reachable grasp poses (%)')
    ax.set_ylim(0, 103)
    ax.grid(axis='y', color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc='lower right', bbox_to_anchor=(1.0, 0.12))
    fig.tight_layout()
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(HERE, f'reachability_vs_finger.{ext}'), dpi=200)
    plt.close(fig)


def fig_by_wall(d):
    d = d.copy()
    d['wall_bin'] = pd.cut(d.wall_dist_m * 100, WALL_BINS_CM, labels=WALL_LABELS, right=False)
    fig, axes = plt.subplots(1, len(WALL_PANELS_CM), figsize=(7.6, 3.0), sharey=True)
    x = np.arange(len(WALL_LABELS))
    for ax, f in zip(axes, WALL_PANELS_CM):
        g = d[d.finger_cm == f]
        for k, m in enumerate(METHODS):
            r = g[g.method == m].groupby('wall_bin', observed=False).gripper.mean().reindex(WALL_LABELS)
            ax.bar(x + (k - 0.5) * 0.38, 100 * r.to_numpy(), width=0.36, color=COLORS[m], label=METHODS[m],
                   hatch=None if m == 'topdown' else '////', edgecolor='white', linewidth=0.6)
        ax.set_title(f'{f} cm fingers', fontsize=9, color=INK)
        ax.set_xticks(x, WALL_LABELS, fontsize=7.5, rotation=45)
        ax.set_xlabel('Distance to nearest wall (cm)')
        ax.grid(axis='y', color=GRID, lw=0.6)
        ax.set_axisbelow(True)
    axes[0].set_ylabel('Reachable grasp poses (%)')
    axes[0].set_ylim(0, 105)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, loc='upper center', ncol=2, fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(HERE, f'reachability_by_wall.{ext}'), dpi=200)
    plt.close(fig)


def table(s, mc):
    s.to_csv(os.path.join(HERE, 'table_e3.csv'), index=False)
    mc = mc.set_index('finger_cm')

    def pct(r, v):
        return f'{100 * r[v]:.1f} [{100 * r[v + "_lo"]:.0f}, {100 * r[v + "_hi"]:.0f}]'

    def pfmt(p):
        return r'$<10^{-4}$' if p < 1e-4 else f'{p:.2g}'

    lines = [r'\begin{table}[htbp]', r'\centering', r'\small',
             r'\caption{Reachable grasp poses (\%) per finger extension and grasp generator, with 95\% '
             r'cluster-bootstrap CI over trusses. 358 grasp points on 52 trusses from 9 crate scans, 5 yaws each '
             r'(1790 poses per cell). Walls: crate walls only; corridor / gripper: crate contents as obstacle '
             r'with the approach corridor or the whole gripper volume left free. Last columns: poses only one '
             r'generator reaches (gripper carve), exact McNemar test.}',
             r'\label{tab:e3_finger_reachability}', r'\begin{tabular}{rlrrrrr}', r'\toprule',
             r'Ext. (cm) & Generator & Walls & Corridor & Gripper & Only this & $p$ \\', r'\midrule']
    for f in sorted(s.finger_cm.unique()):
        for k, m in enumerate(['topdown', 'tilted']):
            r = s[(s.finger_cm == f) & (s.method == m)].iloc[0]
            only = mc.loc[f, 'topdown_only' if m == 'topdown' else 'tilted_only']
            lines.append(f'{f if k == 0 else ""} & {"Top-down" if m == "topdown" else "Tilted"} & {pct(r, "walls")} & '
                         f'{pct(r, "cylinder")} & {pct(r, "gripper")} & {only} & '
                         f'{pfmt(mc.loc[f, "p"]) if k == 0 else ""} \\\\')
        lines.append(r'\addlinespace')
    lines[-1:] = [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    with open(os.path.join(HERE, 'table_e3.tex'), 'w') as fh:
        fh.write('\n'.join(lines) + '\n')


def numbers(d, s, mc):
    out = [f"poses per finger x method: {len(d) // d.groupby(['finger_cm', 'method']).ngroups}; points "
           f"{d.drop_duplicates(['scan', 'cluster', 'point_id']).shape[0]}; trusses {d.truss.nunique()}; "
           f"scans {d.scan.nunique()}; yaw offsets {sorted(d.yaw_offset_deg.unique())}",
           f"points within 5 cm of a wall: {d.drop_duplicates(['scan', 'cluster', 'point_id']).query('wall_dist_m <= 0.05').shape[0]}"]
    for r in s.itertuples():
        out.append(f"{r.finger_cm:>2} cm {r.method:<7}: walls {100 * r.walls:.1f}%, corridor {100 * r.cylinder:.1f}%, "
                   f"gripper {100 * r.gripper:.1f}% [{100 * r.gripper_lo:.1f}, {100 * r.gripper_hi:.1f}]")
    for r in mc.itertuples():
        out.append(f"McNemar gripper {r.finger_cm} cm: tilted-only {r.tilted_only}, topdown-only {r.topdown_only}, p={r.p:.3g}")
    for m in METHODS:
        g = s[(s.method == m) & (s.gripper >= 0.95)]
        out.append(f"shortest extension with >=95% (gripper), {m}: {g.finger_cm.min()} cm")
    t = d[(d.method == 'tilted') & d.gripper & ((d.tilt_x_deg != 0) | (d.tilt_y_deg != 0))]
    for f, g in t.groupby('finger_cm'):
        n = len(d[(d.method == 'tilted') & (d.finger_cm == f)])
        out.append(f"tilted poses that needed a tilt at {f} cm: {len(g)} of {n} "
                   f"(median |tilt| {np.median(np.abs(g.tilt_x_deg) + np.abs(g.tilt_y_deg)):.0f} deg)")
    with open(os.path.join(HERE, 'numbers.txt'), 'w') as fh:
        fh.write('\n'.join(out) + '\n')
    print('\n'.join(out))


if __name__ == '__main__':
    d = load()
    s = summary(d)
    mc = paired(d, 'gripper')
    fig_vs_finger(s)
    fig_by_wall(d)
    table(s, mc)
    numbers(d, s, mc)
