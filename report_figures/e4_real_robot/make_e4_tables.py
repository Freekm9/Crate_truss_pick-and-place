"""E4 (real robot, preliminary): summary table + numbers from experiments/e4 and the
planning metrics logged during the runs. Run: python3 make_e4_tables.py"""
import os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, '..', '..')
E4 = os.path.join(ROOT, 'experiments', 'e4')

runs = pd.read_csv(os.path.join(E4, 'runs.csv'), parse_dates=['start', 'end'])
att = pd.read_csv(os.path.join(E4, 'attempts.csv'), parse_dates=['timestamp'])
met = pd.read_csv(os.path.join(ROOT, 'experiments', 'planning_metrics', 'metrics.csv'), parse_dates=['timestamp'])
met = met[(met.timestamp >= runs.start.min()) & (met.timestamp <= runs.end.max())]

att = att.sort_values(['run_id', 'attempt'])
# Attempt-to-attempt interval: one full cycle (click, plan, pick, place, operator input).
att['cycle_s'] = att.groupby('run_id').timestamp.diff().dt.total_seconds()
# First-attempt success: a truss counts once; an attempt follows a failure -> retry on the same truss.
att['after_failure'] = att.groupby('run_id').removed.shift(1).eq(False)

def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z ** 2 / n
    c = (p + z ** 2 / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / d
    return c - h, c + h


lines, rows = [], []
for method, a in att.groupby('method', sort=False):
    r = runs[runs.method == method]
    n, k = len(a), int(a.removed.sum())
    lo, hi = wilson(k, n)
    first = a[~a.after_failure]
    k1 = int(first.removed.sum())
    cyc = a.cycle_s.dropna()
    rows.append(dict(method=method, runs=len(r), trusses=int(r.trusses_start.sum()), removed=k, attempts=n,
                     success=100 * k / n, lo=100 * lo, hi=100 * hi, first=100 * k1 / len(first),
                     clear=f"{int((r.clearance_fraction == 1).sum())}/{len(r)}",
                     cyc_med=cyc.median(), cyc_q1=cyc.quantile(.25), cyc_q3=cyc.quantile(.75),
                     moved=int((a.disturbance != 'none').sum()), tilted=int(((a.tilt_x_deg != 0) | (a.tilt_y_deg != 0)).sum()),
                     flip=int(a.flip.sum()), predicted=int(a.predicted_reachable.sum())))
    lines.append(f"{method}: runs {len(r)}, trusses {rows[-1]['trusses']}, removed {k}/{n} attempts "
                 f"({100*k/n:.1f}% [{100*lo:.1f}, {100*hi:.1f}]), first-attempt {k1}/{len(first)}, "
                 f"cycle median {cyc.median():.0f} s IQR [{cyc.quantile(.25):.0f}, {cyc.quantile(.75):.0f}] max {cyc.max():.0f} s, "
                 f"disturbed {rows[-1]['moved']}, tilted {rows[-1]['tilted']}, 180-roll {rows[-1]['flip']}, "
                 f"predicted reachable {rows[-1]['predicted']}/{n}; failures: "
                 f"{a.loc[~a.removed, 'notes'].tolist()}")

for mv, m in met.groupby('movement'):
    ok = m[m.planning_success]
    ratio = (ok.path_length / ok.direct_joint_distance)
    lines.append(f"planning {mv} ({', '.join(m.planner_id.unique())}): {int(m.planning_success.sum())}/{len(m)} planned, "
                 f"t_med {m.planning_duration_s.median():.3f} s, t_p95 {m.planning_duration_s.quantile(.95):.3f} s, "
                 f"exec ok {int(ok.execution_success.sum())}/{len(ok)}, exec med {ok.execution_duration_s.median():.2f} s, "
                 f"path ratio med {ratio.median():.3f} max {ratio.max():.3f}")

with open(os.path.join(HERE, 'numbers.txt'), 'w') as f:
    f.write('\n'.join(lines) + '\n')
print('\n'.join(lines))

name = {'topdown': 'Top-down', 'tilted': 'Tilted'}
body = '\n'.join(
    f"{name[r['method']]} & {r['runs']} & {r['removed']}/{r['trusses']} & {r['attempts']} & "
    f"{r['success']:.1f} [{r['lo']:.0f}, {r['hi']:.0f}] & {r['first']:.1f} & {r['clear']} & "
    f"{r['cyc_med']:.0f} [{r['cyc_q1']:.0f}, {r['cyc_q3']:.0f}] & {r['moved']} \\\\" for r in rows)
tex = r"""\begin{table}[htbp]
\centering
\small
\caption{Preliminary real-robot results (E4), structured crate only, 10\,cm fingers, RRTConnect. Success = trusses removed per grasp attempt, with Wilson 95\% CI. Cycle time = interval between consecutive attempts (median [IQR]), including the operator's grasp-point click. Disturbed = attempts that moved a neighbouring truss.}
\label{tab:e4_preliminary}
\setlength{\tabcolsep}{3pt}
\begin{tabular}{lrrrrrrrr}
\toprule
Generator & Runs & Removed & Att. & Success (\%) & First att. (\%) & Cleared & Cycle (s) & Disturbed \\
\midrule
""" + body + r"""
\bottomrule
\end{tabular}
\end{table}
"""
with open(os.path.join(HERE, 'table_e4.tex'), 'w') as f:
    f.write(tex)
pd.DataFrame(rows).to_csv(os.path.join(HERE, 'table_e4.csv'), index=False)
