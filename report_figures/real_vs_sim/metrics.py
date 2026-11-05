"""Planned vs measured per logged move (same FK as the page). Writes metrics.json; prints a summary."""
import json, numpy as np
from e4data import *
DT = .01
def tcp_series(t, q, T):
    tt = np.arange(0, T+DT/2, DT); Q = np.stack([np.interp(tt, t, q[:, j]) for j in range(7)], 1)
    return tt, Q, np.array([fk(x)[2][:3, 3] for x in Q])
def move_metrics(r):
    te, qe = executed_q(r); tp, qp = planned_q(r)
    if tp is None: return None
    Tmax = max(te[-1], tp[-1])+.35
    _, Qp, Pp = tcp_series(tp, qp, Tmax); _, Qe, Pe = tcp_series(te, qe, Tmax)
    gap = np.linalg.norm(Pp-Pe, axis=1); nact = int((max(te[-1], tp[-1]))/DT)+1
    best = (1e9, 0)
    for d in range(0, 41):
        g = np.linalg.norm(Pp[:nact]-Pe[d:nact+d], axis=1) if nact+d <= len(Pe) else None
        if g is None: break
        if g.mean() < best[0]: best = (g.mean(), d)
    d = best[1]; gd = np.linalg.norm(Pp[:nact]-Pe[d:nact+d], axis=1)
    idx = np.arange(0, len(Pp), 1); dmin = np.array([np.min(np.linalg.norm(Pp[:nact:2]-p, axis=1)) for p in Pe[:nact:2]])
    end = Pe[nact-1]; dist = np.linalg.norm(Pe[:nact]-end, axis=1); settle = np.where(dist > .001)[0]
    plen = np.linalg.norm(np.diff(Pp[:nact], axis=0), axis=1).sum(); elen = np.linalg.norm(np.diff(Pe[:nact], axis=0), axis=1).sum()
    return dict(T_plan=float(tp[-1]), T_log=float(te[-1]), T_settle=float(settle[-1]*DT) if len(settle) else 0.0, delay=d*DT, gap_max=float(gap[:nact].max()*1000), gap_rms=float(np.sqrt((gap[:nact]**2).mean())*1000),
                gap_delay_max=float(gd.max()*1000), shape_max=float(dmin.max()*1000), end_err=float(np.linalg.norm(Pp[nact-1]-Pe[nact-1])*1000), joint_max=float(np.abs(Qp[:nact]-Qe[:nact]).max()), len_plan=float(plen), len_real=float(elen), success=bool(r.get('success')), planner=r.get('planner'))
if __name__ == '__main__':
    C = clips_by_run(); M = {}; rows = []
    for run, d in sorted(C.items()):
        for vid, recs in sorted(d['clips'].items()):
            ms = []
            for r in recs:
                m = move_metrics(r); ms.append(m)
                if m: m.update(run=run, movement=r['movement'], kind=r['kind'], method=d['meta']['method'], condition=d['meta']['condition']); rows.append(m)
            M[vid] = ms
    json.dump(dict(moves=M, rows=rows), open('metrics.json', 'w'))
    f = lambda x, p: float(np.percentile(x, p))
    print('n', len(rows), 'failed', sum(not r['success'] for r in rows))
    keys = ['T_plan', 'T_log', 'T_settle', 'delay', 'gap_max', 'gap_delay_max', 'shape_max', 'end_err', 'joint_max', 'len_plan', 'len_real']
    print('%-20s %3s '%('move', 'n')+' '.join('%-13s' % k for k in keys))
    for mv in ['grasp_pre_grasp', 'grasp_descent', 'grasp_back-out', 'grasp_lift', 'go_to_place_above', 'go_to_place', 'go_to_place_retreat', 'go_to_saved_pose']:
        x = [r for r in rows if r['movement'] == mv and r['success']]
        print('%-20s %3d '%(mv, len(x))+' '.join('%-13s' % ('%.3g/%.3g' % (f([r[k] for r in x], 50), f([r[k] for r in x], 95))) for k in keys))
    w = sorted(rows, key=lambda r: -r['shape_max'])[:8]; print(*[(r['run'], r['movement'], r['kind'], round(r['shape_max'], 1), round(r['gap_max'], 1), r['T_plan'], round(r['T_log'], 2), r['success']) for r in w], sep='\n')
    w = sorted(rows, key=lambda r: -(r['T_log']-r['T_plan']))[:6]; print(*[(r['run'], r['movement'], round(r['T_plan'], 2), round(r['T_log'], 2), round(r['T_settle'], 2), round(r['gap_max'], 1), r['success']) for r in w], sep='\n')
