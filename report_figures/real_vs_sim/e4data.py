"""Shared loading for the real-vs-planned viewer: E4 trajectory records grouped per side-camera clip."""
import csv, datetime as D, glob, gzip, json, math, os, re
import numpy as np
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
EXP = os.path.join(ROOT, 'experiments')
VIDEOS = os.path.join(EXP, 'planning_metrics', 'videos')
TCP_Z = 0.2034  # hand frame -> TCP with the 10 cm E4 fingers
JN = ['panda_joint%d' % i for i in range(1, 8)]
DH = [(0, 0, .333), (0, -math.pi/2, 0), (0, math.pi/2, .316), (.0825, math.pi/2, 0), (-.0825, -math.pi/2, .384), (0, math.pi/2, 0), (.088, math.pi/2, 0)]

def _dh(a, al, d, th):
    ca, sa, ct, st = math.cos(al), math.sin(al), math.cos(th), math.sin(th)
    return np.array([[ct, -st, 0, a], [st*ca, ct*ca, -sa, -d*sa], [st*sa, ct*sa, ca, d*ca], [0, 0, 0, 1]])

def fk(q, tcp_z=TCP_Z):
    """Joint-frame origins (8 incl. base; +hand), hand and TCP transforms; same model as the path viewer."""
    T = np.eye(4); pts = [T[:3, 3].copy()]
    for i in range(7):
        T = T @ _dh(DH[i][0], DH[i][1], DH[i][2], q[i]); pts.append(T[:3, 3].copy())
    hand = T @ _dh(0, 0, .107, -math.pi/4); pts.append(hand[:3, 3].copy())
    tcp = hand @ _dh(0, 0, tcp_z, 0)
    return np.array(pts), hand, tcp

def clip_time(name):
    return D.datetime.strptime(os.path.basename(name)[:15], '%Y%m%d_%H%M%S')

def load_records():
    recs = []
    for f in sorted(glob.glob(os.path.join(EXP, 'trajectories', 'trajectories_*.jsonl.gz'))):
        with gzip.open(f, 'rt') as h:
            recs += [json.loads(l) for l in h]
    return recs

def executed_q(r):
    ex = r['executed']; idx = [ex['names'].index(j) for j in JN]
    return np.array(ex['t']), np.array(ex['positions'])[:, idx]

def planned_q(r):
    tr = r.get('trajectory')
    if not tr: return None, None
    idx = [tr['joint_names'].index(j) for j in JN]
    return np.array(tr['time_from_start']), np.array(tr['positions'])[:, idx]

def runs():
    with open(os.path.join(EXP, 'e4', 'runs.csv')) as f: rows = list(csv.DictReader(f))
    return rows

def clips_by_run():
    """{run_id: {'meta': row, 'clips': {video: [records]}}} for E4-tagged records inside the run's time window."""
    rs = {int(r['run_id']): {'meta': r, 'clips': {}} for r in runs()}
    for rec in load_records():
        m = re.match(r'e4 run (\d+) attempt (\d+)', rec.get('tag') or '')
        if not m or not rec.get('video') or not rec['executed']['t']: continue
        run = int(m.group(1)); meta = rs[run]['meta']
        s = D.datetime.strptime(meta['start'], '%Y-%m-%d %H:%M:%S'); e = D.datetime.strptime(meta['end'], '%Y-%m-%d %H:%M:%S')
        if not s <= clip_time(rec['video']) <= e: continue
        rec['_attempt'] = int(m.group(2))
        rs[run]['clips'].setdefault(rec['video'], []).append(rec)
    return {k: v for k, v in rs.items() if v['clips']}
