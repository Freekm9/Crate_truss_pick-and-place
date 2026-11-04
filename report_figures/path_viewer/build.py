"""Build the data files for the planner path viewer page.

Reads the planner-quality runs (manifest.csv per finger length), keeps the simplified and
raw variants, and writes per (finger, set, move) one JSON chunk with every saved path
(joint waypoints + retimed time stamps, compressed with a time-aware Douglas-Peucker in
joint space) and its metrics, plus index.json (queries) and clouds.json (truss points).
"""
import gzip
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = '/root/Crate_truss_pick-and-place'
RUNS = {'10': f'{ROOT}/experiments/planner_benchmark/quality_20261020/manifest.csv',
        '5': f'{ROOT}/experiments/planner_benchmark/quality_finger5cm/manifest.csv'}
VARIANTS = ('simplified', 'raw')
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'site', 'data')
METRICS = ['planning_time_s', 'exec_time_s', 'joint_path_length', 'path_ratio', 'ee_path_length_m',
           'ee_max_dev_m', 'min_clearance_mm', 'min_contents_clearance_mm', 'joint_waste_rad', 'jerk_rms']
BUCKET = 6  # goals per data file
TOL = 0.002  # rad, max joint error of the compressed path at any original sample



def compress(t, q):
    """Indices to keep so that linear interpolation of q over t stays within TOL."""
    keep = {0, len(t) - 1}
    stack = [(0, len(t) - 1)]
    while stack:
        a, b = stack.pop()
        if b - a < 2:
            continue
        span = t[b] - t[a]
        w = (t[a + 1:b] - t[a]) / span if span > 0 else np.linspace(0, 1, b - a + 1)[1:-1]
        interp = q[a] + w[:, None] * (q[b] - q[a])
        err = np.abs(q[a + 1:b] - interp).max(axis=1)
        i = int(err.argmax())
        if err[i] > TOL:
            m = a + 1 + i
            keep.add(m)
            stack += [(a, m), (m, b)]
    return sorted(keep)


def clean(v):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return None
    return round(float(v), 4)


def main():
    os.makedirs(OUT, exist_ok=True)
    index = {'fingers': {}}
    clouds_used = set()
    for finger, manifest in RUNS.items():
        m = pd.read_csv(manifest)
        m = m[m['variant'].isin(VARIANTS)]
        chunks, queries, tcp_z = {}, {}, None
        for set_name, results in m.drop_duplicates('set')[['set', 'results']].values:
            qj = json.load(open(os.path.join(os.path.dirname(results), 'queries.json')))
            tcp_z = qj['meta']['hand_tcp_offset_m']
            place = qj['meta']['place']
            for qq in qj['queries']:
                rec = queries.setdefault((set_name, qq['move']), {}).setdefault(qq['point_id'], {
                    'pid': qq['point_id'], 'x': qq['x'], 'y': qq['y'], 'z': qq['z'], 'wall': qq['wall_dist_m'],
                    'cloud': os.path.basename(qq['cloud']).replace('.pcd', ''), 'goals': []})
                clouds_used.add(qq['cloud'])
                rec['goals'].append({'method': qq['method'], 'reachable': qq['reachable'],
                                     'reason': qq['unreachable_reason'] or '', 'yaw': qq['yaw_deg'],
                                     'tilt': [qq['tilt_x_deg'], qq['tilt_y_deg']], 'id': qq['id'],
                                     'goal_joints': qq['goal_joints'], 'start': qq['start_name']})
        for _, r in m.iterrows():
            d = pd.read_csv(r['results'])
            d['success'] = d['success'].astype(str).str.lower() == 'true'
            d.loc[d['planner'] == 'RRTstar', 'planner'] = 'RRTstar@5'
            # plan_key per query id (a key is shared by the generators when the goals are equal)
            keymap = dict(d[['query_id', 'plan_key']].drop_duplicates().values)
            for qs in [v for (s, _), v in queries.items() if s == r['set']]:
                for qrec in qs.values():
                    for g in qrec['goals']:
                        if g['id'] in keymap:
                            g['key'] = keymap[g['id']]
            u = d.drop_duplicates(['plan_key', 'planner', 'repeat'])
            suffix = '' if r['variant'] == 'simplified' else '·raw'
            met = {}
            for row in u.itertuples(index=False):
                rd = row._asdict()
                met[(rd['plan_key'], rd['planner'] + suffix, int(rd['repeat']))] = rd
            p = r['results'].replace('results_', 'paths_').replace('.csv', '.jsonl.gz')
            seen = set()
            with gzip.open(p, 'rt') as f:
                for line in f:
                    rec = json.loads(line)
                    pl = ('RRTstar@5' if rec['planner'] == 'RRTstar' else rec['planner']) + suffix
                    key, rep = rec['plan_key'], int(rec['repeat'])
                    move = key.split(':')[0]
                    q = np.array(rec['waypoints'], float)
                    t = np.array(rec.get('t') or np.linspace(0, 1, len(q)), float)
                    if len(t) != len(q):
                        t = np.linspace(0, t[-1] if len(t) else 1, len(q))
                    keep = compress(t, q)
                    # Delta-coded: t in ms, q in mrad (per joint along the path)
                    ti = np.round(t[keep] * 1000).astype(int)
                    qi = np.round(q[keep] * 1000).astype(int)
                    entry = {'r': rep, 't': np.diff(ti, prepend=0).tolist(),
                             'q': np.diff(qi, axis=0, prepend=np.zeros((1, 7), int)).ravel().tolist()}
                    mrow = met.get((key, pl, rep))
                    if mrow:
                        entry['m'] = {k: clean(mrow.get(k)) for k in METRICS}
                    ch = chunks.setdefault((r['set'], move), {})
                    ch.setdefault(key, {}).setdefault(pl, {'paths': [], 'fail': []})['paths'].append(entry)
                    seen.add((key, pl, rep))
            for (key, pl, rep), rd in met.items():
                if not rd['success'] and (key, pl, rep) not in seen:
                    move = key.split(':')[0]
                    ch = chunks.setdefault((r['set'], move), {})
                    ch.setdefault(key, {}).setdefault(pl, {'paths': [], 'fail': []})['fail'].append(
                        {'r': rep, 'err': clean(rd.get('error_code')), 'why': rd.get('pink_reason') if
                         isinstance(rd.get('pink_reason'), str) else None})
        fin = {'tcp_z': tcp_z, 'place': place, 'sets': {}}
        for (set_name, move), ch in sorted(chunks.items()):
            # Small files of BUCKET goals each, so the page loads only the selected goal's paths;
            # the crate map colours come from the per-goal summary in the index
            keys = sorted(ch)
            bucket, stats = {}, {}
            for b in range(0, len(keys), BUCKET):
                part = {k: ch[k] for k in keys[b:b + BUCKET]}
                name = f'f{finger}_{set_name}_{move}_{b // BUCKET}.json'
                with open(os.path.join(OUT, name), 'w') as f:
                    json.dump(part, f, separators=(',', ':'))
                for k in part:
                    bucket[k] = name
            for k, e in ch.items():
                lens = [p['m']['ee_path_length_m'] for pl, v in e.items() if not pl.endswith('·raw') and pl != 'pink'
                        for p in v['paths'] if p.get('m') and p['m'].get('ee_path_length_m')]
                fail = sum(len(v['fail']) for pl, v in e.items() if not pl.endswith('·raw') and pl != 'pink')
                stats[k] = [round(max(lens) / min(lens), 3) if lens else 1, fail]
            qs = []
            for qrec in queries.get((set_name, move), {}).values():
                goals = [{k: g[k] for k in ('method', 'reachable', 'reason', 'yaw', 'tilt', 'key') if k in g}
                         for g in qrec['goals']]
                qs.append({**{k: qrec[k] for k in ('pid', 'x', 'y', 'z', 'wall', 'cloud')}, 'goals': goals})
            fin['sets'].setdefault(set_name, {})[move] = {'files': bucket, 'stats': stats, 'queries': qs}
            print(finger, set_name, move, len(ch), 'keys', len(set(bucket.values())), 'files')
        index['fingers'][finger] = fin
    with open(os.path.join(OUT, 'index.json'), 'w') as f:
        json.dump(index, f, separators=(',', ':'))
    # Truss clouds: crop to the crate interior, 8 mm voxels, colour kept
    os.makedirs(os.path.join(OUT, 'clouds'), exist_ok=True)
    for c in sorted(clouds_used):
        pts = np.loadtxt(c, skiprows=11, usecols=(0, 1, 2, 3))
        x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
        inside = (x > 0.335) & (x < 0.585) & (y > -0.08) & (y < 0.28) & (z > 0.008) & (z < 0.2)
        pts = pts[inside]
        vox = np.floor(pts[:, :3] / 0.008).astype(int)
        _, first = np.unique(vox, axis=0, return_index=True)
        pts = pts[first]
        rgb = pts[:, 3].astype(np.uint32)
        with open(os.path.join(OUT, 'clouds', os.path.basename(c).replace('.pcd', '.json')), 'w') as f:
            json.dump({'p': [int(round(v * 1000)) for v in pts[:, :3].ravel()],
                       'c': [int(v) & 0xFFFFFF for v in rgb]}, f, separators=(',', ':'))
        print('cloud', c, len(pts))


if __name__ == '__main__':
    main()
