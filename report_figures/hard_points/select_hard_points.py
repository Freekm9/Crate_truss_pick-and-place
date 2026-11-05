#!/usr/bin/env python3
"""Select hard picking points from the real stem points of the 9 whole-crate scans (E3).
Output keeps the points-file format, so `planner_benchmark.py build --points-file` reads it. No ROS needed.
Groups are chosen on geometry and on the E3 reach results, never on planner results."""
import collections, csv, json, os
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')); EXP = os.path.join(ROOT, 'experiments')
SRC = os.path.join(EXP, 'finger_reachability', 'points_20261014_120918.json'); OUT = os.path.join(EXP, 'hard_points')
REACH = {'0': 'finger_0cm_20261014_122002.csv', '5': 'finger_5cm_20261014_125134.csv', '10': 'finger_10cm_20261014_133750.csv'}
LOW_Z = 0.07        # stem height in world z; about the lowest third of the points
CORNER_M = 0.06     # closer than this to two walls
WALL_M = 0.03
RISE_R = 0.06       # radius for "how much higher is the crate content around the point"
BURIED_M = 0.04
OPEN_WALL_M = 0.08; OPEN_RISE_M = 0.02
RIM_Z = 0.17        # above this a cloud point is not crate content (arm, noise)
WALL_SKIP = 0.01    # wall surface points must not count as content
PER_GROUP = 8; PER_CONTROL = 6
KNOWN = {'20261014_120728:237': 'only point where PTP is blocked and samplers detour (5 cm benchmark)'}

def read_pcd(path):
    b = open(path, 'rb').read(); i = b.index(b'DATA'); j = b.index(b'\n', i)
    h = dict(l.split(' ', 1) for l in b[:j].decode(errors='ignore').splitlines() if ' ' in l)
    if h['DATA'].strip() == 'ascii': return np.loadtxt(b[j+1:].decode().splitlines(), usecols=(0, 1, 2))
    if h['DATA'].strip() != 'binary': raise ValueError(path)
    names = h['FIELDS'].split(); sizes = h['SIZE'].split(); types = h['TYPE'].split(); counts = h.get('COUNT', ' '.join('1'*len(names))).split()
    dt = [(n, t.lower()+s, int(c)) if int(c) > 1 else (n, t.lower()+s) for n, s, t, c in zip(names, sizes, types, counts)]
    a = np.frombuffer(b, np.dtype(dt), int(h['POINTS']), j+1)
    return np.c_[a['x'], a['y'], a['z']].astype(float)

def main():
    src = json.load(open(SRC)); meta = src['meta']; P = src['points']; x0, x1, y0, y1, z0, _ = meta['interior']
    clouds = {}
    for f in {p['cloud'] for p in P}:
        c = read_pcd(f); c = c[np.isfinite(c).all(1)]
        clouds[f] = c[(c[:, 0] > x0+WALL_SKIP) & (c[:, 0] < x1-WALL_SKIP) & (c[:, 1] > y0+WALL_SKIP) & (c[:, 1] < y1-WALL_SKIP) & (c[:, 2] < RIM_Z)]
    reach = collections.defaultdict(collections.Counter)
    for cm, name in REACH.items():
        for r in csv.DictReader(open(os.path.join(EXP, 'finger_reachability', 'full_20261014', name))):
            e = reach[(r['scan'], int(r['point_id']))]; e[cm, r['method'], 'n'] += 1; e[cm, r['method'], 'ok'] += r['reachable_gripper'] == 'True'
    for p in P:
        c = clouds[p['cloud']]; s = c[np.hypot(c[:, 0]-p['x'], c[:, 1]-p['y']) < RISE_R, 2]
        p['above_floor_m'] = round(p['z']-z0, 4); p['rise_6cm_m'] = round(max(0., float(np.percentile(s, 98))-p['z']), 4) if len(s) > 5 else 0.
        p['dist_x_wall_m'] = round(min(p['x']-x0, x1-p['x']), 4); p['dist_y_wall_m'] = round(min(p['y']-y0, y1-p['y']), 4)
        e = reach[(p['scan'], p['id'])]
        p['reach'] = {'%scm_%s' % (cm, m): '%d/%d' % (e[cm, m, 'ok'], e[cm, m, 'n']) for cm in REACH for m in ('topdown', 'tilted')}
        p['misses'] = sum(e[cm, m, 'n']-e[cm, m, 'ok'] for cm in ('0', '5') for m in ('topdown', 'tilted'))
    def corner(p): return p['dist_x_wall_m'] < CORNER_M and p['dist_y_wall_m'] < CORNER_M
    groups = [
        ('reach_limit', 'Short-finger limit: a tilted grasp fails at 5 cm', lambda p: p['reach']['5cm_tilted'].split('/')[0] != p['reach']['5cm_tilted'].split('/')[1]),
        ('low_corner', 'Low in a corner', lambda p: p['z'] < LOW_Z and corner(p)),
        ('low_wall', 'Low against a wall', lambda p: p['z'] < LOW_Z and p['wall_dist_m'] < WALL_M),
        ('low_buried', 'Low between taller content', lambda p: p['z'] < LOW_Z and p['rise_6cm_m'] >= BURIED_M and p['wall_dist_m'] >= WALL_M),
        ('low_open', 'Control: low, open, away from walls', lambda p: p['z'] < LOW_Z and p['rise_6cm_m'] < OPEN_RISE_M and p['wall_dist_m'] >= OPEN_WALL_M),
    ]
    def rank(xs):  # hardest first gets 0: low, close to a wall, surrounded, many short-finger misses
        out = np.zeros(len(xs))
        for key in (lambda p: p['z'], lambda p: p['wall_dist_m'], lambda p: -p['rise_6cm_m'], lambda p: -p['misses']):
            v = np.array([key(p) for p in xs]); out += np.unique(v, return_inverse=True)[1]
        return out
    used = set(); chosen = []; pool_sizes = {}
    for key, label, test in groups:
        pool = [p for p in P if test(p)]; pool_sizes[key] = (len(pool), len({(p['scan'], p['cluster']) for p in pool}))
        r = rank(pool) if key != 'low_open' else np.array([p['z'] for p in pool])  # the control is only ordered by height
        order = sorted(range(len(pool)), key=lambda i: (r[i], pool[i]['scan'], pool[i]['id']))
        forced = [i for i in order if '%s:%d' % (pool[i]['scan'], pool[i]['id']) in KNOWN]
        got = []; trusses = set(); n = PER_CONTROL if key == 'low_open' else PER_GROUP
        for i in forced+order:  # one point per truss per group, so one truss cannot fill a group
            p = pool[i]; t = (p['scan'], p['cluster']); u = (p['scan'], p['id'])
            if t in trusses or u in used or len(got) >= n: continue
            trusses.add(t); used.add(u); got.append(p)
            p['group'] = key; p['why'] = KNOWN.get('%s:%d' % u, label)
        chosen += got
    Q = {}
    for name in ('typical', 'edge'):
        ids = {q.get('point_id') for q in json.load(open(os.path.join(EXP, 'planner_benchmark', 'quality_20261020', name, 'queries.json')))['queries']}
        Q[name] = ids
    for p in chosen:
        pid = '%s:%d' % (p['scan'], p['id']); p['in_benchmark'] = [n for n, ids in Q.items() if pid in ids]
    os.makedirs(OUT, exist_ok=True)
    m = dict(source=SRC, created_by='report_figures/hard_points/select_hard_points.py', interior=meta['interior'], scene=meta['scene'],
             thresholds=dict(LOW_Z=LOW_Z, CORNER_M=CORNER_M, WALL_M=WALL_M, RISE_R=RISE_R, BURIED_M=BURIED_M, OPEN_WALL_M=OPEN_WALL_M, OPEN_RISE_M=OPEN_RISE_M, RIM_Z=RIM_Z), pool_sizes=pool_sizes)
    json.dump(dict(meta=m, points=chosen), open(os.path.join(OUT, 'points_hard.json'), 'w'), indent=1)
    cols = ['group', 'scan', 'id', 'cluster', 'x', 'y', 'z', 'above_floor_m', 'wall_dist_m', 'rise_6cm_m', 'yaw_stem_deg', 'misses', 'in_benchmark', 'why']
    with open(os.path.join(OUT, 'points_hard.csv'), 'w', newline='') as f:
        w = csv.writer(f); w.writerow(cols+list(chosen[0]['reach'])); [w.writerow([('+'.join(p[c]) if c == 'in_benchmark' else p[c]) for c in cols]+list(p['reach'].values())) for p in chosen]
    # Crate seen from above, and the same points as height against wall distance
    style = {'reach_limit': ('#C2410C', 'X'), 'low_corner': ('#1D4ED8', 's'), 'low_wall': ('#0F766E', '^'), 'low_buried': ('#7E22CE', 'D'), 'low_open': ('#4B5563', 'o')}
    fig, ax = plt.subplots(1, 2, figsize=(12, 5.2), gridspec_kw=dict(width_ratios=[1, 1.15]))
    ax[0].add_patch(plt.Rectangle((y0, x0), y1-y0, x1-x0, fill=False, ec='#111827', lw=1.5))
    ax[0].scatter([p['y'] for p in P], [p['x'] for p in P], s=9, c='#D1D5DB', label='All %d stem points' % len(P))
    ax[1].scatter([p['wall_dist_m']*100 for p in P], [p['above_floor_m']*100 for p in P], s=9, c='#D1D5DB')
    for key, label, _ in groups:
        g = [p for p in chosen if p['group'] == key]; c, mk = style[key]
        ax[0].scatter([p['y'] for p in g], [p['x'] for p in g], s=60, c=c, marker=mk, ec='white', lw=.8, label='%s (%d)' % (label, len(g)))
        ax[1].scatter([p['wall_dist_m']*100 for p in g], [p['above_floor_m']*100 for p in g], s=60, c=c, marker=mk, ec='white', lw=.8)
    ax[0].set_aspect('equal'); ax[0].set_xlabel('y (m)'); ax[0].set_ylabel('x (m)'); ax[0].set_title('Crate seen from above', loc='left', fontsize=11)
    ax[1].set_xlabel('Distance to nearest wall (cm)'); ax[1].set_ylabel('Stem height above crate floor (cm)'); ax[1].set_title('Lower left is hardest', loc='left', fontsize=11)
    ax[1].grid(alpha=.25); h, l = ax[0].get_legend_handles_labels(); fig.legend(h, l, loc='upper center', bbox_to_anchor=(.5, .02), ncol=3, frameon=False, fontsize=9)
    for a in ax: a.spines[['top', 'right']].set_visible(False)
    fig.savefig(os.path.join(OUT, 'points_hard_map.png'), dpi=150, bbox_inches='tight')
    print('pool sizes (points, trusses):', pool_sizes)
    for key, label, _ in groups:
        g = [p for p in chosen if p['group'] == key]
        print('%-12s n=%d trusses=%d above floor %.1f-%.1f cm  wall %.1f-%.1f cm  rise %.1f-%.1f cm  in earlier benchmark %d' % (key, len(g), len({(p['scan'], p['cluster']) for p in g}), min(p['above_floor_m'] for p in g)*100, max(p['above_floor_m'] for p in g)*100, min(p['wall_dist_m'] for p in g)*100, max(p['wall_dist_m'] for p in g)*100, min(p['rise_6cm_m'] for p in g)*100, max(p['rise_6cm_m'] for p in g)*100, sum(bool(p['in_benchmark']) for p in g)))
    print('total', len(chosen), 'trusses', len({(p['scan'], p['cluster']) for p in chosen}), 'of', len({(p['scan'], p['cluster']) for p in P}), 'scans', len({p['scan'] for p in chosen}))
    print('all points above floor min %.1f cm; rise pct' % (min(p['above_floor_m'] for p in P)*100), np.percentile([p['rise_6cm_m'] for p in P], [5, 50, 95, 100]).round(3))

if __name__ == '__main__': main()
