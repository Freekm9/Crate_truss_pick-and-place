#!/usr/bin/env python3
"""Height coverage of a points file (finger_reachability.py points), per scan, to judge a recording while it is made;
optionally writes the low points as a new points file for planner_benchmark.py build --points-file. No ROS needed.

  low_points.py POINTS.json [--empty-cloud EMPTY.pcd] [--max-above-floor-cm 5.5 --out LOW.json]
--empty-cloud measures the floor in a scan of the empty crate (default: the floor in the MoveIt scene)."""
import argparse, json
import numpy as np
CORNER_M = 0.06

def measured_floor(path, interior):
    x0, x1, y0, y1, *_ = interior
    c = np.loadtxt(path, skiprows=11, usecols=(0, 1, 2)); c = c[np.isfinite(c).all(1)]
    c = c[(c[:, 0] > x0+.03) & (c[:, 0] < x1-.03) & (c[:, 1] > y0+.03) & (c[:, 1] < y1-.03) & (c[:, 2] < .1)]
    return float(np.median(c[:, 2])), float(np.percentile(c[:, 2], 95)-np.percentile(c[:, 2], 5))

def main():
    a = argparse.ArgumentParser(); a.add_argument('points'); a.add_argument('--empty-cloud'); a.add_argument('--max-above-floor-cm', type=float); a.add_argument('--out'); a = a.parse_args()
    if a.out and a.max_above_floor_cm is None: raise SystemExit('--out needs --max-above-floor-cm')
    d = json.load(open(a.points)); P = d['points']; I = d['meta']['interior']; x0, x1, y0, y1, floor, _ = I
    print('Floor in the MoveIt scene: z %.1f cm' % (floor*100))
    if a.empty_cloud:
        floor, spread = measured_floor(a.empty_cloud, I); print('Floor measured in the empty crate: z %.1f cm (5-95%% spread %.1f cm); heights below use this' % (floor*100, spread*100))
    for p in P:
        p['above_floor_m'] = round(p['z']-floor, 4); p['corner'] = min(p['x']-x0, x1-p['x']) < CORNER_M and min(p['y']-y0, y1-p['y']) < CORNER_M
    print('\n%-16s %7s %7s %8s | points at most this high (cm) %s | wall<3cm corner  (of points <=5.5 cm)' % ('scan', 'trusses', 'points', 'lowest', ''))
    print('%-16s %7s %7s %8s | %5s %5s %5s %5s' % ('', '', '', 'cm', '4.5', '5', '5.5', '6'))
    def row(name, q):
        z = np.array([p['above_floor_m']*100 for p in q]); low = [p for p in q if p['above_floor_m'] <= .055]
        print('%-16s %7d %7d %8.1f | %5d %5d %5d %5d | %8d %6d' % (name, len({(p['scan'], p['cluster']) for p in q}), len(q), z.min(), *[int((z <= t).sum()) for t in (4.5, 5, 5.5, 6)], sum(p['wall_dist_m'] < .03 for p in low), sum(p['corner'] for p in low)))
    for s in sorted({p['scan'] for p in P}): row(s, [p for p in P if p['scan'] == s])
    row('all', P)
    if a.out:
        q = [p for p in P if p['above_floor_m']*100 <= a.max_above_floor_cm]
        d['meta'].update(source=a.points, max_above_floor_cm=a.max_above_floor_cm, floor_z=floor); d['points'] = q
        json.dump(d, open(a.out, 'w'), indent=1); print('\n%d of %d points on %d trusses -> %s' % (len(q), len(P), len({(p['scan'], p['cluster']) for p in q}), a.out))

if __name__ == '__main__': main()
