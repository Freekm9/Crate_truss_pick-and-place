#!/usr/bin/env python3
"""Live planner feedback: choose a pick point in a saved crate scan in the browser and see what each planner does.
Plans only, in a simulator on its own ROS master (run_live.sh starts it). Nothing can be executed from here.

Uses the benchmark's own steps, so a result compares with the 10 cm and 5 cm experiments: orientation choice as
simple_pick_point.py (test_crate_reachability), crate contents carved as in the benchmark, nearest-IK joint goal of the pre-grasp
pose (or its 180-degree roll), planner_benchmark.plan_query and path_metrics."""
import argparse, json, math, os, sys, threading, time, types
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
GRASP = os.path.join(ROOT, 'src/learning_approach_to_robotic_grasping_of_vine_tomatoes/grasp/src')
sys.path[:0] = [GRASP, os.path.join(GRASP, 'stem_segmentation')]
EXP = os.path.join(ROOT, 'experiments')
CAPTURE = [-0.17830975928446877, -0.6060327979481354, 0.4456400372147912, -2.5271529010268137, 0.31681683511204184, 1.960740570757357, 0.8233889236590929]  # the robot's capture pose, as in run_planner_quality.sh
POINT_FILES = [('low', os.path.join(EXP, 'hard_points', 'points_low.json')), ('october', os.path.join(EXP, 'finger_reachability', 'points_20261014_120918.json'))]
# Fast planners first, so most of the answer is there before RRT* has used its whole budget
PLANNERS = [dict(id='RRTConnect', spec='RRTConnect', simplify=True), dict(id='BiTRRT', spec='BiTRRT', simplify=True), dict(id='pilz_ptp', spec='pilz_ptp', simplify=True),
            dict(id='pilz_lin', spec='pilz_lin', simplify=True), dict(id='RRTConnect·raw', spec='RRTConnect', simplify=False), dict(id='BiTRRT·raw', spec='BiTRRT', simplify=False),
            dict(id='chomp', spec='chomp', simplify=True), dict(id='pink', spec='pink', simplify=True), dict(id='RRTstar@1', spec='RRTstar@1', simplify=True), dict(id='RRTstar@5', spec='RRTstar@5', simplify=True)]
METRICS = ('exec_time_s', 'ee_path_length_m', 'ee_max_dev_m', 'joint_path_length', 'path_ratio', 'min_clearance_mm', 'min_contents_clearance_mm', 'goal_deviation_rad', 'num_waypoints', 'joint_reversals')
ERRORS = {-1: 'planning failed', -2: 'invalid motion plan', -4: 'control failed', -6: 'timed out', -7: 'preempted', -10: 'start in collision', -11: 'start violates constraints', -12: 'goal in collision', -13: 'goal violates constraints', -31: 'no IK solution', 99999: 'failed'}
DEFAULTS = dict(planning_time=5.0, attempts=100, repeats=1, speed=0.3, contents=True, carve='gripper', orientation='tilted', tilt_x=0, tilt_y=0, ik_seeds=30)
CRATE_MARGIN = 0.03; CLOUD_VOXEL = 0.003


def clean(v):
    """NaN is not valid JSON for the browser."""
    return v if isinstance(v, (int, float, str, bool)) and not (isinstance(v, float) and not math.isfinite(v)) else None


class Catalog:
    """The saved scans with their detected stem points and camera images. Offline, needs no ROS."""
    def __init__(self):
        self.scans = {}; self.clouds = {}
        for name, path in POINT_FILES:
            d = json.load(open(path)); self.interior = d['meta']['interior'] if name == 'october' else getattr(self, 'interior', d['meta']['interior'])
            self.floor = d['meta'].get('floor_z_measured', getattr(self, 'floor', None))
            for p in d['points']:
                s = self.scans.setdefault(p['scan'], dict(id=p['scan'], group=p.get('layout', name), cloud=p['cloud'], points=[]))
                s['points'].append(dict(x=p['x'], y=p['y'], z=p['z'], yaw=p['yaw_stem_deg'], wall=p['wall_dist_m'], truss=p['cluster'], id=p['id']))
        for s in self.scans.values():
            png = os.path.splitext(s['cloud'])[0]+'.png'; s['image'] = png if os.path.exists(png) else None

    def summary(self):
        order = {'packed': 0, 'edge': 1, 'october': 2}
        return [dict(id=s['id'], group=s['group'], n=len(s['points']), image=bool(s['image'])) for s in sorted(self.scans.values(), key=lambda s: (order.get(s['group'], 9), s['id']))]

    def xyz(self, scan):
        from pcd_io import load_pcd
        s = self.scans[scan]
        if s['cloud'] not in self.clouds: self.clouds[s['cloud']] = load_pcd(s['cloud'])
        return self.clouds[s['cloud']]

    def page_data(self, scan):
        """Thinned coloured cloud of the crate for the map and the 3D view, plus the stem points."""
        xyz, rgb = self.xyz(scan); x0, x1, y0, y1, *_ = self.interior; m = CRATE_MARGIN
        keep = (xyz[:, 0] > x0-m) & (xyz[:, 0] < x1+m) & (xyz[:, 1] > y0-m) & (xyz[:, 1] < y1+m) & (xyz[:, 2] < .2)
        xyz, rgb = xyz[keep], rgb[keep]
        order = np.argsort(-xyz[:, 2]); xyz, rgb = xyz[order], rgb[order]  # the highest point of a voxel column is what the camera sees
        _, first = np.unique(np.floor(xyz/CLOUD_VOXEL).astype(np.int64), axis=0, return_index=True); xyz, rgb = xyz[first], rgb[first]
        c = np.round(rgb*255).astype(np.int64); rgb = (c[:, 0] << 16) | (c[:, 1] << 8) | c[:, 2]  # load_pcd gives 0..1
        return dict(id=scan, p=np.round(xyz*1000).astype(int).ravel().tolist(), c=rgb.astype(np.int64).tolist(), stems=self.scans[scan]['points'], image=bool(self.scans[scan]['image']))


class Job:
    def __init__(self, jid, req): self.id, self.req, self.state, self.trace, self.goal, self.results, self.error = jid, req, 'queued', [], None, [], None; self.t0 = time.time()
    def note(self, text, ok=None): self.trace.append(dict(text=text, ok=ok, t=round(time.time()-self.t0, 2)))


class Planner:
    """All MoveIt calls happen on this one worker thread. A new request replaces the waiting one and stops the running one between plans."""
    def __init__(self, catalog):
        import rospy
        from moveit_msgs.srv import GetPositionIK
        from move_robot import planner_benchmark as pb
        from move_robot import test_crate_reachability as tcr
        rospy.init_node('live_planner', disable_signals=True)
        self.pb, self.tcr, self.catalog = pb, tcr, catalog
        self.bench = pb.Bench('panda'); self.bench.tcp = self.bench.tcp_offset_m()
        self.ik = tcr.IKChecker('panda'); self.ik_service = rospy.ServiceProxy(f'{self.bench.ns}/compute_ik', GetPositionIK)
        self.contents = self.bench.contents_obstacle('gripper'); self.simplify = None
        self.lock = threading.Condition(); self.jobs = {}; self.next = None; self.current = None; self.count = 0
        threading.Thread(target=self.loop, daemon=True).start()

    def submit(self, req):
        with self.lock:
            if self.next is not None: self.next.state = 'cancelled'  # replaced before it started
            self.count += 1; job = Job(self.count, req); self.jobs[job.id] = job; self.next = job
            for old in [k for k in self.jobs if k < job.id-20]: del self.jobs[old]
            self.lock.notify(); return job.id

    def superseded(self, job):
        return self.next is not None and self.next.id > job.id

    def loop(self):
        while True:
            with self.lock:
                while self.next is None: self.lock.wait()
                job, self.next = self.next, None; self.current = job
            job.state = 'running'
            try:
                self.run(job); job.state = 'cancelled' if self.superseded(job) else 'done'
            except Exception as e:
                job.error = '%s: %s' % (type(e).__name__, e); job.state = 'error'; import traceback; traceback.print_exc()
            finally:
                try: self.contents.remove()
                except Exception: pass

    def set_simplify(self, value):
        if self.simplify != value: self.pb.set_ompl_simplify(self.bench, value); self.simplify = value

    def orientation(self, job, base):
        """The same order as simple_pick_point._find_reachable_orientation; every check is reported to the page."""
        r, tcr, ik = job.req, self.tcr, self.ik
        if r['orientation'] == 'manual':
            pose = tcr.tilt_pose(base, r['tilt_x'], r['tilt_y']) if (r['tilt_x'] or r['tilt_y']) else base
            ok = ik.check_grasp(pose); job.note('Chosen tilt %d° / %d°: whole pick sequence %s' % (r['tilt_x'], r['tilt_y'], 'feasible' if ok else 'not feasible'), ok)
            return (r['tilt_x'], r['tilt_y'], False, pose) if ok else None
        ok = ik.check_grasp(base); job.note('Straight down: %s' % ('feasible' if ok else 'not feasible'), ok)
        if ok: return 0, 0, False, base
        flipped = tcr.flip_approach_roll(base); ok = ik.check_grasp(flipped); job.note('Straight down, wrist rolled 180°: %s' % ('feasible' if ok else 'not feasible'), ok)
        if ok: return 0, 0, True, flipped
        if r['orientation'] == 'topdown': job.note('Top-down generator tries no tilts'); return None
        tilts = sorted(((dx, dy) for dx in tcr.TILT_ANGLES_DEG for dy in tcr.Y_TILT_ANGLES_DEG if dx or dy), key=lambda t: abs(t[0])+abs(t[1]))
        for n, (dx, dy) in enumerate(tilts):
            if self.superseded(job): return None
            pose = tcr.tilt_pose(base, dx, dy)
            if ik.check_grasp(pose): job.note('Tilt %d° / %d° feasible, after %d other tilts failed' % (dx, dy, n), True); return dx, dy, False, pose
        job.note('None of the %d tilts is feasible' % len(tilts), False); return None

    def run(self, job):
        r, pb, tcr, bench = job.req, self.pb, self.tcr, self.bench
        base = tcr.build_straight_down_pose(r['x'], r['y'], r['z'], r['yaw'], bench.frame)
        chosen = self.orientation(job, base)
        if chosen is None:
            if not self.superseded(job): job.goal = dict(reachable=False, reason='no_orientation', text='No feasible gripper orientation')
            return
        dx, dy, flip, grasp = chosen
        candidates = [(grasp, flip)]; rolled = tcr.flip_approach_roll(grasp)
        if self.ik.check_grasp(rolled): candidates.append((rolled, not flip))
        approach = tcr.get_approach_vec(grasp.pose); columns, cell, floor = None, self.contents.cell_m, 0.0
        if r['contents']:
            self.contents.carve_mode = r['carve']
            n = self.contents.add([r['x'], r['y'], r['z']], approach, xyz=self.catalog.xyz(r['scan'])[0], grasp_quat=pb.pose_to_list(grasp)[3:])
            columns, floor = list(self.contents.columns), self.contents.floor_z; job.note('Crate contents added as %d columns, carved around the approach (%s)' % (n, r['carve']), True)
        else: job.note('Crate contents left out', None)
        best = None
        for pose, is_flipped in candidates:
            pre = tcr.offset_pose(pose, approach, tcr.PRE_GRASP_OFFSET_M)
            q, dist, found = pb.nearest_ik(bench, self.ik_service, pre, CAPTURE, r['ik_seeds'])
            if q is not None and (best is None or dist < best['direct']): best = dict(q=q, direct=dist, found=found, pose=pose, pre=pre, flip=is_flipped)
        if best is None:
            reason = 'no_goal_ik'; text = 'No collision-free pre-grasp joint goal'
            if r['contents']:
                self.contents.remove()
                if any(pb.nearest_ik(bench, self.ik_service, tcr.offset_pose(p, approach, tcr.PRE_GRASP_OFFSET_M), CAPTURE, 3)[0] is not None for p, _ in candidates):
                    reason, text = 'goal_in_crate_contents', 'The pre-grasp pose is inside the crate contents'
            job.note(text, False); job.goal = dict(reachable=False, reason=reason, text=text); return
        job.note('Pre-grasp joint goal: nearest of %d IK solutions, %.2f rad from the capture pose%s' % (best['found'], best['direct'], ', wrist rolled 180°' if best['flip'] else ''), True)
        job.goal = dict(reachable=True, tilt_x=dx, tilt_y=dy, flip=best['flip'], goal_joints=best['q'], direct=round(best['direct'], 4), start=CAPTURE, grasp=pb.pose_to_list(best['pose']), pre=pb.pose_to_list(best['pre']), resolve_s=round(time.time()-job.t0, 2))
        query = dict(move='pre_grasp', start_joints=CAPTURE, goal_joints=best['q'], goal_pose=pb.pose_to_list(best['pre']), pre_grasp_pose=pb.pose_to_list(best['pre']), direct_joint_distance=best['direct'])
        pb.RETIME_VELOCITY_SCALING = pb.RETIME_ACCELERATION_SCALING = r['speed']
        wanted = [p for p in PLANNERS if p['id'] in r['planners']]
        for rep in range(r['repeats']):
            for p in wanted:
                if self.superseded(job): return
                self.set_simplify(p['simplify'])
                args = types.SimpleNamespace(planning_time=r['planning_time'], num_planning_attempts=r['attempts'] if p['simplify'] else 1, goal_type='joint')
                out = dict(planner=p['id'], repeat=rep, success=False)
                try:
                    row, wp = pb.plan_query(bench, p['spec'], query, args)
                    code = row.get('error_code')
                    out.update(success=row['success'], plan_s=row['planning_time_s'], code=code if code != '' else None, why=row.get('pink_reason') or ('' if row['success'] else ERRORS.get(code, 'code %s' % code)))
                    if wp:
                        m, path = pb.path_metrics(bench, wp, query, columns, cell_m=cell, floor_z=floor)
                        out.update(metrics={k: clean(m.get(k)) for k in METRICS}, t=path['t'], q=path['waypoints'])
                except Exception as e:  # one bad plan must not stop the others
                    out.update(why='%s: %s' % (type(e).__name__, str(e)[:200]))
                job.results.append(out)
        self.set_simplify(True)


def number(body, key, default, lo, hi):
    v = float(body.get(key, default))
    if not math.isfinite(v) or not lo <= v <= hi: raise ValueError('%s must be between %g and %g' % (key, lo, hi))
    return v


def parse(body, catalog):
    x0, x1, y0, y1, *_ = catalog.interior; m = 0.05
    if body.get('scan') not in catalog.scans: raise ValueError('unknown scan')
    d = DEFAULTS; orientation = body.get('orientation', d['orientation']); carve = body.get('carve', d['carve'])
    if orientation not in ('tilted', 'topdown', 'manual'): raise ValueError('unknown orientation')
    if carve not in ('gripper', 'cylinder'): raise ValueError('unknown carve mode')
    ids = {p['id'] for p in PLANNERS}; planners = [p for p in body.get('planners', list(ids)) if p in ids]
    return dict(scan=body['scan'], x=number(body, 'x', 0, x0-m, x1+m), y=number(body, 'y', 0, y0-m, y1+m), z=number(body, 'z', 0, -0.02, 0.3), yaw=number(body, 'yaw', 0, -360, 720),
                orientation=orientation, carve=carve, tilt_x=int(number(body, 'tilt_x', 0, -80, 80)), tilt_y=int(number(body, 'tilt_y', 0, -80, 80)), contents=bool(body.get('contents', d['contents'])),
                planning_time=number(body, 'planning_time', d['planning_time'], .1, 20), attempts=int(number(body, 'attempts', d['attempts'], 1, 200)), repeats=int(number(body, 'repeats', d['repeats'], 1, 5)),
                speed=number(body, 'speed', d['speed'], .05, 1), ik_seeds=d['ik_seeds'], planners=planners)


def handler(catalog, planner, finger_m, tcp_z):
    class H(BaseHTTPRequestHandler):
        protocol_version = 'HTTP/1.1'
        def log_message(self, *a): pass
        def send(self, code, body, kind='application/json'):
            if not isinstance(body, bytes): body = json.dumps(body, separators=(',', ':')).encode()
            self.send_response(code); self.send_header('Content-Type', kind); self.send_header('Content-Length', str(len(body))); self.send_header('Cache-Control', 'no-store'); self.end_headers(); self.wfile.write(body)
        def do_GET(self):
            u = urlparse(self.path); parts = [x for x in u.path.split('/') if x]
            if not parts: return self.send(200, open(os.path.join(HERE, 'site', 'index.html'), 'rb').read(), 'text/html; charset=utf-8')
            if parts == ['api', 'meta']: return self.send(200, dict(scans=catalog.summary(), interior=catalog.interior, floor=catalog.floor, planners=[p['id'] for p in PLANNERS], defaults=DEFAULTS, finger_m=finger_m, tcp_z=tcp_z, start=CAPTURE))
            if len(parts) == 3 and parts[:2] == ['api', 'scan'] and parts[2] in catalog.scans: return self.send(200, catalog.page_data(parts[2]))
            if len(parts) == 3 and parts[:2] == ['api', 'image'] and parts[2] in catalog.scans and catalog.scans[parts[2]]['image']: return self.send(200, open(catalog.scans[parts[2]]['image'], 'rb').read(), 'image/png')
            if len(parts) == 3 and parts[:2] == ['api', 'job'] and parts[2].isdigit():
                j = planner.jobs.get(int(parts[2]))
                if j is None: return self.send(404, dict(error='unknown job'))
                since = int(parse_qs(u.query).get('since', ['0'])[0])
                return self.send(200, dict(id=j.id, state=j.state, trace=j.trace, goal=j.goal, error=j.error, n=len(j.results), results=j.results[since:], elapsed=round(time.time()-j.t0, 2)))
            self.send(404, dict(error='not found'))
        def do_POST(self):
            if self.path != '/api/plan': return self.send(404, dict(error='not found'))
            try: req = parse(json.loads(self.rfile.read(int(self.headers.get('Content-Length', 0)))), catalog)
            except (ValueError, KeyError, TypeError) as e: return self.send(400, dict(error=str(e)))
            self.send(200, dict(job=planner.submit(req)))
    return H


def main():
    a = argparse.ArgumentParser(); a.add_argument('--port', type=int, default=8765); a = a.parse_args()
    master = os.environ.get('ROS_MASTER_URI', 'http://localhost:11311')
    if master.rstrip('/').endswith(':11311'): raise SystemExit('Refusing to run on the robot\'s ROS master (11311). Use run_live.sh, which starts its own simulator.')
    catalog = Catalog(); planner = Planner(catalog); tcp = planner.bench.tcp; finger = round(tcp-0.1034, 3)
    srv = ThreadingHTTPServer(('127.0.0.1', a.port), handler(catalog, planner, finger, tcp))
    print('Live planner ready: open http://localhost:%d (fingers %.0f cm, ROS master %s)' % (a.port, finger*100, master), flush=True)
    try: srv.serve_forever()
    except KeyboardInterrupt: pass

if __name__ == '__main__': main()
