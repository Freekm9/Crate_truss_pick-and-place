#!/usr/bin/env python3
"""Why do planners fail on low picks? Same pick with the crate contents, without them, and 5 cm higher, on samples of the low packed,
low edge and October points; plus what the straight joint line collides with. Sim only, on its own ROS master. Appends JSON lines to OUT."""
import json, os, random, sys, time, types
import numpy as np
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
G = os.path.join(ROOT, 'src/learning_approach_to_robotic_grasping_of_vine_tomatoes/grasp/src'); sys.path[:0] = [G, os.path.join(G, 'stem_segmentation')]
EXP = os.path.join(ROOT, 'experiments'); OUT = os.path.join(EXP, 'hard_points', 'why_low_fails.jsonl'); N = int(sys.argv[1]) if len(sys.argv) > 1 else 20
CAPTURE = [-0.17830975928446877, -0.6060327979481354, 0.4456400372147912, -2.5271529010268137, 0.31681683511204184, 1.960740570757357, 0.8233889236590929]
PLANNERS = ['RRTConnect', 'pilz_ptp', 'pilz_lin', 'chomp']
import rospy
from moveit_msgs.msg import RobotState
from moveit_msgs.srv import GetPositionIK, GetStateValidity, GetStateValidityRequest
rospy.init_node('why_low_fails', disable_signals=True)
from move_robot import planner_benchmark as pb, test_crate_reachability as tcr
from pcd_io import load_pcd
from common.grasp_motion import PRE_GRASP_GRIPPER_WIDTH_M
bench = pb.Bench('panda'); bench.tcp = bench.tcp_offset_m(); ik = tcr.IKChecker('panda'); iks = rospy.ServiceProxy(bench.ns+'/compute_ik', GetPositionIK); valid = rospy.ServiceProxy(bench.ns+'/check_state_validity', GetStateValidity)
contents = bench.contents_obstacle('gripper'); print('TCP %.4f' % bench.tcp, flush=True)
rng = random.Random(0); low = json.load(open(os.path.join(EXP, 'hard_points', 'points_low.json')))['points']; old = json.load(open(os.path.join(EXP, 'finger_reachability', 'points_20261014_120918.json')))['points']
sets = [('packed', rng.sample([p for p in low if p['layout'] == 'packed'], N)), ('edge', rng.sample([p for p in low if p['layout'] == 'edge'], N)), ('october', rng.sample(old, N))]
clouds = {}
def cloud(p):
    if p['cloud'] not in clouds: clouds[p['cloud']] = load_pcd(p['cloud'])[0]
    return clouds[p['cloud']]
def contacts(q0, q1):
    """First collision along the straight joint line, as MoveIt's own check reports it."""
    r = GetStateValidityRequest(); r.group_name = bench.move_group.get_name(); s = RobotState(); s.is_diff = True; fingers = ['panda_finger_joint1', 'panda_finger_joint2']; s.joint_state.name = list(bench.joint_names)+fingers
    for f in np.linspace(0, 1, 41):
        s.joint_state.position = list(np.array(q0)*(1-f)+np.array(q1)*f)+[PRE_GRASP_GRIPPER_WIDTH_M/2]*2; r.robot_state = s; a = valid(r)
        if not a.valid: return dict(at=round(float(f), 3), pairs=sorted({tuple(sorted((c.contact_body_1, c.contact_body_2))) for c in a.contacts})[:6])
    return None
def case(p, name, z, use):
    rec = dict(set=name[0], cond=name[1], scan=p['scan'], id=p['id'], x=p['x'], y=p['y'], z=z, wall=p['wall_dist_m'], plans={})
    base = tcr.build_straight_down_pose(p['x'], p['y'], z, p['yaw_stem_deg'], bench.frame); ok, dx, dy, flip = tcr.find_reachable_tilt(ik, base); rec.update(orient=[dx, dy, flip])
    if not ok: rec['stop'] = 'no_orientation'; return rec
    grasp = tcr.flip_approach_roll(base) if flip else tcr.tilt_pose(base, dx, dy) if (dx or dy) else base; cands = [grasp]; roll = tcr.flip_approach_roll(grasp)
    if ik.check_grasp(roll): cands.append(roll)
    app = tcr.get_approach_vec(grasp.pose); cols = None; floor = 0.0
    if use:
        rec['columns'] = contents.add([p['x'], p['y'], z], app, xyz=cloud(p), grasp_quat=pb.pose_to_list(grasp)[3:]); cols, floor = list(contents.columns), contents.floor_z
        c = np.array(cols); d = np.hypot(c[:, 0]-p['x'], c[:, 1]-p['y']); rec['top_near_6cm'] = round(float(c[d < .06, 2].max()), 4) if (d < .06).any() else None; rec['top_max'] = round(float(c[:, 2].max()), 4)
    best = None
    for g in cands:
        q, dist, _ = pb.nearest_ik(bench, iks, tcr.offset_pose(g, app, tcr.PRE_GRASP_OFFSET_M), CAPTURE, 30)
        if q is not None and (best is None or dist < best[1]): best = (q, dist, tcr.offset_pose(g, app, tcr.PRE_GRASP_OFFSET_M))
    if best is None: rec['stop'] = 'goal_in_contents'; contents.remove(); return rec
    rec['direct'] = round(best[1], 3); rec['contact'] = contacts(CAPTURE, best[0])
    query = dict(move='pre_grasp', start_joints=CAPTURE, goal_joints=best[0], goal_pose=pb.pose_to_list(best[2]), pre_grasp_pose=pb.pose_to_list(best[2]), direct_joint_distance=best[1])
    for sp in PLANNERS:
        try:
            row, wp = pb.plan_query(bench, sp, query, types.SimpleNamespace(planning_time=5.0, num_planning_attempts=100, goal_type='joint')); e = dict(ok=row['success'], s=row['planning_time_s'], code=row['error_code'])
            if wp: m, _ = pb.path_metrics(bench, wp, query, cols, cell_m=contents.cell_m, floor_z=floor); e.update(ratio=m['path_ratio'], tcp=m['ee_path_length_m'], truss=None if not np.isfinite(m['min_contents_clearance_mm']) else m['min_contents_clearance_mm'])
        except Exception as x: e = dict(ok=False, code='exception', why=str(x)[:120])
        rec['plans'][sp] = e
    if use: contents.remove()
    return rec
t0 = time.time(); done = 0
with open(OUT, 'a') as f:
    for name, pts in sets:
        for p in pts:
            for cond, z, use in (('as_is', p['z'], True), ('no_contents', p['z'], False), ('raised_5cm', p['z']+.05, True)):
                try: r = case(p, (name, cond), z, use)
                except Exception as x: r = dict(set=name, cond=cond, scan=p['scan'], id=p['id'], stop='exception: %s' % str(x)[:160]); contents.remove()
                f.write(json.dumps(r)+'\n'); f.flush(); done += 1
            print('%d/%d %.0fs %s' % (done, 9*N, time.time()-t0, name), flush=True)
print('finished', flush=True)
