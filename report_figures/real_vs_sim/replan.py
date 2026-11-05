#!/usr/bin/env python3
"""Plan every real pre-grasp and transport (lift -> above place) move of the E4 runs with video again with the other planners, in the sim (nothing is
executed): same start and goal joints as the logged plan, the logged crate-contents height map as the obstacle for pre-grasp (none for transport), and
the benchmark's own plan_query/path_metrics, so the results compare with the thesis planner data.
Appends to experiments/real_vs_sim_replans/plans.jsonl (one line per plan) and resumes from it.
Needs a sim on its own ROS master (run_replans.sh starts one)."""
import argparse, json, os, random, sys, time, types
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, os.path.join(ROOT, 'src/learning_approach_to_robotic_grasping_of_vine_tomatoes/grasp/src'))
sys.path.insert(0, HERE)
from e4data import clips_by_run, planned_q, EXP  # noqa: E402

MAIN = ['RRTConnect', 'BiTRRT', 'RRTstar@1', 'RRTstar@5', 'chomp', 'pilz_ptp', 'pilz_lin', 'pink']
RAW = ['RRTConnect', 'BiTRRT']  # OMPL output without MoveIt's path simplification

MOVES = {'pre_grasp': 'grasp_pre_grasp', 'transport': 'go_to_place_above'}  # the benchmark's move -> the robot's logged move

def queries(moves):
    wanted = {MOVES[m]: m for m in moves}; out = []
    for run, d in sorted(clips_by_run().items()):
        for vid, recs in sorted(d['clips'].items()):
            for r in recs:
                if r['movement'] not in wanted or not r.get('trajectory'): continue
                _, q = planned_q(r); move = wanted[r['movement']]
                # Transport is planned without the crate contents, as in drop_off() and the benchmark
                out.append(dict(run=run, video=vid, attempt=r['_attempt'], move=move, start=q[0].tolist(), goal=q[-1].tolist(), scene=r.get('scene') if move == 'pre_grasp' else None))
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--limit', type=int); ap.add_argument('--moves', nargs='+', default=['pre_grasp', 'transport'], choices=sorted(MOVES)); ap.add_argument('--repeats', type=int, default=5); ap.add_argument('--out', default=os.path.join(EXP, 'real_vs_sim_replans', 'plans.jsonl'))
    a = ap.parse_args()
    import rospy
    from scipy.spatial.transform import Rotation as R
    from moveit_msgs.msg import CollisionObject  # noqa: F401
    rospy.init_node('real_vs_sim_replan')
    from move_robot.planner_benchmark import Bench, plan_query, path_metrics, set_ompl_simplify
    from common.crate_contents import columns_to_collision_object, CRATE_CONTENTS_ID
    bench = Bench('panda'); bench.tcp = bench.tcp_offset_m()
    if abs(bench.tcp-0.2034) > 1e-3: raise SystemExit('sim TCP is %.4f m; the E4 robot has 0.2034 m (10 cm fingers)' % bench.tcp)
    os.makedirs(os.path.dirname(a.out), exist_ok=True); done = set()
    if os.path.exists(a.out):
        for l in open(a.out):
            try: x = json.loads(l); done.add((x['video'], x['planner'], x['repeat'], x['simplify']))
            except ValueError: pass
    Q = queries(a.moves)[:a.limit]; planner = bench.pink.get_planner(bench.move_group); rng = random.Random(0)
    prev = set_ompl_simplify(bench, True); t0 = time.time()
    def set_scene(q):
        if CRATE_CONTENTS_ID in bench.scene.get_known_object_names(): bench.scene.remove_world_object(CRATE_CONTENTS_ID); rospy.sleep(.2)
        s = q['scene']
        if not s: return None, 0.02, 0.0
        cols = [tuple(c) for c in s['columns']]
        bench.scene.add_object(columns_to_collision_object(cols, s['floor_z'], s['cell_m'], bench.frame))
        for _ in range(60):
            if CRATE_CONTENTS_ID in bench.scene.get_known_object_names(): break
            rospy.sleep(.05)
        return cols, s['cell_m'], s['floor_z']
    try:
        with open(a.out, 'a') as out:
            for qi, q in enumerate(Q):
                cols, cell, floor = set_scene(q)
                tcp = planner.frame_pose(q['goal'], 'panda_hand_tcp')
                goal_pose = list(np.asarray(tcp.translation).tolist())+R.from_matrix(np.asarray(tcp.rotation)).as_quat().tolist()
                query = dict(move=q['move'], start_joints=q['start'], goal_joints=q['goal'], goal_pose=goal_pose, pre_grasp_pose=goal_pose,
                             direct_joint_distance=float(np.linalg.norm(np.array(q['goal'])-np.array(q['start']))))
                stats = {}
                for simplify, plist, att in ((True, MAIN, 100), (False, RAW, 1)):  # 100 attempts as plan_movement.py and the benchmark; raw as run_planner_quality.sh
                    set_ompl_simplify(bench, simplify)
                    args = types.SimpleNamespace(planning_time=5.0, num_planning_attempts=att, goal_type='joint')
                    for rep in range(a.repeats):
                        order = list(plist); rng.shuffle(order)
                        for p in order:
                            if rospy.is_shutdown(): return
                            key = (q['video'], p, rep, simplify)
                            if key in done: continue
                            rec = dict(video=q['video'], run=q['run'], attempt=q['attempt'], move=q['move'], planner=p, repeat=rep, simplify=simplify, success=False)
                            try:
                                row, wp = plan_query(bench, p, query, args)
                                rec.update(success=bool(row['success']), error=row.get('error_code', ''), why=row.get('pink_reason', ''), plan_s=row['planning_time_s'], planner_s=row.get('planner_time_s', ''))
                                if wp:
                                    m, path = path_metrics(bench, wp, query, cols, cell_m=cell, floor_z=floor)
                                    rec.update(metrics=m, waypoints=path['waypoints'], t=path['t'])
                            except Exception as e:  # one bad plan must not stop the run; recorded as failed
                                rec.update(error='exception', why=str(e)[:200])
                            out.write(json.dumps(rec)+'\n'); out.flush()
                            stats[p+('' if simplify else '·raw')] = stats.get(p+('' if simplify else '·raw'), 0)+int(rec['success'])
                print('[%d/%d] %.0fs run %d attempt %d ' % (qi+1, len(Q), time.time()-t0, q['run'], q['attempt'])+' '.join('%s %d' % kv for kv in sorted(stats.items())), flush=True)
    finally:
        try:
            if CRATE_CONTENTS_ID in bench.scene.get_known_object_names(): bench.scene.remove_world_object(CRATE_CONTENTS_ID)
            set_ompl_simplify(bench, prev)
        except Exception: pass
    print('finished', flush=True)

if __name__ == '__main__': main()
