"""Writes site/data for the real-vs-simulated page: index.json (runs, clips, summary) and one file per E4 run."""
import csv, json, os, statistics as st
import numpy as np
from e4data import *
sync = json.load(open('sync.json')); met = json.load(open('metrics.json')); cam = json.load(open('camera.json'))
C = clips_by_run()
def enc(t, q):
    t = np.asarray(t, float); q = np.asarray(q, float)
    keep = np.r_[True, np.diff(t) > 1e-4]; t = t[keep]; q = q[keep]
    ti = np.round(t*1000).astype(int); qi = np.round(q*1e4).astype(int)
    return dict(t=np.diff(ti, prepend=0).tolist(), q=np.diff(qi, axis=0, prepend=np.zeros((1, 7), int)).ravel().tolist())
att = {}
with open(os.path.join(EXP, 'e4', 'attempts.csv')) as f:
    for r in csv.DictReader(f):
        m = C.get(int(r['run_id']))
        if m and m['meta']['start'] <= r['timestamp'] <= m['meta']['end']: att[(int(r['run_id']), int(r['attempt']))] = r
TYPE = {'grasp': 'Grasp', 'drop_off': 'Drop-off', 'go_to_saved_pose': 'Return'}
def ctype(v):
    return next(k for k in TYPE if ('_'+k+'_side') in v)
idx = []
for run, d in sorted(C.items()):
    clips = []
    for vid, recs in sorted(d['clips'].items()):
        s = sync[vid]; ms = met['moves'][vid]; moves = []
        for r, sm, m in zip(recs, s['moves'], ms):
            te, qe = executed_q(r); tp, qp = planned_q(r); nominal = False
            if tp is None:  # a 'go' move logs only its goal: straight joint line over the measured duration
                g = r.get('target'); nominal = True; tp = np.array([0, te[-1]]); qp = np.array([qe[0], [g[j] for j in JN] if g else qe[-1]])
            moves.append(dict(m=r['movement'], k=r['kind'], off=sm['off'], sc=sm['score'], tr=sm['travel'], nom=nominal, ok=bool(r.get('success')),
                              plan=enc(tp, qp), real=enc(te, qe), x=None if m is None else {k: round(m[k], 4) for k in ('T_plan', 'T_settle', 'delay', 'gap_max', 'gap_delay_max', 'shape_max', 'end_err')}))
        a = att.get((run, recs[0]['_attempt'])); scene = next((r['scene'] for r in recs if r.get('scene')), None)
        c = dict(v=vid, f=vid.replace('_RRTConnect', ''), dur=s['dur'], ty=ctype(vid), att=recs[0]['_attempt'], moves=moves)
        if scene: c['scene'] = dict(c=[[int(round(a_*1000)) for a_ in col] for col in scene['columns']], fl=scene['floor_z'], cell=scene['cell_m'])
        if a and c['ty'] == 'grasp': c['g'] = dict(x=float(a['x']), y=float(a['y']), z=float(a['z']), tx=float(a['tilt_x_deg']), ty=float(a['tilt_y_deg']), yaw=float(a['yaw_deg']), pick=a['pick_result'], rem=a['removed'])
        clips.append(c)
    os.makedirs('site/data', exist_ok=True)
    with open('site/data/run%02d.json' % run, 'w') as f: json.dump(dict(id=run, clips=clips), f, separators=(',', ':'))
    me = d['meta']; idx.append(dict(id=run, cond=me['condition'], method=me['method'], rep=int(me['replicate']), start=me['start'], clips=len(clips), trusses=int(me['trusses_start']), removed=int(me['removed'])))
rows = met['rows']
def mp(x): return [round(float(np.median(x)), 4), round(float(np.percentile(x, 95)), 4)]
ORDER = [('grasp_pre_grasp', 'Capture → pre-grasp', 'planner'), ('grasp_descent', 'Descent', 'cartesian'), ('grasp_back-out', 'Back-out', 'cartesian'), ('grasp_lift', 'Lift', 'cartesian'),
         ('go_to_place_above', 'To above place', 'cartesian'), ('go_to_place', 'Lower to place', 'cartesian'), ('go_to_place_retreat', 'Retreat', 'cartesian'), ('go_to_saved_pose', 'Return to capture', 'cartesian')]
summary = []
for key, label, kind in ORDER:
    x = [r for r in rows if r['movement'] == key]
    summary.append(dict(key=key, label=label, kind=kind, n=len(x), **{k: mp([r[k] for r in x]) for k in ('T_plan', 'T_settle', 'delay', 'gap_max', 'gap_delay_max', 'shape_max', 'end_err')}))
# Simulation benchmark timing (10 cm, RRT-Connect, typical goals) vs the robot's plans
bench = {}
with open(os.path.join(EXP, 'planner_benchmark', 'quality_20261020', 'typical', 'results_20261020_111920.csv')) as f:
    for r in csv.DictReader(f):
        if r['planner'] == 'RRTConnect' and r['exec_time_s']:
            bench.setdefault(r['move'], []).append((float(r['exec_time_s']), float(r['ee_path_length_m'])))
pre = [r for r in rows if r['movement'] == 'grasp_pre_grasp']; ab = [r for r in rows if r['movement'] == 'go_to_place_above']; pl = [r for r in rows if r['movement'] == 'go_to_place']
real_trans_t = [a['T_settle'] for a in ab]; real_trans_p = [a['T_plan'] for a in ab]; real_trans_l = [a['len_real'] for a in ab]  # the benchmark's transport ends above the place pose: go_to_place_above only
bc = dict(pre=dict(sim_t=mp([b[0] for b in bench['pre_grasp']]), sim_l=mp([b[1] for b in bench['pre_grasp']]), plan_t=mp([r['T_plan'] for r in pre]), real_t=mp([r['T_settle'] for r in pre]), real_l=mp([r['len_real'] for r in pre])),
          trans=dict(sim_t=mp([b[0] for b in bench['transport']]), sim_l=mp([b[1] for b in bench['transport']]), plan_t=mp(real_trans_p), real_t=mp(real_trans_t), real_l=mp(real_trans_l)))
with open('site/data/index.json', 'w') as f:
    json.dump(dict(runs=idx, summary=summary, bench=bc, camera=cam, tcp_z=TCP_Z, n_moves=len(rows), n_weak=sum(1 for c in sync.values() for m in c['moves'] if m['travel'] > .2 and m['score'] < .6)), f, separators=(',', ':'))
print('runs', [r['id'] for r in idx], 'bytes', sum(os.path.getsize('site/data/'+x) for x in os.listdir('site/data')))
print(json.dumps(bc)); print(json.dumps(summary[0]))
# Other planners on the same start/goal (replan.py): one file per grasp clip, summary per planner
PLANS = os.path.join(EXP, 'real_vs_sim_replans', 'plans.jsonl')
if os.path.exists(PLANS):
    by = {}
    with open(PLANS) as f:
        for l in f:
            try: x = json.loads(l)
            except ValueError: continue
            by.setdefault(x['video'], []).append(x)
    by = {k: v for k, v in by.items() if len(v) >= 50}  # 10 planner variants x 5 repeats: a move still being planned waits
    os.makedirs('site/data/plans', exist_ok=True)
    def pid(x): return x['planner']+('' if x['simplify'] else '·raw')
    robot = {};
    for d in C.values():
        for vid, recs in d['clips'].items():
            pres = [r for r in recs if r['movement'] in ('grasp_pre_grasp', 'go_to_place_above') and r.get('trajectory')]
            assert len(pres) <= 1, vid
            if pres:
                t, q = planned_q(pres[0]); robot[vid] = np.array([fk(v)[2][:3, 3] for v in q[np.linspace(0, len(q)-1, min(len(q), 40)).astype(int)]])
    stats = {}; moves_done = {}
    def finite(v): return v if isinstance(v, (int, float)) and np.isfinite(v) else None  # NaN is not valid JSON for the browser
    for vid, xs in by.items():
        out = {}
        for x in xs:
            e = out.setdefault(pid(x), dict(paths=[], fail=[])); mv = x.get('move', 'pre_grasp'); moves_done.setdefault(mv, set()).add(vid); st_ = stats.setdefault((mv, pid(x)), dict(n=0, ok=0, exec=[], pt=[], tcp=[], ratio=[], far=[], wall=[], truss=[])); st_['n'] += 1; st_['pt'].append(x.get('plan_s') or 0)
            if x['success'] and x.get('waypoints') and len(x['waypoints']) == len(x['t']):
                m = x['metrics']; p = enc(x['t'], x['waypoints'])
                p.update(r=x['repeat'], pt=x.get('plan_s'), m={k: finite(m.get(k)) for k in ('exec_time_s', 'ee_path_length_m', 'joint_path_length', 'min_clearance_mm', 'min_contents_clearance_mm', 'goal_deviation_rad')})
                e['paths'].append(p); st_['ok'] += 1; st_['exec'].append(m['exec_time_s']); st_['tcp'].append(m['ee_path_length_m'])
                if m.get('path_ratio') != '': st_['ratio'].append(m['path_ratio'])
                W = np.array(x['waypoints']); W = W[np.linspace(0, len(W)-1, min(len(W), 40)).astype(int)]; Pa = np.array([fk(v)[2][:3, 3] for v in W]); Rb = robot[vid]
                st_['far'].append(float(np.max(np.min(np.linalg.norm(Pa[:, None]-Rb[None], axis=2), axis=1))))
                for k, kk in (('min_clearance_mm', 'wall'), ('min_contents_clearance_mm', 'truss')):
                    if isinstance(m.get(k), (int, float)) and np.isfinite(m[k]): st_[kk].append(m[k])
            else:
                e['fail'].append(dict(r=x['repeat'], err=x.get('error', ''), why=x.get('why', '')))
        with open('site/data/plans/'+vid.replace('_RRTConnect', '').replace('.mp4', '')+'.json', 'w') as f: json.dump(out, f, separators=(',', ':'))
    for r in idx:
        fn = 'site/data/run%02d.json' % r['id']; d = json.load(open(fn))
        for c in d['clips']: c['pl'] = c['v'] in by
        with open(fn, 'w') as f: json.dump(d, f, separators=(',', ':'))
    ORD = ['RRTConnect', 'BiTRRT', 'RRTstar@1', 'RRTstar@5', 'chomp', 'pilz_ptp', 'pilz_lin', 'pink', 'RRTConnect·raw', 'BiTRRT·raw']
    groups = []
    for mv in ('pre_grasp', 'transport'):
        rows_ = [dict(id=k, n=s['n'], ok=s['ok'], exec=mp(s['exec']) if s['exec'] else None, pt=mp(s['pt']), tcp=mp(s['tcp']) if s['tcp'] else None, ratio=mp(s['ratio']) if s['ratio'] else None, far=mp(s['far']) if s['far'] else None,
                     wall=round(min(s['wall']), 1) if s['wall'] else None, truss=round(min(s['truss']), 1) if s['truss'] else None) for k, s in (((k, stats[(mv, k)]) for k in ORD if (mv, k) in stats))]
        if rows_: groups.append(dict(key=mv, moves=len(moves_done.get(mv, ())), total=75, rows=rows_))
    pl = [r for g in groups for r in g['rows']]
    ix = json.load(open('site/data/index.json')); ix.update(planner_groups=groups, plan_repeats=5)
    with open('site/data/index.json', 'w') as f: json.dump(ix, f, separators=(',', ':'))
    print('plans for', len(by), 'clips;', os.popen('du -sh site/data/plans').read().strip()); print(*[(p['id'], '%d/%d' % (p['ok'], p['n']), p['exec'], p['tcp'], p['ratio'], p['far'], p['wall'], p['truss']) for p in pl], sep='\n')
# Round trip: decoded move ends must match the log
def dec(e):
    t = np.cumsum(e['t'])/1000; q = np.cumsum(np.array(e['q']).reshape(-1, 7), axis=0)/1e4; return t, q
err = 0
for r in idx:
    for c in json.load(open('site/data/run%02d.json' % r['id']))['clips']:
        recs = C[r['id']]['clips'][c['v']]
        for m, rc in zip(c['moves'], recs):
            t, q = dec(m['real']); _, qq = executed_q(rc); err = max(err, float(np.abs(q[-1]-qq[-1]).max()))
print('round trip max joint error rad', err)
