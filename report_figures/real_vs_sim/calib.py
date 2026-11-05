"""Side-camera pose from hand-picked image points (crate rim, robot base, fingertip/flange in two frames),
assuming the D435 colour intrinsics at 640x480. Writes camera.json; overlay check image to argv[1]."""
import json, sys, cv2, numpy as np
from e4data import *
sync = json.load(open('sync.json')); C = clips_by_run()
ALL = {v: r for d in C.values() for v, r in d['clips'].items()}
def q_at(vid, t):
    """Measured joints at clip time t, from the synced records."""
    recs, ms = ALL[vid], sync[vid]['moves']; last = None
    for r, m in zip(recs, ms):
        tt, q = executed_q(r)
        if t < m['off']: break
        last = np.array([np.interp(t-m['off'], tt, q[:, j]) for j in range(7)])
    if last is None:
        tt, q = executed_q(recs[0]); last = q[0]
    return last
FX = 612.0; K = np.array([[FX, 0, 320], [0, FX, 240], [0, 0, 1.]])
A = '20261026_142308_RRTConnect_grasp_side.mp4'; B = '20261027_110043_RRTConnect_grasp_side.mp4'
pa, _, ta = fk(q_at(A, 2.0)); pb, _, tb = fk(q_at(B, 2.0))
obj = [[.585, .28, .14], [.335, .28, .14], [.585, -.08, .14], [.335, -.08, .14], [.585, .28, .005], [.335, .28, .005],
       [0, 0, 0], ta[:3, 3], tb[:3, 3], pa[8], pb[8]]
img = [[222, 340], [385, 335], [240, 312], [378, 305], [244, 420], [381, 420],
       [523, 392], [358, 278], [337, 290], [352, 148], [335, 172]]
obj = np.array(obj, float); img = np.array(img, float)
ok, rv, tv = cv2.solvePnP(obj, img, K, None, flags=cv2.SOLVEPNP_SQPNP if hasattr(cv2, 'SOLVEPNP_SQPNP') else cv2.SOLVEPNP_ITERATIVE)
rv, tv = cv2.solvePnPRefineLM(obj, img, K, None, rv, tv)
pr, _ = cv2.projectPoints(obj, rv, tv, K, None); pr = pr[:, 0]
print('residual px', np.round(np.linalg.norm(pr-img, axis=1), 1), 'rms %.1f' % np.sqrt(((pr-img)**2).sum(1).mean()))
R, _ = cv2.Rodrigues(rv); cam = (-R.T @ tv).ravel(); fwd = R.T @ np.array([0, 0, 1.]); up = R.T @ np.array([0, -1., 0])
print('camera at', cam.round(3), 'forward', fwd.round(3), 'up', up.round(3))
json.dump(dict(pos=cam.tolist(), forward=fwd.tolist(), up=up.tolist(), fx=FX, w=640, h=480), open('camera.json', 'w'), indent=1)
# Overlay check: logged arm + TCP on frames while the robot moves
def frame(v, t):
    c = cv2.VideoCapture(os.path.join(VIDEOS, v)); c.set(1, int(t*30)); ok, f = c.read(); c.release(); return f
out = []
pick = []
for v in sorted(sync):
    if sync[v]['run'] in (5, 12) : pick.append(v)
tiles = []
for v in [pick[1], pick[2], pick[31 if len(pick) > 31 else -2], pick[-3], pick[-2]]:
    for m in sync[v]['moves'][:4]:
        if m['travel'] < .5: continue
        t = m['off']+m['T']*.5; f = frame(v, t); q = q_at(v, t)
        p, h, tcp = fk(q); P = np.vstack([p, tcp[:3, 3]]); ip, _ = cv2.projectPoints(P, rv, tv, K, None); ip = ip[:, 0].astype(int)
        for i in range(len(ip)-1):
            cv2.line(f, tuple(ip[i]), tuple(ip[i+1]), (0, 255, 0), 2)
        cv2.circle(f, tuple(ip[-1]), 5, (0, 0, 255), 2)
        cv2.putText(f, '%s %s %.2f' % (v[9:15], m['movement'], m['score']), (6, 470), 0, .45, (255, 255, 0), 1); tiles.append(cv2.resize(f, (400, 300)))
n = len(tiles)//3*3; tiles = tiles[:min(n, 12)]
cv2.imwrite(sys.argv[1], np.vstack([np.hstack(tiles[i:i+3]) for i in range(0, len(tiles), 3)]), [cv2.IMWRITE_JPEG_QUALITY, 85])
