"""Find when each logged move happens inside its video clip: maximise the overlap of the frame-difference motion
signal with the logged joint speed. The log only has 1 s timestamps. Writes sync.json."""
import json, os, sys, datetime as D
import cv2, numpy as np
from e4data import *

def motion(path):
    v = cv2.VideoCapture(path); fps = v.get(5) or 30; prev = None; e = []
    while True:
        ok, f = v.read()
        if not ok: break
        g = cv2.GaussianBlur(cv2.cvtColor(cv2.resize(f, (160, 120)), cv2.COLOR_BGR2GRAY), (5, 5), 0).astype(np.float32)
        e.append(0 if prev is None else float(np.mean(np.abs(g-prev) > 8)))
        prev = g
    v.release(); return np.array(e), fps

def speed(rec, dt):
    t, q = executed_q(rec)
    T = t[-1]; n = max(2, int(T/dt)+1); tt = np.arange(n)*dt
    Q = np.stack([np.interp(tt, t, q[:, j]) for j in range(7)], 1)
    s = np.r_[0, np.abs(np.diff(Q, axis=0)).sum(1)]/dt
    return s, T

out = {}
C = clips_by_run()
for run, d in sorted(C.items()):
    for vid, recs in sorted(d['clips'].items()):
        e, fps = motion(os.path.join(VIDEOS, vid)); dt = 1/fps
        e_s = np.convolve(e, np.ones(3)/3, 'same'); base = np.percentile(e_s, 20); sig = np.clip(e_s-base, 0, None)
        t0 = clip_time(vid); N = len(sig); offs = np.arange(-2.0, e.size*dt, dt); res = []
        # Dynamic programme over the moves of a clip: keep their logged order, no overlap, near the 1 s log time,
        # and maximise the overlap of joint speed and image motion; moves may run off either end of the clip.
        prev_best = None; back = []
        for rec in recs:
            s, T = speed(rec, dt); nominal = (D.datetime.strptime(rec['time'], '%Y-%m-%d %H:%M:%S')-t0).total_seconds()
            weight = max(float(s.sum()*dt), 1e-3); sc = np.full(len(offs), -1e3); raw = np.zeros(len(offs))
            for k, off in enumerate(offs):
                if not nominal-2.5 <= off <= nominal+1.0: continue
                i = int(round(off/dt)); a = max(0, i); b = min(N, i+len(s))
                if b-a < 2: sc[k] = 0.0; continue
                seg = sig[a:b]; ss = s[a-i:b-i]; raw[k] = (seg*ss).sum()/(np.linalg.norm(seg)*np.linalg.norm(s)+1e-9)
                sc[k] = weight*raw[k]
            if prev_best is None:
                tot = sc.copy(); bk = -np.ones(len(offs), int)
            else:
                shift = int(round((prev_T-0.08)/dt)); tot = np.full(len(offs), -1e6); bk = -np.ones(len(offs), int)
                bi = 0
                for k in range(len(offs)):
                    j = k-shift
                    if j < 0: continue
                    if prev_best[j] > prev_best[bi]: bi = j
                    tot[k] = prev_best[bi]+sc[k]; bk[k] = bi
            back.append((bk, raw, rec, nominal, T, s)); prev_best = tot; prev_T = T
        k = int(np.argmax(prev_best)); seq = []
        for bk, raw, rec, nominal, T, s in reversed(back):
            seq.append((k, raw[k], rec, nominal, T, s)); k = bk[k]
        for k, sco, rec, nominal, T, s in reversed(seq):
            res.append(dict(movement=rec['movement'], kind=rec['kind'], off=round(float(offs[k]), 3), score=round(float(sco), 3), nominal=nominal, T=round(T, 3), moved=float(s.mean()) > 0.02, travel=round(float(s.sum()*dt), 3)))
        out[vid] = dict(run=run, fps=fps, dur=len(e)/fps, moves=res)
    print(run, len(d['clips']), flush=True)
json.dump(out, open('sync.json', 'w'))
sc = np.array([m['score'] for c in out.values() for m in c['moves'] if m['travel']>.2]); print('moves', len(sc), 'score median %.2f p10 %.2f; <0.5: %d' % (np.median(sc), np.percentile(sc, 10), (sc < .5).sum()))
d = np.array([m['off']-m['nominal'] for c in out.values() for m in c['moves'] if m['moved']]); print('off-nominal p5,50,95', np.percentile(d, [5, 50, 95]).round(2))
for mv in sorted({m['movement'] for c in out.values() for m in c['moves']}):
    x = [m['score'] for c in out.values() for m in c['moves'] if m['movement'] == mv and m['moved']]; print(mv, len(x), round(float(np.median(x)), 2) if x else None)
