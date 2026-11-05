"""H.264 copies of the side clips of the E4 runs that have video, for the page. argv[1] = compiled h264_transcode."""
import json, os, subprocess, sys
from concurrent.futures import ThreadPoolExecutor
from e4data import *
sync = json.load(open('sync.json'))
def one(v):
    o = 'site/clips/'+v.replace('_RRTConnect', '').replace('.mp4', '')+'.mp4'
    if os.path.exists(o) and os.path.getsize(o) > 0: return o
    subprocess.run([sys.argv[1], os.path.join(VIDEOS, v), o, '640', '28'], check=True, capture_output=True); return o
with ThreadPoolExecutor(4) as ex: fs = list(ex.map(one, sorted(sync)))
print('done', len(fs), sum(os.path.getsize(f) for f in fs)/1e6, 'MB')
