# Real vs simulated paths page

Side-camera video of the real robot next to a 3D twin following the logged MoveIt plan, on one clock, with the
measured arm overlaid, the gripper distance over time and per-move numbers.

Published at https://claude.ai/artifact/GKQxcijmVFMeQX7ZBLC17m (private; share from the page).

The twin replays the logged plan: it is not a physics simulation. The measured arm comes from the joint encoders
through the same kinematic model, so link or finger bending and calibration errors only show in the video.

Rebuild, in this order, from `experiments/trajectories` and `experiments/planning_metrics/videos`:

    python3 sync.py        # places each move in its video (log timestamps have 1 s resolution) -> sync.json
    python3 calib.py OUT.jpg   # side-camera pose from hand-picked image points -> camera.json + overlay check image
    python3 metrics.py     # planned vs measured per move -> metrics.json
    gcc -O2 h264_transcode.c -o h264_transcode $(pkg-config --cflags --libs libavformat libavcodec libavutil libswscale)
    python3 transcode_all.py ./h264_transcode   # browser-playable copies -> site/clips (git-ignored)
    ./run_replans.sh       # about 1 h: other planners on each real pre-grasp move, in an isolated sim (ROS master 11313),
                           # nothing executed -> experiments/real_vs_sim_replans/plans.jsonl (resumes when restarted)
    python3 build.py       # -> site/data (git-ignored); moves without all 50 plans yet are left out

Open locally: `cd site && python3 -m http.server`, then browse to http://localhost:8000.
