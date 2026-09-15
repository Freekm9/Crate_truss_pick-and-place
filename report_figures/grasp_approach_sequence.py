"""
Grasp approach sequence for the TOP-DOWN orientation only (deg_x = deg_y = 0).

plan_movement.py's grasp() steps the gripper through several poses strung along
the approach vector (the TCP z-axis). This draws the gripper at each of them so
the pre-grasp -> grasp -> retreat motion is visible on the scanned truss:

    pre-grasp   PRE_GRASP_OFFSET_M   back along approach   (5 cm in the pipeline)
    grasp       at the picked point
    sink        SINK_OFFSET_M        further in            (1 cm; deepest point,
                                                            where the gripper closes)
    retreat     RETREAT_OFFSET_M     back along approach   (25 cm; pulling away)

Only the straight-down orientation is shown, on purpose -- the tilt/roll fan is
what grasp_tilt_pointcloud.py is for; here the orientation is fixed and the
POSITION is what changes.

The truss (point cloud, hand cuts, saturation/Z-band filters, display transform,
estimated yaw + approach flip) is reused verbatim from grasp_tilt_pointcloud.py,
so it looks identical to that figure. The offsets below are DISPLAY distances:
grasp_tilt_pointcloud.py enlarges the cloud CLOUD_SCALE x, so the real
centimetre offsets would be invisible / off-frame. OFFSET_DISPLAY_SCALE blows
them up while keeping the pipeline's ratios (pre : retreat = 1 : 5).
"""

import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.spatial.transform import Rotation as R

import grasp_tilt_pointcloud as gtp

# ----------------------------- CONFIG ---------------------------------

# Real offsets along the approach vector, metres, from simple_pick_point.py
# (PRE_GRASP_OFFSET_M / SINK_OFFSET_M / RETREAT_OFFSET_M). Positive = back
# along the approach vector (away from the truss); negative = further in.
REAL_OFFSETS_M = {
    "pre-grasp": 0.05,
    "grasp": 0.0,
    "sink": -0.01,
    "retreat": 0.25,
}

# Which phases to draw, in motion order. Drop "sink" for the bare
# pre-grasp / grasp / retreat story; keep it to show the deepest point.
SHOW_PHASES = ["pre-grasp", "grasp", "retreat"]

# The reused cloud is enlarged, so scale the offsets up to match for display.
# retreat ends up at REAL_OFFSETS_M["retreat"] * this many metres from the point.
OFFSET_DISPLAY_SCALE = 2.0

PHASE_COLORS = {
    "pre-grasp": "deepskyblue",
    "grasp": "gold",
    "sink": "orangered",
    "retreat": "blueviolet",
}

# Draw the gripper so its fingertip plane lands on the phase point (True), or
# so its palm/back does (False).
FINGERTIPS_ON_PHASE_POINT = True

DRAW_APPROACH_LINE = True  # dashed guide line through all phase points

FIGSIZE = (10, 9)
OUTPUT_PATH = "/tmp/claude-0/-root/3047a74c-ec3a-4b61-bbd6-9cab87991a9a/scratchpad/grasp_approach_sequence.png"

# ------------------------------------------------------------------------


def prepare_cloud():
    """Same pipeline as grasp_tilt_pointcloud.main(), up to the display cloud."""
    xyz, colors = gtp.load_pcd_ascii(gtp.PCD_PATH)
    xyz, colors = gtp.apply_cuts(xyz, colors, gtp.REMOVE_CUTS)
    xyz, colors = gtp.filter_by_saturation(xyz, colors, gtp.SATURATION_THRESHOLD)

    est_point, est_yaw = gtp.estimate_yaw_and_point(xyz, gtp.Z_BAND)
    grasp_point = gtp.GRASP_POINT_OVERRIDE if gtp.GRASP_POINT_OVERRIDE is not None else est_point
    yaw_deg = gtp.YAW_DEG_OVERRIDE if gtp.YAW_DEG_OVERRIDE is not None else est_yaw

    pivot = gtp.CLOUD_SCALE_PIVOT if gtp.CLOUD_SCALE_PIVOT is not None else grasp_point
    xyz = (xyz - pivot) * gtp.CLOUD_SCALE + pivot + gtp.CLOUD_TRANSLATION

    if gtp.MAX_POINTS_DISPLAYED is not None and len(xyz) > gtp.MAX_POINTS_DISPLAYED:
        rng = np.random.default_rng(0)
        idx = rng.choice(len(xyz), size=gtp.MAX_POINTS_DISPLAYED, replace=False)
        xyz, colors = xyz[idx], (colors[idx] if colors is not None else None)

    return xyz, colors, grasp_point, yaw_deg


def main():
    xyz, colors, grasp_point, yaw_deg = prepare_cloud()

    # Top-down base orientation: yaw about Z, then the same approach flip
    # grasp_tilt_pointcloud.py uses. tilt_pose(0, 0) would just return this.
    base_rot = R.from_euler("z", np.radians(yaw_deg))
    if gtp.FLIP_APPROACH_FOR_DISPLAY:
        base_rot = base_rot * R.from_euler("x", 180, degrees=True)
    approach_dir = base_rot.apply([0, 0, 1])
    approach_dir = approach_dir / np.linalg.norm(approach_dir)
    finger_axis = base_rot.apply([0, 1, 0])

    # Phase point = grasp_point shifted back along the approach vector, mirroring
    # simple_pick_point._offset_pose (position -= approach_vec * distance).
    phases = []
    for name in SHOW_PHASES:
        dist = REAL_OFFSETS_M[name] * OFFSET_DISPLAY_SCALE
        point = grasp_point - approach_dir * dist
        phases.append((name, point, REAL_OFFSETS_M[name]))

    fig = plt.figure(figsize=FIGSIZE)
    ax = fig.add_subplot(111, projection="3d")
    ax.computed_zorder = False

    ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c=colors, s=gtp.POINT_SIZE,
               alpha=0.6, depthshade=False, zorder=1)
    ax.scatter(*grasp_point, color="yellow", s=40, zorder=gtp.GRASP_ZORDER,
               label="grasp point")

    if DRAW_APPROACH_LINE:
        pts = np.array([p for _, p, _ in phases])
        lo = grasp_point - approach_dir * (max(REAL_OFFSETS_M[n] for n in SHOW_PHASES)
                                          * OFFSET_DISPLAY_SCALE)
        hi = grasp_point - approach_dir * (min(REAL_OFFSETS_M[n] for n in SHOW_PHASES)
                                          * OFFSET_DISPLAY_SCALE)
        ax.plot(*zip(lo, hi), color="dimgray", linewidth=1.0, linestyle="dashed",
                zorder=gtp.FORK_ZORDER - 1)

    # Nearer grippers over farther ones (few phases, but keep it consistent).
    cam = gtp.camera_dir(gtp.ELEV_VIEW, gtp.AZIM_VIEW)
    order = sorted(range(len(phases)), key=lambda i: phases[i][1] @ cam)
    handles = []
    for rank, i in enumerate(order):
        name, point, real_m = phases[i]
        color = PHASE_COLORS[name]
        fork_pos = point - approach_dir * gtp.PRONG_LENGTH if FINGERTIPS_ON_PHASE_POINT else point
        gtp.draw_gripper_fork(ax, fork_pos, approach_dir, finger_axis, color,
                              zorder=gtp.FORK_ZORDER + rank)
    for name, _, real_m in phases:  # legend in motion order, not draw order
        sign = "+" if real_m > 0 else ("" if real_m == 0 else "−")
        label = f"{name}  ({sign}{abs(real_m):g} m)" if real_m else f"{name}  (0 m)"
        handles.append(Line2D([0], [0], color=PHASE_COLORS[name], lw=3, label=label))

    # Auto-fit limits: cloud + every phase point, padded like the other figure.
    span_pts = np.vstack([xyz[:, :3], [p for _, p, _ in phases], [grasp_point]])
    lo, hi = span_pts.min(axis=0), span_pts.max(axis=0)
    m = (hi - lo) * gtp.AXIS_MARGIN_FRAC
    lo, hi = lo - m, hi + m
    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    ax.set_zlim(lo[2], hi[2])
    ax.set_box_aspect((hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]))

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title("Grasp approach sequence, top-down orientation\n"
                  f"({os.path.basename(gtp.PCD_PATH)}, offsets ×{OFFSET_DISPLAY_SCALE:g} "
                  f"for display, real ratio kept)",
                  fontsize=11, pad=20)
    ax.view_init(elev=gtp.ELEV_VIEW, azim=gtp.AZIM_VIEW)

    ax.legend(handles=handles, title="Phase  (offset along approach)",
              loc="upper left", fontsize=9, title_fontsize=9,
              bbox_to_anchor=(-0.02, 0.95))

    fig.subplots_adjust(top=0.85, bottom=0.05)
    plt.savefig(OUTPUT_PATH, dpi=200)
    print(f"Saved to {OUTPUT_PATH}")
    print(f"grasp point: {grasp_point}, yaw: {yaw_deg:.2f} deg, phases: {SHOW_PHASES}")


if __name__ == "__main__":
    main()
