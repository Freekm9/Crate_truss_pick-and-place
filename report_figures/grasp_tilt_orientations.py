"""
Visualizes the fallback orientation search from SimplePickPoint._tilt_pose()
in simple_pick_point.py: starting from a straight-down grasp, the approach
vector is tilted `deg_x` around the vine/stem axis (TCP local X), then the
resulting frame is tilted `deg_y` around the post-X-tilt gripper-closing axis
(TCP local Y). The rotation composition below mirrors that function exactly
(same scipy Rotation calls, same order), so this is not an approximation.

This is the cylinder-model twin of grasp_tilt_pointcloud.py: identical figure
styling (same sweep angles, fork linework, single-hue palettes, yellow grasp
marker, view, legends) -- only the scanned point cloud is swapped for an
idealized stem cylinder along the world X (vine) axis. One deliberate
difference: the forks here are z-ordered by their real depth from the camera
(so a blue fork in front of a cyan one draws in front), whereas the point-cloud
script just floats every fork above the cloud with one flat z-order.

  - X-tilt sweep (TILT_ANGLES_DEG), Y-tilt held at 0 -- the primary sweep from
    top-down (0deg) to horizontal (+-90deg) around the stem (blue).
  - Y-tilt sweep (Y_TILT_ANGLES_DEG) applied on top of one fixed X-tilt,
    showing the secondary roll around the closing axis (cyan).
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.spatial.transform import Rotation as R

# ----------------------------- CONFIG ---------------------------------

# Stem cylinder drawn along world X == the vine axis (TCP local X of the base
# pose). Kept at a realistic peduncle scale; the fork geometry below is sized
# to match it (not the 4x-enlarged cloud in grasp_tilt_pointcloud.py).
STEM_RADIUS = 0.006
STEM_LENGTH = 0.20
STEM_COLOR = "tab:green"
STEM_ALPHA = 0.5

# Same sweep set as grasp_tilt_pointcloud.py.
TILT_ANGLES_DEG = [0, 30, 60, -30, -60]
Y_TILT_ANGLES_DEG = [0, 30, 60, -30, -60]
Y_TILT_DEMO_X = 0  # deg_x held fixed while deg_y sweeps

# Fork geometry -- proportioned to the cylinder above.
STANDOFF = 0.2
PRONG_LENGTH = 0.05
FORK_OPENING = 0.03

# generate_grasp_pose() builds the base pose in camera_frame; a "top-down"
# grasp only means -Z after transform_pose() rotates it into the planning
# frame. True here (as in grasp_tilt_pointcloud.py) so the 0deg gripper sits
# above the stem approaching downward. Flip to False to approach from below.
FLIP_APPROACH_FOR_DISPLAY = True

FIGSIZE = (10, 9)
ELEV_VIEW, AZIM_VIEW = 25, -60
OUTPUT_PATH = "/tmp/claude-0/-root/3047a74c-ec3a-4b61-bbd6-9cab87991a9a/scratchpad/grasp_tilt_orientations.png"

X_TILT_COLOR = "tab:blue"
Y_TILT_COLOR = "tab:cyan"

# ------------------------------------------------------------------------


def tilt_pose(deg_x, deg_y, base_rot):
    """Mirrors SimplePickPoint._tilt_pose(): rotate deg_x about the vine axis
    (base local X), then rotate deg_y about the resulting local Y (closing axis)."""
    vine_axis = base_rot.apply([1, 0, 0])
    vine_axis = vine_axis / np.linalg.norm(vine_axis)
    x_tilt_rot = R.from_rotvec(np.radians(deg_x) * vine_axis)
    x_tilted_rot = x_tilt_rot * base_rot

    y_axis = x_tilted_rot.apply([0, 1, 0])
    y_axis = y_axis / np.linalg.norm(y_axis)
    y_tilt_rot = R.from_rotvec(np.radians(deg_y) * y_axis)

    return y_tilt_rot * x_tilted_rot


FORK_ZORDER = 10       # base z-order for forks; each fork adds its depth rank
GRASP_ZORDER = 100     # yellow grasp marker / axis arrow, always on top


def camera_dir(elev_deg, azim_deg):
    """Unit vector pointing from the scene toward the camera, for the given
    view_init(elev, azim). Larger dot(point, this) == closer to the viewer."""
    e, a = np.radians(elev_deg), np.radians(azim_deg)
    return np.array([np.cos(a) * np.cos(e),
                     np.sin(a) * np.cos(e),
                     np.sin(e)])


def draw_stem(ax, radius, length, color=STEM_COLOR, alpha=STEM_ALPHA, n=48):
    """Stem cylinder aligned with the world X axis (the vine axis)."""
    theta = np.linspace(0, 2 * np.pi, n)
    x = np.linspace(-length / 2, length / 2, 2)
    theta_grid, x_grid = np.meshgrid(theta, x)
    y_grid = radius * np.cos(theta_grid)
    z_grid = radius * np.sin(theta_grid)
    ax.plot_surface(x_grid, y_grid, z_grid, color=color, alpha=alpha,
                     linewidth=0, antialiased=True, shade=True, zorder=1)
    for xc in (-length / 2, length / 2):
        ax.plot([xc] * n, radius * np.cos(theta), radius * np.sin(theta),
                 color="k", linewidth=0.8, zorder=1)


def draw_gripper_fork(ax, position, approach_dir, finger_axis, color, zorder=FORK_ZORDER):
    approach_dir = approach_dir / np.linalg.norm(approach_dir)
    finger_axis = finger_axis / np.linalg.norm(finger_axis)

    prong_back = position
    prong_front = position + approach_dir * PRONG_LENGTH
    for sign in (-1, 1):
        offset = finger_axis * FORK_OPENING * sign
        ax.plot(*zip(prong_back + offset, prong_front + offset),
                 color=color, linewidth=4, solid_capstyle="round", zorder=zorder)

    palm_left = prong_back - finger_axis * FORK_OPENING
    palm_right = prong_back + finger_axis * FORK_OPENING
    ax.plot(*zip(palm_left, palm_right), color=color, linewidth=4,
             solid_capstyle="round", zorder=zorder)
    ax.plot(*zip(prong_back, (prong_back - approach_dir * 0.02)), color=color, linewidth=4,
             solid_capstyle="round", zorder=zorder)

    ax.quiver(*prong_back, *(approach_dir * PRONG_LENGTH * 0.8),
               color="dimgray", linewidth=1.0, arrow_length_ratio=0.3,
               linestyle="dashed", zorder=zorder)


def build_forks(tilts, base_rot, grasp_point, color):
    """Resolve each (deg_x, deg_y) tilt into the geometry needed to draw one
    fork, plus a legend handle. No drawing yet -- main() sorts them by camera
    depth first so nearer forks overdraw farther ones."""
    forks = []
    for deg_x, deg_y in tilts:
        rot = tilt_pose(deg_x, deg_y, base_rot)
        approach_dir = rot.apply([0, 0, 1])
        finger_axis = rot.apply([0, 1, 0])
        position = grasp_point - approach_dir * STANDOFF
        forks.append({
            "position": position,
            "approach_dir": approach_dir,
            "finger_axis": finger_axis,
            "color": color,
            "centroid": position + approach_dir * PRONG_LENGTH * 0.5,
            "handle": Line2D([0], [0], color=color, lw=2.5,
                             label=f"({deg_x}°, {deg_y}°)"),
        })
    return forks


def main():
    grasp_point = np.zeros(3)  # stem centre / grasp target

    base_rot = R.identity()
    if FLIP_APPROACH_FOR_DISPLAY:
        base_rot = base_rot * R.from_euler("x", 180, degrees=True)

    fig = plt.figure(figsize=FIGSIZE)
    ax = fig.add_subplot(111, projection="3d")
    # Fall back to plain 2D zorder so the forks can be forced above the stem
    # surface regardless of camera depth (mirrors grasp_tilt_pointcloud.py).
    ax.computed_zorder = False

    draw_stem(ax, STEM_RADIUS, STEM_LENGTH)

    ax.scatter(*grasp_point, color="yellow", s=40, zorder=GRASP_ZORDER,
               label="grasp point")
    ax.quiver(*grasp_point, *(0.08 * (grasp_point + [0, 1, 0])),
               color="yellow", linewidth=2.0, arrow_length_ratio=0.5,
               linestyle="dashed", zorder=GRASP_ZORDER)

    x_forks = build_forks([(dx, 0) for dx in TILT_ANGLES_DEG],
                          base_rot, grasp_point, X_TILT_COLOR)
    y_forks = build_forks([(Y_TILT_DEMO_X, dy) for dy in Y_TILT_ANGLES_DEG if dy != 0],
                          base_rot, grasp_point, Y_TILT_COLOR)
    x_handles = [f["handle"] for f in x_forks]
    y_handles = [f["handle"] for f in y_forks]

    # Draw farthest-first, giving each fork a z-order that climbs with camera
    # depth, so a fork genuinely in front (blue or cyan) overdraws one behind it
    # instead of whichever sweep happened to be drawn last.
    cam = camera_dir(ELEV_VIEW, AZIM_VIEW)
    ordered = sorted(x_forks + y_forks, key=lambda f: f["centroid"] @ cam)
    for rank, f in enumerate(ordered):
        draw_gripper_fork(ax, f["position"], f["approach_dir"], f["finger_axis"],
                          f["color"], zorder=FORK_ZORDER + rank)

    pad = STANDOFF + PRONG_LENGTH + 0.02
    stem_pts = np.array([[sx, sy, sz]
                         for sx in (-STEM_LENGTH / 2, STEM_LENGTH / 2)
                         for sy in (-STEM_RADIUS, STEM_RADIUS)
                         for sz in (-STEM_RADIUS, STEM_RADIUS)])
    pts_all = np.vstack([stem_pts, grasp_point + pad, grasp_point - pad])
    mins, maxs = pts_all.min(axis=0), pts_all.max(axis=0)
    ax.set_xlim(mins[0], maxs[0])
    ax.set_ylim(mins[1], maxs[1])
    ax.set_zlim(mins[2], maxs[2])
    ax.set_box_aspect(maxs - mins)

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title("Fallback grasp orientation search around the cylinder stem model\n"
                  "(mirrors simple_pick_point.py: _tilt_pose)",
                  fontsize=11, pad=20)
    ax.view_init(elev=ELEV_VIEW, azim=AZIM_VIEW)

    legend1 = ax.legend(handles=x_handles, title="X-tilt, deg_y=0\n(around vine axis)",
                         loc="upper left", fontsize=8, title_fontsize=8,
                         bbox_to_anchor=(-0.02, 0.95))
    ax.add_artist(legend1)
    ax.legend(handles=y_handles, title=f"Y-tilt, deg_x={Y_TILT_DEMO_X}°\n(around closing axis)",
              loc="upper right", fontsize=8, title_fontsize=8,
              bbox_to_anchor=(1.05, 0.95))

    fig.subplots_adjust(top=0.85, bottom=0.05)
    plt.savefig(OUTPUT_PATH, dpi=200)
    print(f"Saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
