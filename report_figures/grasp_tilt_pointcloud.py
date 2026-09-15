"""
Same fallback-orientation search as grasp_tilt_orientations.py (mirrors
SimplePickPoint._tilt_pose() in simple_pick_point.py), but drawn around a real
scanned point cloud of a tomato truss instead of an idealized cylinder.

Faithful to the source pipeline's own assumption: generate_grasp_pose() only
ever derives a horizontal yaw from the two picked pixels
(quaternion_from_euler(0, 0, yaw)), so the "vine axis" the tilt search starts
from is always the horizontal projection of the picked direction -- never a
full 3D-fitted stem axis. Rather than inventing a 3D principal axis, this
script estimates that same horizontal yaw automatically from the cloud's
dominant X-Y direction, as a stand-in for "the two mouse clicks".
"""

import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.spatial.transform import Rotation as R

# ----------------------------- CONFIG ---------------------------------

PCD_PATH = "/root/Crate_truss_pick-and-place/experiments/map/raw.pcd"
# (was bbox_filtered.pcd; raw.pcd is the unfiltered 354k-point scan -- rely on
#  REMOVE_CUTS below plus the saturation/Z-band filters to clean it.)

# Hand-drawn cuts from report_figures/build_pcd_trim_editor.py (open the built
# HTML as an Artifact, lasso the clutter, paste the exported block here).
# Each cut drops points whose two in-plane coordinates fall inside the polygon:
#   view "xy" tests (x, y)   "xz" tests (x, z)   "yz" tests (y, z)
# A point is removed if it lies inside ANY polygon for that view. Applied to
# the full-resolution cloud right after load, before everything else.
REMOVE_CUTS = [
    {"view": "xy", "poly": [[0.60079, 0.87756], [-0.04748, 1.60949], [0.39473, 0.60251], [-0.18918, 0.72516], [-0.43385, 1.87513], [-0.15675, 2.42537], [1.20895, 2.22744], [2.93488, 0.52526], [2.95072, -0.43667], [1.07436, -0.67814]]},
    {"view": "xy", "poly": [[0.38833, 0.15202], [0.39636, 0.15814], [0.39694, 0.17287], [0.42314, 0.18033], [0.43194, 0.16905], [0.43233, 0.16025], [0.44935, 0.18435], [0.4549, 0.17804], [0.44074, 0.15011], [0.44533, 0.14513], [0.46064, 0.15298], [0.48015, 0.13729], [0.47996, 0.11969], [0.46791, 0.10879], [0.46494, 0.10646], [0.46708, 0.09898], [0.46953, 0.08156], [0.45693, 0.06893], [0.45646, 0.06307], [0.46094, 0.05378], [0.4583, 0.03542], [0.43718, 0.02153], [0.42686, 0.02429], [0.41997, 0.00834], [0.41194, -0.01094], [0.39027, -0.01583], [0.3802, 0.00453], [0.36719, 0.03936], [0.36719, 0.03936], [0.3674, 0.05153], [0.37184, 0.05511], [0.37254, 0.06045], [0.37428, 0.06881], [0.36944, 0.07679], [0.37277, 0.10355], [0.38422, 0.15139], [0.31925, 0.1642], [0.18145, 0.05209], [0.24072, -0.20444], [0.63426, -0.25108], [0.75572, 0.17258], [0.71394, 0.27461], [0.3487, 0.34698], [0.32062, 0.17358]]},
    {"view": "xy", "poly": [[0.38047, 0.15843], [0.37899, 0.14143], [0.30621, 0.15695], [0.33872, 0.18872]]},
    {"view": "xz", "poly": [[0.38341, 0.0754], [0.40235, 0.07757], [0.40888, 0.12943], [0.38279, 0.13036], [0.37658, 0.0754]]},
    {"view": "xz", "poly": [[0.4491, 0.03638], [0.47021, 0.03937], [0.47446, 0.02898], [0.44784, 0.02945]]},
    {"view": "xz", "poly": [[0.41885, 0.03024], [0.48691, 0.03213], [0.48486, -0.003], [0.33915, -0.00379], [0.33836, 0.03339]]},
]

# The raw cloud has some floating clutter (wires/table points) above and
# below the main trellis band; keep only the main band for the yaw/point
# estimate and for display. Set to None to disable filtering.
Z_BAND = (0.03, 0.09)

# The gray/tan clutter (wires, table, background) is much less saturated than
# the truss itself (vivid red tomatoes, green stem/pedicels): mean saturation
# ~0.11 for clutter vs ~0.54 for the truss in this scan. Points with HSV
# saturation below this threshold are dropped before anything else runs.
# Set to None to disable.
SATURATION_THRESHOLD = 0.2

# Random subsample cap for scatter performance. Set to None to draw every
# point that survives the cuts + saturation filter (~74k here -- fine for a
# one-off render, just a little slower).
MAX_POINTS_DISPLAYED = None

POINT_SIZE = 6.0  # matplotlib scatter marker area (was 1.5)

# Display-only affine transform applied to the point cloud (NOT to the
# grasp point or the gripper forks, which stay fixed where the tilt math
# puts them). Use these to nudge/enlarge the cloud until the real peduncle
# visually lines up with the fixed fork convergence point.
#   CLOUD_SCALE: >1 enlarges the cloud, <1 shrinks it, about CLOUD_SCALE_PIVOT
#   CLOUD_TRANSLATION: [dx, dy, dz] shift applied after scaling (e.g. negative
#                       dz moves the cloud down)
#   CLOUD_SCALE_PIVOT: point to scale about; None = the estimated/overridden
#                      grasp point itself, so scaling doesn't move the one
#                      point that needs to line up with the forks -- only
#                      CLOUD_TRANSLATION does that afterwards.
CLOUD_SCALE = 4
CLOUD_TRANSLATION = np.array([0.03, 0.0, 0.0])
CLOUD_SCALE_PIVOT = None

# Manual overrides (set to bypass automatic estimation), e.g.:
#   YAW_DEG_OVERRIDE = 90.0
#   GRASP_POINT_OVERRIDE = np.array([0.41, 0.09, 0.056])
YAW_DEG_OVERRIDE = None
GRASP_POINT_OVERRIDE = None

# generate_grasp_pose() builds the base pose in camera_frame; it only gets its
# true "top-down" meaning after transform_pose() rotates it into the planning
# frame via the camera's extrinsics, which aren't recoverable from a static
# .pcd file alone. Defaulted to True here as a best guess: clutter sits above
# the truss around Z~0.10-0.13 (plausibly an overhead support wire) and below
# it around Z<0.02 (plausibly the table), suggesting Z increases upward in
# this cloud -- so a "top-down" grasp should point down (-Z), not up. Flip
# back to False if that guess is wrong for your actual planning frame.
FLIP_APPROACH_FOR_DISPLAY = True

TILT_ANGLES_DEG = [0, 30, 60, -30, -60]
Y_TILT_ANGLES_DEG = [0, 30, 60, -30, -60]
Y_TILT_DEMO_X = 0  # deg_x held fixed while deg_y sweeps

# Combined pitch+roll orientations (deg_x, deg_y), drawn on top of the two
# single-axis sweeps above so you can see mixed tilts like 30 deg pitch +
# 30 deg roll. Set to [] for none.
#   SHOW_ALL_COMBINATIONS = True ignores this list and instead draws the full
#   TILT_ANGLES_DEG x Y_TILT_ANGLES_DEG grid, minus the pure-pitch / pure-roll
#   cases the other two sweeps already cover.
#COMBO_TILT_ANGLES = []#(30, 30), (30, -30), (-30, 30), (-30, -30)]
SHOW_ALL_COMBINATIONS = True

STANDOFF = 0.2
PRONG_LENGTH = 0.05
FORK_OPENING = 0.03

FIGSIZE = (10, 9)
ELEV_VIEW, AZIM_VIEW = 25, -70

# --- Axis ranges shown in the figure -----------------------------------
# Tighter ranges make the truss and grippers bigger in the plot. Each of
# XLIM / YLIM / ZLIM is either:
#   None      -> auto-fit to the drawn content (cloud + forks), padded by
#                AXIS_MARGIN_FRAC of the content extent on that axis
#   (lo, hi)  -> use exactly this range, in metres (anything outside is clipped)
# e.g. XLIM = (0.15, 0.70) to crop the wide X span and zoom in.
XLIM = None
YLIM = None
ZLIM = None
AXIS_MARGIN_FRAC = 0.05   # applied only to axes left as None
OUTPUT_PATH = "/tmp/claude-0/-root/3047a74c-ec3a-4b61-bbd6-9cab87991a9a/scratchpad/grasp_tilt_pointcloud.png"

X_TILT_COLOR = "tab:blue"
Y_TILT_COLOR = "tab:cyan"
COMBO_TILT_COLOR = "tab:red"

# ------------------------------------------------------------------------


def load_pcd_ascii(path):
    with open(path) as f:
        lines = f.readlines()
    fields = None
    data_start = None
    for i, line in enumerate(lines):
        if line.startswith("FIELDS"):
            fields = line.split()[1:]
        elif line.startswith("DATA"):
            data_start = i + 1
            break
    data = np.loadtxt(lines[data_start:], dtype=np.float64)
    xyz = data[:, :3]
    colors = None
    if fields is not None and "rgb" in fields:
        rgb_col = fields.index("rgb")
        rgb_raw = data[:, rgb_col].astype(np.uint32)
        r = ((rgb_raw >> 16) & 255) / 255.0
        g = ((rgb_raw >> 8) & 255) / 255.0
        b = (rgb_raw & 255) / 255.0
        colors = np.stack([r, g, b], axis=1)
    return xyz, colors


_CUT_AXES = {"xy": (0, 1), "xz": (0, 2), "yz": (1, 2)}


def _points_in_poly(pts, poly):
    """Even-odd ray-cast test, vectorised over pts. Matches the Trim Bench
    editor's hit-test exactly (build_pcd_trim_editor.py), including how it
    treats self-intersecting polygons -- unlike matplotlib's nonzero-winding
    Path.contains_points."""
    x, y = pts[:, 0], pts[:, 1]
    inside = np.zeros(len(pts), dtype=bool)
    n = len(poly)
    j = n - 1
    with np.errstate(divide="ignore", invalid="ignore"):
        for i in range(n):
            xi, yi = poly[i]
            xj, yj = poly[j]
            crosses = (yi > y) != (yj > y)
            xcross = (xj - xi) * (y - yi) / (yj - yi) + xi
            inside ^= crosses & (x < xcross)
            j = i
    return inside


def apply_cuts(xyz, colors, cuts):
    """Drop points that fall inside any hand-drawn cut polygon (see REMOVE_CUTS).
    Polygons are in world metres, tested against the two in-plane axes named by
    the cut's view."""
    if not cuts:
        return xyz, colors
    drop = np.zeros(len(xyz), dtype=bool)
    for cut in cuts:
        ai, bi = _CUT_AXES[cut["view"]]
        poly = np.asarray(cut["poly"], dtype=np.float64)
        drop |= _points_in_poly(xyz[:, [ai, bi]], poly)
    keep = ~drop
    print(f"apply_cuts: removed {drop.sum()} / {len(xyz)} points "
          f"({len(cuts)} cut{'s' if len(cuts) != 1 else ''})")
    return xyz[keep], (colors[keep] if colors is not None else None)


def filter_by_saturation(xyz, colors, threshold):
    if threshold is None or colors is None:
        return xyz, colors
    maxc = colors.max(axis=1)
    minc = colors.min(axis=1)
    sat = np.where(maxc > 0, (maxc - minc) / np.where(maxc == 0, 1, maxc), 0)
    keep = sat > threshold
    return xyz[keep], colors[keep]


def estimate_yaw_and_point(xyz, z_band):
    if z_band is not None:
        mask = (xyz[:, 2] > z_band[0]) & (xyz[:, 2] < z_band[1])
        xyz_f = xyz[mask]
    else:
        xyz_f = xyz

    grasp_point = xyz_f.mean(axis=0)
    xy_centered = xyz_f[:, :2] - grasp_point[:2]
    cov2d = xy_centered.T @ xy_centered / len(xy_centered)
    eigvals, eigvecs = np.linalg.eigh(cov2d)
    principal_xy = eigvecs[:, -1]
    yaw_deg = np.degrees(np.arctan2(principal_xy[1], principal_xy[0]))
    return grasp_point, yaw_deg


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


FORK_ZORDER = 10       # base z-order for forks (always above the cloud at z=1);
                       # each fork adds its camera-depth rank on top of this
GRASP_ZORDER = 100     # yellow grasp marker / axis arrow, always on top


def camera_dir(elev_deg, azim_deg):
    """Unit vector pointing from the scene toward the camera, for the given
    view_init(elev, azim). Larger dot(point, this) == closer to the viewer."""
    e, a = np.radians(elev_deg), np.radians(azim_deg)
    return np.array([np.cos(a) * np.cos(e),
                     np.sin(a) * np.cos(e),
                     np.sin(e)])


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
    ax.plot(*zip(prong_back, (prong_back - approach_dir *0.04)), color=color, linewidth=4,
             solid_capstyle="round", zorder=zorder)

    #ax.quiver(*prong_back, *(approach_dir * PRONG_LENGTH * 0.8),
    #           color="dimgray", linewidth=1.0, arrow_length_ratio=0.3,
    #           linestyle="dashed", zorder=zorder)


def build_forks(tilts, base_rot, grasp_point, color):
    """Resolve each (deg_x, deg_y) tilt into the geometry needed to draw one
    fork, plus a legend handle. No drawing yet -- main() sorts them by camera
    depth first so nearer forks overdraw farther ones (blue over cyan or the
    reverse, whichever is really in front)."""
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
    xyz, colors = load_pcd_ascii(PCD_PATH)
    xyz, colors = apply_cuts(xyz, colors, REMOVE_CUTS)
    xyz, colors = filter_by_saturation(xyz, colors, SATURATION_THRESHOLD)

    # Grasp point / yaw are estimated from the ORIGINAL (untransformed) cloud,
    # so they don't drift as you tune CLOUD_SCALE/CLOUD_TRANSLATION below.
    est_point, est_yaw = estimate_yaw_and_point(xyz, Z_BAND)
    grasp_point = GRASP_POINT_OVERRIDE if GRASP_POINT_OVERRIDE is not None else est_point
    yaw_deg = YAW_DEG_OVERRIDE if YAW_DEG_OVERRIDE is not None else est_yaw

    # Display-only transform: scale the cloud about a pivot, then translate.
    # The grasp point and gripper forks are NOT affected -- only the cloud moves.
    pivot = CLOUD_SCALE_PIVOT if CLOUD_SCALE_PIVOT is not None else grasp_point
    xyz = (xyz - pivot) * CLOUD_SCALE + pivot + CLOUD_TRANSLATION

    if MAX_POINTS_DISPLAYED is not None and len(xyz) > MAX_POINTS_DISPLAYED:
        rng = np.random.default_rng(0)
        idx = rng.choice(len(xyz), size=MAX_POINTS_DISPLAYED, replace=False)
        xyz_disp, colors_disp = xyz[idx], colors[idx] if colors is not None else None
    else:
        xyz_disp, colors_disp = xyz, colors
    print(f"displaying {len(xyz_disp)} / {len(xyz)} points")
    base_rot = R.from_euler("z", np.radians(yaw_deg))
    if FLIP_APPROACH_FOR_DISPLAY:
        base_rot = base_rot * R.from_euler("x", 180, degrees=True)

    fig = plt.figure(figsize=FIGSIZE)
    ax = fig.add_subplot(111, projection="3d")
    # mplot3d normally re-sorts every artist by its own average camera depth each
    # draw, so the (large) point cloud can end up drawn on top of the (thin) fork
    # lines even where the forks are logically in front. Disabling that and
    # falling back to plain 2D zorder lets us force the forks above the cloud --
    # and rank the forks against each other by real camera depth (see below).
    ax.computed_zorder = False

    ax.scatter(xyz_disp[:, 0], xyz_disp[:, 1], xyz_disp[:, 2],
               c=colors_disp, s=POINT_SIZE, alpha=0.6, depthshade=False, zorder=1)
    ax.scatter(*grasp_point, color="yellow", s=40, zorder=GRASP_ZORDER,
               label="grasp point")
    ax.quiver(*grasp_point, *(0.08*(grasp_point + [0,1,0])),
               color="yellow", linewidth=2.0, arrow_length_ratio=0.5,
               linestyle="dashed", zorder=GRASP_ZORDER)

    x_forks = build_forks([(dx, 0) for dx in TILT_ANGLES_DEG],
                          base_rot, grasp_point, X_TILT_COLOR)
    y_forks = build_forks([(Y_TILT_DEMO_X, dy) for dy in Y_TILT_ANGLES_DEG if dy != 0],
                          base_rot, grasp_point, Y_TILT_COLOR)
    if SHOW_ALL_COMBINATIONS:
        combo_angles = [(dx, dy) for dx in TILT_ANGLES_DEG for dy in Y_TILT_ANGLES_DEG
                        if dx != 0 and dy != 0]
    else:
        combo_angles = list(COMBO_TILT_ANGLES)
    combo_forks = build_forks(combo_angles, base_rot, grasp_point, COMBO_TILT_COLOR)

    x_handles = [f["handle"] for f in x_forks]
    y_handles = [f["handle"] for f in y_forks]
    if len(combo_forks) <= 8:
        combo_handles = [f["handle"] for f in combo_forks]
    else:
        combo_handles = [Line2D([0], [0], color=COMBO_TILT_COLOR, lw=2.5,
                                label=f"{len(combo_forks)} (deg_x, deg_y) combos")]

    # Draw farthest-first, giving each fork a z-order that climbs with camera
    # depth, so a fork genuinely in front overdraws one behind it instead of
    # whichever sweep happened to be drawn last.
    all_forks = x_forks + y_forks + combo_forks
    cam = camera_dir(ELEV_VIEW, AZIM_VIEW)
    ordered = sorted(all_forks, key=lambda f: f["centroid"] @ cam)
    for rank, f in enumerate(ordered):
        draw_gripper_fork(ax, f["position"], f["approach_dir"], f["finger_axis"],
                          f["color"], zorder=FORK_ZORDER + rank)

    # Auto-fit bounds: the displayed cloud plus every fork's back and front
    # point. Any of XLIM/YLIM/ZLIM that is set overrides its axis exactly.
    fork_ends = np.array(
        [f["position"] for f in all_forks]
        + [f["position"] + f["approach_dir"] * PRONG_LENGTH for f in all_forks]
    )
    content = np.vstack([xyz[:, :3], fork_ends, [grasp_point]])
    lo, hi = content.min(axis=0), content.max(axis=0)
    m = (hi - lo) * AXIS_MARGIN_FRAC
    auto_lo, auto_hi = lo - m, hi + m

    xlim = XLIM if XLIM is not None else (auto_lo[0], auto_hi[0])
    ylim = YLIM if YLIM is not None else (auto_lo[1], auto_hi[1])
    zlim = ZLIM if ZLIM is not None else (auto_lo[2], auto_hi[2])
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_zlim(*zlim)
    # Physically proportional box (1 m looks the same on every axis), so
    # shrinking a range is a true zoom rather than a stretch.
    ax.set_box_aspect((xlim[1] - xlim[0], ylim[1] - ylim[0], zlim[1] - zlim[0]))

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    cuts_note = f", {len(REMOVE_CUTS)} hand cuts" if REMOVE_CUTS else ""
    ax.set_title("Fallback grasp orientation search on the scanned truss\n"
                  f"({os.path.basename(PCD_PATH)}{cuts_note}, "
                  f"estimated yaw={yaw_deg:.1f}°)",
                  fontsize=11, pad=20)
    ax.view_init(elev=ELEV_VIEW, azim=AZIM_VIEW)

    legend1 = ax.legend(handles=x_handles, title="X-tilt, deg_y=0\n(around vine axis)",
                         loc="upper left", fontsize=8, title_fontsize=8,
                         bbox_to_anchor=(-0.02, 0.95))
    ax.add_artist(legend1)
    legend2 = ax.legend(handles=y_handles,
                        title=f"Y-tilt, deg_x={Y_TILT_DEMO_X}°\n(around closing axis)",
                        loc="upper right", fontsize=8, title_fontsize=8,
                        bbox_to_anchor=(1.05, 0.95))
    if combo_handles:
        ax.add_artist(legend2)
        ax.legend(handles=combo_handles, title="Pitch + roll\n(deg_x, deg_y)",
                  loc="lower left", fontsize=8, title_fontsize=8,
                  bbox_to_anchor=(-0.02, 0.08))

    fig.subplots_adjust(top=0.85, bottom=0.05)
    plt.savefig(OUTPUT_PATH, dpi=200)
    print(f"Saved to {OUTPUT_PATH}")
    print(f"Estimated grasp point: {grasp_point}, yaw: {yaw_deg:.2f} deg")


if __name__ == "__main__":
    main()
