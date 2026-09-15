"""
3D visualization of candidate grasp orientations around a truss stem
(modeled as a cylinder). Each gripper is drawn as a two-prong fork
(parallel-jaw finger pair) placed at its pre-grasp standoff, oriented
along its approach axis toward the stem.

Tweak the CONFIG block below to match your report's needs.
"""

import numpy as np
import matplotlib.pyplot as plt

# ----------------------------- CONFIG ---------------------------------

# Truss stem, modeled as a cylinder aligned with the Z axis
STEM_RADIUS = 0.006   # stem radius (m)
STEM_HEIGHT = 0.15    # stem segment length (m)

# Candidate grasps: each defined by (azimuth_deg, elevation_deg, roll_deg)
# azimuth: angle around the vertical (z) axis, 0 = +x axis -> sets yaw (position + approach direction)
# elevation: angle up from the horizontal plane, 90 = straight down from top -> sets pitch of the approach axis
# roll: rotation of the fork about its own approach axis (0 = default "upright" orientation)
GRASP_ANGLES = [
    (0, 0, 0),
    (60, 0, 45),
    (120, 0, 90),
    (180, 0, 0),
    (240, 0, 45),
    (300, 0, 90),
    (0, 90, 0),  # top-down grasp
]

STANDOFF = 0.10        # distance from object surface to pre-grasp position
PRONG_LENGTH = 0.05   # length of each finger prong
FORK_OPENING = 0.025   # half-distance between the two prongs
PALM_OFFSET = 0.0   # small gap between prong back-ends and the "palm" bar

FIGSIZE = (8, 8)
ELEV_VIEW, AZIM_VIEW = 22, -60
OUTPUT_PATH = "/tmp/claude-0/-root/3047a74c-ec3a-4b61-bbd6-9cab87991a9a/scratchpad/grasp_orientations.png"

# ------------------------------------------------------------------------


def spherical_to_cart(az_deg, el_deg, radius):
    az, el = np.radians(az_deg), np.radians(el_deg)
    x = radius * np.cos(el) * np.cos(az)
    y = radius * np.cos(el) * np.sin(az)
    z = radius * np.sin(el)
    return np.array([x, y, z])


def draw_cylinder(ax, radius, height, color="tab:green", alpha=0.5, n=48):
    """Draw a cylinder (the truss stem) aligned with the Z axis, centered at the origin."""
    theta = np.linspace(0, 2 * np.pi, n)
    z = np.linspace(-height / 2, height / 2, 2)
    theta_grid, z_grid = np.meshgrid(theta, z)
    x_grid = radius * np.cos(theta_grid)
    y_grid = radius * np.sin(theta_grid)
    ax.plot_surface(x_grid, y_grid, z_grid, color=color, alpha=alpha,
                     linewidth=0, antialiased=True, shade=True)

    # cap outlines for a cleaner silhouette
    for zc in (-height / 2, height / 2):
        ax.plot(radius * np.cos(theta), radius * np.sin(theta), zc,
                color="k", linewidth=0.8)


def rotate_around_axis(v, axis, angle_deg):
    """Rotate vector v about unit vector axis by angle_deg (Rodrigues' rotation formula)."""
    axis = axis / np.linalg.norm(axis)
    angle = np.radians(angle_deg)
    return (v * np.cos(angle)
            + np.cross(axis, v) * np.sin(angle)
            + axis * np.dot(axis, v) * (1 - np.cos(angle)))


def surface_distance(direction, radius, height):
    """Distance from the cylinder's center axis-origin to its surface along `direction`."""
    dx, dy, dz = direction
    horiz = np.hypot(dx, dy)
    half_h = height / 2
    if horiz < 1e-9:
        return half_h / abs(dz)
    t_lateral = radius / horiz
    if abs(dz) < 1e-9:
        return t_lateral
    t_cap = half_h / abs(dz)
    return min(t_lateral, t_cap)


def draw_gripper_fork(ax, position, approach_dir, roll_deg=0.0, up_hint=np.array([0, 0, 1])):
    """Draw a 2-prong fork gripper: two parallel finger lines + a palm bar,
    positioned at `position`, pointing along `approach_dir` toward the object.
    `roll_deg` rotates the fork about its own approach axis."""
    approach_dir = approach_dir / np.linalg.norm(approach_dir)

    # pick a default "upright" finger axis perpendicular to the approach direction
    if abs(np.dot(approach_dir, up_hint)) > 0.95:
        ref = np.array([1.0, 0.0, 0.0])
    else:
        ref = up_hint
    finger_axis = np.cross(approach_dir, ref)
    finger_axis /= np.linalg.norm(finger_axis)

    # roll: rotate the finger axis about the approach axis itself
    finger_axis = rotate_around_axis(finger_axis, approach_dir, roll_deg)

    prong_back = position
    prong_front = position + approach_dir * PRONG_LENGTH
    palm_back = position - approach_dir * PALM_OFFSET

    for sign in (-1, 1):
        offset = finger_axis * FORK_OPENING * sign
        back = prong_back + offset
        front = prong_front + offset
        ax.plot(*zip(back, front), color="tab:blue", linewidth=2.5, solid_capstyle="round")

    # palm bar connecting the two prong backs (perpendicular cross-piece)
    palm_left = palm_back - finger_axis * FORK_OPENING
    palm_right = palm_back + finger_axis * FORK_OPENING
    ax.plot(*zip(palm_left, palm_right), color="tab:blue", linewidth=2.5, solid_capstyle="round")

    # fork origin
    ax.plot(*zip(palm_back, (palm_back * 1.2)), color="tab:blue", linewidth=2.5, solid_capstyle="round")

    # approach-axis arrow from palm to object
    ax.quiver(*palm_back, *(approach_dir * (PALM_OFFSET + PRONG_LENGTH * 0.8)),
               color="gray", linewidth=1.0, arrow_length_ratio=0.25, linestyle="dashed")


def main():
    fig = plt.figure(figsize=FIGSIZE)
    ax = fig.add_subplot(111, projection="3d")

    draw_cylinder(ax, STEM_RADIUS, STEM_HEIGHT)

    for az, el, roll in GRASP_ANGLES:
        direction = spherical_to_cart(az, el, 1.0)  # unit vector from stem axis to gripper
        surf_dist = surface_distance(direction, STEM_RADIUS, STEM_HEIGHT)
        position = direction * (surf_dist + STANDOFF)
        approach_dir = -direction  # gripper approaches toward the stem
        draw_gripper_fork(ax, position, approach_dir, roll_deg=roll)

    lim = max(STEM_RADIUS, STEM_HEIGHT / 2) + STANDOFF + PRONG_LENGTH + 0.02
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-lim, lim)
    ax.set_box_aspect([1, 1, 1])
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Candidate Grasp Orientations (Pre-grasp Poses)")
    ax.view_init(elev=ELEV_VIEW, azim=AZIM_VIEW)

    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=200)
    print(f"Saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
