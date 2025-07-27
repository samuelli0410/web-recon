import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import time

# ─── User‐Configurable Parameters ─────────────────────────────────────────────
bug_start       = np.array([0.0, 0.0, 0.5])   # starting (x,y,z) of the bug
thread_thickness = 1.0                        # line width for *all* threads
# ───────────────────────────────────────────────────────────────────────────────

# Simulation parameters
num_steps            = 500
step_scale           = 0.05
levy_alpha           = 1.5
capture_threshold_xy = 0.02
capture_threshold_z  = 0.01
bug_radius           = 0.03

# Define planar web (z=0): concentric circles + radial spokes
num_circles  = 5
num_radials  = 12
radii        = np.linspace(0.2, 0.8, num_circles)
angles       = np.linspace(0, 2*np.pi, num_radials, endpoint=False)

# Precompute circle meshes
circle_meshes = []
for r in radii:
    thetas = np.linspace(0, 2*np.pi, 200)
    xs = r * np.cos(thetas)
    ys = r * np.sin(thetas)
    zs = np.zeros_like(xs)
    circle_meshes.append((xs, ys, zs))

# Precompute radial spokes
radial_segs = []
for a in angles:
    x_end, y_end = 0.8*np.cos(a), 0.8*np.sin(a)
    radial_segs.append(((0.0, 0.0), (x_end, y_end)))

# Generate random 3D threads
num_threads   = 8
thread_segs_3d = []
for _ in range(num_threads):
    p1 = np.random.uniform(-1, 1, size=3)
    p2 = np.random.uniform(-1, 1, size=3)
    thread_segs_3d.append((p1, p2))

def levy_step_3d(alpha):
    u = np.random.rand()
    theta = np.random.rand() * 2*np.pi
    phi   = np.arccos(2*np.random.rand() - 1)
    vec   = np.array([np.sin(phi)*np.cos(theta),
                      np.sin(phi)*np.sin(theta),
                      np.cos(phi)])
    return (vec / np.linalg.norm(vec)) * (u**(-1/alpha)) * step_scale

def point_to_segment_dist_2d(p, seg):
    p, a, b = np.array(p), np.array(seg[0]), np.array(seg[1])
    ab = b - a
    t  = np.clip(np.dot(p - a, ab) / np.dot(ab, ab), 0.0, 1.0)
    proj = a + t*ab
    return np.linalg.norm(p - proj)

def point_to_segment_dist_3d(p, seg):
    p, a, b = np.array(p), np.array(seg[0]), np.array(seg[1])
    ab = b - a
    t  = np.clip(np.dot(p - a, ab) / np.dot(ab, ab), 0.0, 1.0)
    proj = a + t*ab
    return np.linalg.norm(p - proj)

# Initialize bug
pos        = bug_start.copy()
trajectory = [pos.copy()]
captured   = False
capture_time = None
start_time   = time.time()

# Precompute a unit‐sphere mesh
u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
sphere_x0 = np.cos(u)*np.sin(v)
sphere_y0 = np.sin(u)*np.sin(v)
sphere_z0 = np.cos(v)

# Set up 3D plot
fig = plt.figure(figsize=(7,7))
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim(-1, 1); ax.set_ylim(-1, 1); ax.set_zlim(0, 1)
ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')

# Draw planar web with parametrized thickness
for xs, ys, zs in circle_meshes:
    ax.plot(xs, ys, zs, color='gray', lw=thread_thickness)
for seg in radial_segs:
    x0,y0 = seg[0]; x1,y1 = seg[1]
    ax.plot([x0,x1], [y0,y1], [0,0], color='gray', lw=thread_thickness)

# Draw random 3D threads with same thickness
for p1, p2 in thread_segs_3d:
    ax.plot([p1[0],p2[0]],
            [p1[1],p2[1]],
            [p1[2],p2[2]],
            color='gray', lw=thread_thickness, linestyle='--')

# Plot handles
bug_surf    = [ax.plot_surface(
    sphere_x0*bug_radius + pos[0],
    sphere_y0*bug_radius + pos[1],
    sphere_z0*bug_radius + pos[2],
    color='red', alpha=0.6)]
trail_line, = ax.plot([], [], [], 'r-', lw=1, alpha=0.6)
timer_text  = ax.text2D(0.02, 0.95, "", transform=ax.transAxes)

def init():
    trail_line.set_data([], [])
    trail_line.set_3d_properties([])
    timer_text.set_text("Time: 0.00s")
    return trail_line, timer_text

def update(frame):
    global pos, captured, capture_time, bug_surf

    if not captured:
        pos += levy_step_3d(levy_alpha)
        trajectory.append(pos.copy())

        # check collision with planar web
        if pos[2] - bug_radius <= capture_threshold_z:
            r_xy = np.linalg.norm(pos[:2])
            # circles
            for r in radii:
                if abs(r_xy - r) <= capture_threshold_xy + bug_radius:
                    captured = True; break
            # radials
            if not captured:
                for seg in radial_segs:
                    if point_to_segment_dist_2d(pos[:2], seg) <= capture_threshold_xy + bug_radius:
                        captured = True; break
            # 3D threads
            if not captured:
                for seg3d in thread_segs_3d:
                    if point_to_segment_dist_3d(pos, seg3d) <= bug_radius:
                        captured = True; break

        if captured:
            capture_time = time.time() - start_time

    # update trail
    xs, ys, zs = zip(*trajectory)
    trail_line.set_data(xs, ys)
    trail_line.set_3d_properties(zs)

    # update timer
    elapsed = time.time() - start_time
    if captured:
        timer_text.set_text(f"Captured in {capture_time:.2f}s")
    else:
        timer_text.set_text(f"Time: {elapsed:.2f}s")

    # redraw bug sphere
    for surf in bug_surf:
        surf.remove()
    bug_surf = [ax.plot_surface(
        sphere_x0*bug_radius + pos[0],
        sphere_y0*bug_radius + pos[1],
        sphere_z0*bug_radius + pos[2],
        color='red', alpha=0.6)]

    return trail_line, timer_text, *bug_surf

ani = FuncAnimation(
    fig, update, frames=num_steps,
    init_func=init, blit=False,
    interval=50, repeat=False
)

plt.tight_layout()
plt.show()
