import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import time
import open3d as o3d
from scipy.spatial import cKDTree

# ───── User‐Configurable Params ───────────────────────────────────────────
pcd_path         = "video_processing/point_clouds/64_skeleton.pcd"
web_scale        = 0.1                      # shrink the web by this factor
bug_start        = np.array([0.0, 0.0, 0.5])# starting (x,y,z)
bug_radius       = 1                      # bigger sphere for visibility
thread_point_size= 5                        # make web-points easier to see
# ─────────────────────────────────────────────────────────────────────────

# Simulation parameters
num_steps   = 500
step_scale  = 3
levy_alpha  = 1.5

# Load & scale the web point-cloud
pcd = o3d.io.read_point_cloud(pcd_path)
web_points = np.asarray(pcd.points) * web_scale

# Build KD-tree on the scaled points
kd_tree = cKDTree(web_points)

def levy_step_3d(alpha):
    u = np.random.rand()
    theta = np.random.rand() * 2*np.pi
    phi   = np.arccos(2*np.random.rand() - 1)
    vec   = np.array([
        np.sin(phi)*np.cos(theta),
        np.sin(phi)*np.sin(theta),
        np.cos(phi)
    ])
    return (vec / np.linalg.norm(vec)) * (u**(-1/alpha)) * step_scale

# Initialize bug state
pos         = bug_start.copy()
trajectory  = [pos.copy()]
captured    = False
start_time  = time.time()
capture_time= None

# Precompute unit‐sphere mesh for plotting
u, v = np.mgrid[0:2*np.pi:30j, 0:np.pi:15j]
sphere_x0 = np.cos(u)*np.sin(v)
sphere_y0 = np.sin(u)*np.sin(v)
sphere_z0 = np.cos(v)

# Set up 3D plot
fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(111, projection='3d')

# ➊ Compute combined bounds for web & bug, then pad
min_b = np.minimum(web_points.min(axis=0), bug_start - bug_radius)
max_b = np.maximum(web_points.max(axis=0), bug_start + bug_radius)
pad   = (max_b - min_b) * 0.1

ax.set_xlim(min_b[0] - pad[0], max_b[0] + pad[0])
ax.set_ylim(min_b[1] - pad[1], max_b[1] + pad[1])
ax.set_zlim(min_b[2] - pad[2], max_b[2] + pad[2])

ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')

# Plot the scaled web
ax.scatter(
    web_points[:,0],
    web_points[:,1],
    web_points[:,2],
    c='gray', s=thread_point_size, alpha=0.7
)

# Initialize bug sphere & trail
bug_surf = [ax.plot_surface(
    sphere_x0*bug_radius + pos[0],
    sphere_y0*bug_radius + pos[1],
    sphere_z0*bug_radius + pos[2],
    color='red', alpha=0.6)]
trail_line, = ax.plot([], [], [], 'r-', lw=2, alpha=0.6)
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

        # collision: nearest web‐point within bug_radius?
        dist, _ = kd_tree.query(pos)
        if dist <= bug_radius:
            captured = True
            capture_time = time.time() - start_time

    # update trail
    xs, ys, zs = zip(*trajectory)
    trail_line.set_data(xs, ys)
    trail_line.set_3d_properties(zs)

    # update timer
    if captured:
        timer_text.set_text(f"Captured in {capture_time:.2f}s")
    else:
        elapsed = time.time() - start_time
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
