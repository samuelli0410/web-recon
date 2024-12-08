import open3d as o3d
import numpy as np
import pyvista as pv

# Load the .pcd file
input_pcd_path = "video_processing/point_clouds/@032 255 2024-11-08 04-05-31.pcd"
pcd = o3d.io.read_point_cloud(input_pcd_path)

# Convert Open3D point cloud to NumPy array
points = np.asarray(pcd.points)

# Create a PyVista PolyData object
point_cloud = pv.PolyData(points)

# Initialize colors: white for all points
colors = np.full((len(points), 3), 255, dtype=np.uint8)

# If the .pcd file has colors, use them; otherwise, stick with white
if pcd.has_colors():
    colors = (np.asarray(pcd.colors) * 255).astype(np.uint8)

point_cloud["colors"] = colors

# Keep track of selected indices globally
selected_indices = set()
hovered_index = [-1]  # Mutable variable to store hovered index

# Initialize PyVista plotter
plotter = pv.Plotter()
plotter.add_mesh(
    point_cloud, scalars="colors", rgb=True, point_size=12, render_points_as_spheres=True
)

# Timer callback for continuous hover detection
def update_hovered_point(caller, event):
    picked_position = plotter.pick_mouse_position()  # Get the mouse hover position
    picked_index = point_cloud.find_closest_point(picked_position)  # Find the nearest point index
    if picked_index != -1 and picked_index != hovered_index[0]:
        hovered_index[0] = picked_index  # Update hovered index
        print(f"Hovering over point {picked_index}")

# Callback function to handle selection when pressing "A"
def select_point_callback():
    if hovered_index[0] != -1 and hovered_index[0] not in selected_indices:
        selected_indices.add(hovered_index[0])  # Add index to selected set
        colors[hovered_index[0]] = [255, 0, 0]  # Change color to red
        point_cloud["colors"] = colors  # Update PolyData colors
        plotter.update_scalars(colors, render=True)  # Render updated colors
        print(f"Point {hovered_index[0]} selected.")

# Add key event for 'A' to trigger the callback
plotter.add_key_event("a", select_point_callback)

# Add a timer observer for continuous hover updates
plotter.iren.add_observer("TimerEvent", update_hovered_point)
plotter.iren.create_timer(50, repeating=True)  # Check hover position every 50ms

print("Hover over a point and press 'A' to select it. Selected points will remain red.")
plotter.show()

# Save the labeled point cloud
labels = np.zeros(len(points), dtype=np.uint8)  # Default labels: 0
if selected_indices:
    labels[list(selected_indices)] = 1  # Label selected points as class 1

point_cloud["labels"] = labels

# Save labeled point cloud to disk
labeled_pcd_path = input_pcd_path.split(".pcd")[0] + " labeled.pcd"
pcd.colors = o3d.utility.Vector3dVector(colors / 255.0)  # Normalize colors to [0, 1] for Open3D
pcd.points = o3d.utility.Vector3dVector(points)
o3d.io.write_point_cloud(labeled_pcd_path, pcd)
