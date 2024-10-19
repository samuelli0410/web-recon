import open3d as o3d
import numpy as np
import matplotlib.cm as cm
import matplotlib.colors as colors

import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
from pcd_generation_improved import voxelize, voxel_to_pcd

class VoxelGridApp:
    def __init__(self, voxel_grid):
        self.voxel_grid = voxel_grid
        self.original_voxel_grid = voxel_grid.copy()

        # Get the voxel grid dimensions
        self.grid_dims = voxel_grid.shape

        # Initialize the Open3D application
        self.app = gui.Application.instance
        self.app.initialize()

        # Create the main window
        self.window = gui.Application.instance.create_window("Voxel Grid Viewer", 1024, 768)

        # Create the 3D scene widget
        self.scene = gui.SceneWidget()
        self.scene.scene = rendering.Open3DScene(self.window.renderer)
        self.window.add_child(self.scene)

        # Keep track of labels
        self.labels = []
        # Keep track of visibility states
        self.grid_visible = True
        self.axes_visible = True

        # Create UI elements
        em = self.window.theme.font_size

        self.panel = gui.Vert(0, gui.Margins(em, em, em, em))

        # Input boxes for cropping dimensions
        self.x_min_input = gui.NumberEdit(gui.NumberEdit.INT)
        self.x_max_input = gui.NumberEdit(gui.NumberEdit.INT)
        self.y_min_input = gui.NumberEdit(gui.NumberEdit.INT)
        self.y_max_input = gui.NumberEdit(gui.NumberEdit.INT)
        self.z_min_input = gui.NumberEdit(gui.NumberEdit.INT)
        self.z_max_input = gui.NumberEdit(gui.NumberEdit.INT)

        # Set default values
        self.x_min_input.int_value = 0
        self.x_max_input.int_value = self.voxel_grid.shape[0]
        self.y_min_input.int_value = 0
        self.y_max_input.int_value = self.voxel_grid.shape[1]
        self.z_min_input.int_value = 0
        self.z_max_input.int_value = self.voxel_grid.shape[2]

        # Create labels and layouts for inputs
        self.panel.add_child(gui.Label("X range:"))
        h = gui.Horiz(0.25 * em)
        h.add_child(self.x_min_input)
        h.add_child(gui.Label("to"))
        h.add_child(self.x_max_input)
        self.panel.add_child(h)

        self.panel.add_child(gui.Label("Y range:"))
        h = gui.Horiz(0.25 * em)
        h.add_child(self.y_min_input)
        h.add_child(gui.Label("to"))
        h.add_child(self.y_max_input)
        self.panel.add_child(h)

        self.panel.add_child(gui.Label("Z range:"))
        h = gui.Horiz(0.25 * em)
        h.add_child(self.z_min_input)
        h.add_child(gui.Label("to"))
        h.add_child(self.z_max_input)
        self.panel.add_child(h)

        # Create buttons
        self.crop_button = gui.Button("Crop Voxel Grid")
        self.crop_button.horizontal_padding_em = 0.5
        self.crop_button.vertical_padding_em = 0
        self.crop_button.set_on_clicked(self.crop_voxel_grid)

        self.save_button = gui.Button("Save Cropped Voxel Grid")
        self.save_button.horizontal_padding_em = 0.5
        self.save_button.vertical_padding_em = 0
        self.save_button.set_on_clicked(self.save_cropped_voxel_grid)

        h = gui.Horiz(0.25 * em)
        h.add_child(self.crop_button)
        h.add_child(self.save_button)
        self.panel.add_child(h)

        # Create toggle buttons
        self.toggle_grid_button = gui.Button("Toggle Gridlines")
        self.toggle_grid_button.horizontal_padding_em = 0.5
        self.toggle_grid_button.vertical_padding_em = 0
        self.toggle_grid_button.set_on_clicked(self.toggle_grid)

        self.toggle_axes_button = gui.Button("Toggle Axes")
        self.toggle_axes_button.horizontal_padding_em = 0.5
        self.toggle_axes_button.vertical_padding_em = 0
        self.toggle_axes_button.set_on_clicked(self.toggle_axes)

        h = gui.Horiz(0.25 * em)
        h.add_child(self.toggle_grid_button)
        h.add_child(self.toggle_axes_button)
        self.panel.add_child(h)

        # Add the panel to the window
        self.window.add_child(self.panel)

        # Set the layout callback
        self.window.set_on_layout(self._on_layout)

        # Visualize the voxel grid
        self._update_scene(self.voxel_grid)

    def _on_layout(self, layout_context):
        # Layout the panel and scene widgets
        r = self.window.content_rect
        width = 17 * layout_context.theme.font_size
        self.panel.frame = gui.Rect(r.x, r.y, width, r.height)
        self.scene.frame = gui.Rect(r.x + width, r.y, r.width - width, r.height)

    def _create_grid(self, bounds, spacing):
        # Create a grid based on the bounding box of the voxel grid
        min_bound = bounds.min_bound
        max_bound = bounds.max_bound

        x_range = np.arange(min_bound[0], max_bound[0] + spacing, spacing)
        y_range = np.arange(min_bound[1], max_bound[1] + spacing, spacing)
        z_range = np.arange(min_bound[2], max_bound[2] + spacing, spacing)

        lines = []
        points = []

        # Lines parallel to X-axis
        for y in y_range:
            for z in z_range:
                points.append([min_bound[0], y, z])
                points.append([max_bound[0], y, z])
                lines.append([len(points) - 2, len(points) - 1])

        # Lines parallel to Y-axis
        for x in x_range:
            for z in z_range:
                points.append([x, min_bound[1], z])
                points.append([x, max_bound[1], z])
                lines.append([len(points) - 2, len(points) - 1])

        # Lines parallel to Z-axis
        for x in x_range:
            for y in y_range:
                points.append([x, y, min_bound[2]])
                points.append([x, y, max_bound[2]])
                lines.append([len(points) - 2, len(points) - 1])

        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(points)
        line_set.lines = o3d.utility.Vector2iVector(lines)
        line_set.paint_uniform_color([0.7, 0.7, 0.7])  # Light gray color
        return line_set

    def _add_axis_labels(self, bounds):
        # Remove existing labels
        for label in self.labels:
            self.scene.remove_3d_label(label)
        self.labels = []

        # Create labels at regular intervals along the axes
        num_intervals = 5  # Number of intervals along each axis
        x_ticks = np.linspace(bounds.min_bound[0], bounds.max_bound[0], num_intervals)
        y_ticks = np.linspace(bounds.min_bound[1], bounds.max_bound[1], num_intervals)
        z_ticks = np.linspace(bounds.min_bound[2], bounds.max_bound[2], num_intervals)

        # X-axis labels
        for x in x_ticks:
            pos = [x, bounds.min_bound[1], bounds.min_bound[2]]
            text = f"{x:.1f}"
            label = self.scene.add_3d_label(pos, text)
            self.labels.append(label)

        # Y-axis labels
        for y in y_ticks:
            pos = [bounds.min_bound[0], y, bounds.min_bound[2]]
            text = f"{y:.1f}"
            label = self.scene.add_3d_label(pos, text)
            self.labels.append(label)

        # Z-axis labels
        for z in z_ticks:
            pos = [bounds.min_bound[0], bounds.min_bound[1], z]
            text = f"{z:.1f}"
            label = self.scene.add_3d_label(pos, text)
            self.labels.append(label)

    def _update_scene(self, voxel_grid):
        # Convert voxel grid to point cloud
        x, y, z = np.nonzero(voxel_grid)
        points = np.column_stack((x, y, z)).astype(np.float64)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)

        # Gradient coloring based on depth (Z-axis)
        z_vals = points[:, 2]
        z_min = z_vals.min()
        z_max = z_vals.max()
        if z_max > z_min:
            z_norm = (z_vals - z_min) / (z_max - z_min)
        else:
            z_norm = np.zeros_like(z_vals)

        # Use a colormap to map normalized Z-values to colors
        colormap = cm.get_cmap('viridis')
        colors_mapped = colormap(z_norm)[:, :3]  # Get RGB values

        pcd.colors = o3d.utility.Vector3dVector(colors_mapped)

        # Clear the scene and add new geometry
        self.scene.scene.clear_geometry()
        mat = rendering.MaterialRecord()
        mat.shader = "defaultUnlit"
        mat.point_size = 2
        self.scene.scene.add_geometry("voxel_points", pcd, mat)

        # Add coordinate axes
        if self.axes_visible:
            axes_size = max(self.grid_dims) * 0.1  # 10% of the largest dimension
            axes = o3d.geometry.TriangleMesh.create_coordinate_frame(
                size=axes_size, origin=[0, 0, 0])
            self.scene.scene.add_geometry("axes", axes, rendering.MaterialRecord())

        # Add grid
        bounds = pcd.get_axis_aligned_bounding_box()
        if self.grid_visible:
            grid_spacing = max(self.grid_dims) / 10  # Adjust spacing as needed
            grid = self._create_grid(bounds, spacing=grid_spacing)
            grid_mat = rendering.MaterialRecord()
            grid_mat.shader = "unlitLine"
            grid_mat.line_width = 1.0
            self.scene.scene.add_geometry("grid", grid, grid_mat)

            # Add axis labels
            self._add_axis_labels(bounds)
        else:
            # Remove existing labels
            for label in self.labels:
                self.scene.remove_3d_label(label)
            self.labels = []

        # Set the camera view
        self.scene.setup_camera(60, bounds, bounds.get_center())

    def crop_voxel_grid(self):
        # Get the crop dimensions from the input boxes
        x_min = max(self.x_min_input.int_value, 0)
        x_max = min(self.x_max_input.int_value, self.original_voxel_grid.shape[0])
        y_min = max(self.y_min_input.int_value, 0)
        y_max = min(self.y_max_input.int_value, self.original_voxel_grid.shape[1])
        z_min = max(self.z_min_input.int_value, 0)
        z_max = min(self.z_max_input.int_value, self.original_voxel_grid.shape[2])

        # Validate the cropping range
        if x_min >= x_max or y_min >= y_max or z_min >= z_max:
            gui.Application.instance.post_to_main_thread(
                self.window,
                lambda: gui.Dialog.show_message("Error", "Invalid cropping range.")
            )
            return

        # Crop the voxel grid and update the visualization
        self.voxel_grid = self.original_voxel_grid[x_min:x_max, y_min:y_max, z_min:z_max]
        # Update grid dimensions
        self.grid_dims = self.voxel_grid.shape
        self._update_scene(self.voxel_grid)

    def save_cropped_voxel_grid(self):
        pcd = voxel_to_pcd(self.voxel_grid)
        o3d.io.write_point_cloud(file_name, pcd)

    def toggle_grid(self):
        self.grid_visible = not self.grid_visible
        # Update the scene
        self._update_scene(self.voxel_grid)

    def toggle_axes(self):
        self.axes_visible = not self.axes_visible
        # Update the scene
        self._update_scene(self.voxel_grid)

file_name = "video_processing/point_clouds/@015 255 2024-10-08 04-46-18.pcd"
def main():
    pcd = o3d.io.read_point_cloud(file_name)
    voxel_grid = voxelize(pcd)
    app = VoxelGridApp(voxel_grid)
    gui.Application.instance.run()

if __name__ == '__main__':
    main()
