import open3d as o3d
import numpy as np

def visualize_pcd(file_path, point_size=5.0):
    # Load the point cloud
    pcd = o3d.io.read_point_cloud(file_path)
    
    # Check if point cloud data is loaded correctly
    if pcd.is_empty():
        print("Error: Point cloud data is empty or could not be loaded.")
        return
    
    # Paint all points black
    black = np.array([0.0, 0.0, 0.0])
    pcd.paint_uniform_color(black)
    
    # Create a visualizer and add the point cloud
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pcd)
    
    # Set the point size
    render_opt = vis.get_render_option()
    render_opt.point_size = point_size
    
    # Run the visualizer
    vis.run()
    vis.destroy_window()

# Example usage:
file_path = "video_processing/point_clouds/46_cubes_cubes/cube_1_1_1.pcd"
visualize_pcd(file_path, point_size=10.0)
