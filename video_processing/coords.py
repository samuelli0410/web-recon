import open3d as o3d
import numpy as np

def main():
    cloud = o3d.io.read_point_cloud("video_processing/point_clouds/@013 255 2024-10-05 03-18-53(spiderRemoved).pcd")
    # Set all points to black
    cloud.colors = o3d.utility.Vector3dVector(np.zeros((len(cloud.points), 3)))  # Set all points to black


    

    # Create a visualizer object
    vis = o3d.visualization.Visualizer()
    vis.create_window()

    # Add the point cloud to the visualizer
    vis.add_geometry(cloud)

    # Get render option and set point size
    render_option = vis.get_render_option()
    render_option.point_size = 0.01  # Adjust point size (e.g., smaller than default, 1.0)

    # Visualize the point cloud
    vis.run()
    vis.destroy_window()

if __name__ == "__main__":
    main()
