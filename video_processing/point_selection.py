import open3d as o3d
import numpy as np
import pandas as pd

def main():
    # Load the point cloud from your specified file path.
    pcd_path = "video_processing/point_clouds/tangle001 255 2025-02-08 18-50-40 0.35.pcd"
    pcd = o3d.io.read_point_cloud(pcd_path)
    
    if pcd.is_empty():
        print(f"Error: Failed to load point cloud from {pcd_path}")
        return

    # Set the color of all points to grey.
    grey_color = [0.5, 0.5, 0.5]
    pcd.paint_uniform_color(grey_color)
    
    # Create the VisualizerWithVertexSelection instance.
    vis = o3d.visualization.VisualizerWithVertexSelection()
    vis.create_window(window_name='Select Points')
    vis.add_geometry(pcd)
    
    # Run the GUI. Use Shift + Left Click to select points.
    # The window must be closed to continue.
    vis.run()  # Blocks until the window is closed.
    
    # Retrieve the picked point objects.
    picked_points = vis.get_picked_points()
    print("Picked point objects:", picked_points)
    
    # Extract information from each picked point: coordinates and index.
    data = []
    for pt in picked_points:
        # pt.coord is assumed to be an iterable with three values: x, y, z
        x, y, z = pt.coord
        idx = int(pt.index)
        data.append({"x": x, "y": y, "z": z, "index": idx})
    
    # Create a DataFrame.
    df = pd.DataFrame(data, columns=["x", "y", "z", "index"])
    print("DataFrame created:\n", df)
    
    # Save the DataFrame to a CSV file.
    csv_name = pcd_path.split(".pcd")[0] + " selected.csv"
    df.to_csv(csv_name, index=False)
    print(f"Saved {len(df)} selected points to '{csv_name}'")
    
    # Clean up and close the window.
    vis.destroy_window()

if __name__ == '__main__':
    main()
