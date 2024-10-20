# Created by Github Copilot
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Load the Point Cloud Data (PCD)
def load_pcd(file_path):
    print("Loading the point cloud...")
    pcd = o3d.io.read_point_cloud(file_path)
    points = np.asarray(pcd.points)
    if points.size == 0:
        raise ValueError("The point cloud data is empty.")
    print(f"Loaded point cloud with {len(points)} points.")
    return points

# Step 2: Calculate the fractal dimension using the box-counting method
def calculate_fractal_dimension(points, min_box_size=1, max_box_size=50, num_box_sizes=10):
    print("Calculating fractal dimension using the box-counting method...")
    
    # Get the bounds of the point cloud
    min_bound = points.min(axis=0)
    max_bound = points.max(axis=0)
    
    # Generate box sizes
    box_sizes = np.logspace(np.log10(min_box_size), np.log10(max_box_size), num=num_box_sizes)
    
    # Count the number of occupied boxes for each box size
    occupied_boxes = []
    for box_size in box_sizes:
        num_boxes = np.ceil((max_bound - min_bound) / box_size).astype(int)
        if np.any(num_boxes == 0):
            continue  # Skip this box size if it results in zero boxes in any dimension
        grid = np.zeros(num_boxes, dtype=bool)
        
        # Determine which boxes are occupied
        indices = ((points - min_bound) / box_size).astype(int)
        indices = np.clip(indices, 0, num_boxes - 1)  # Ensure indices are within bounds
        grid[tuple(indices.T)] = True
        
        # Count the number of occupied boxes
        occupied_boxes.append(np.sum(grid))
    
    if len(occupied_boxes) == 0:
        raise ValueError("No valid box sizes found for the given point cloud data.")
    
    # Perform a linear fit to the log-log plot of box size vs. number of occupied boxes
    log_box_sizes = np.log(box_sizes[:len(occupied_boxes)])
    log_occupied_boxes = np.log(occupied_boxes)
    coeffs = np.polyfit(log_box_sizes, log_occupied_boxes, 1)
    fractal_dimension = -coeffs[0]
    
    # Plot the log-log plot
    plt.figure()
    plt.plot(log_box_sizes, log_occupied_boxes, 'o', label='Data')
    plt.plot(log_box_sizes, np.polyval(coeffs, log_box_sizes), '-', label=f'Fit: D={fractal_dimension:.2f}')
    plt.xlabel('log(Box Size)')
    plt.ylabel('log(Number of Occupied Boxes)')
    plt.legend()
    plt.title('Box-Counting Method for Fractal Dimension')
    plt.show()
    
    print(f"Estimated fractal dimension: {fractal_dimension:.2f}")
    return fractal_dimension

if __name__ == "__main__":
    # Example usage
    file_path = '../point_clouds/thin_test.pcd'
    points = load_pcd(file_path)
    fractal_dimension = calculate_fractal_dimension(points)