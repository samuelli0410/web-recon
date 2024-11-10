import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

# Step 1: Load the Point Cloud Data (PCD)
def load_pcd(file_path):
    print("Loading the point cloud...")
    pcd = o3d.io.read_point_cloud(file_path)
    points = np.asarray(pcd.points)
    print(f"Loaded point cloud with {len(points)} points.")
    return points

# Step 2: Subdivide into 8000 regions (20x20x20) and count points in each region
def subdivide_and_count(points):
    print("Subdividing space into 8000 rectangular regions (20x20x20)...")
    
    # Get the bounds of the point cloud
    min_bound = points.min(axis=0)
    max_bound = points.max(axis=0)
    
    # Compute the width of each bin (subdivision) along each dimension
    bin_size = (max_bound - min_bound) / 100.0  # 20 segments per dimension
    
    # Initialize a 3D array to count points in each of the 8000 subregions
    count_grid = np.zeros((100, 100, 100))
    
    # For each point, determine which subregion (bin) it belongs to
    for point in points:
        indices = np.floor((point - min_bound) / bin_size).astype(int)
        indices = np.clip(indices, 0, 99)  # Ensure indices stay within bounds
        count_grid[tuple(indices)] += 1
    
    total_points = len(points)
    print("Finished counting points in each subregion.")
    return count_grid, total_points

# Step 3: Calculate and normalize density levels
def calculate_density(count_grid, total_points):
    print("Calculating density levels...")
    
    # Normalize the counts by the total number of points to get the density in each block
    density_grid = count_grid / total_points  # This gives density as a fraction of total points
    
    print("Density calculation completed.")
    return density_grid

# Step 4: Infer Density from Neighboring Blocks
def infer_density(density_grid):
    print("Inferring density for each block based on its neighbors...")
    
    # Get the shape of the density grid
    depth, height, width = density_grid.shape
    
    # Create an array to store the inferred densities
    inferred_density_grid = np.zeros_like(density_grid)
    
    # Iterate over each block in the grid, avoiding the edges for simplicity
    for i in range(1, depth - 1):
        for j in range(1, height - 1):
            for k in range(1, width - 1):
                # Collect densities of the neighboring blocks
                neighbors = [
                    density_grid[i-1, j, k], density_grid[i+1, j, k],  # neighbors along x-axis
                    density_grid[i, j-1, k], density_grid[i, j+1, k],  # neighbors along y-axis
                    density_grid[i, j, k-1], density_grid[i, j, k+1]   # neighbors along z-axis
                ]
                
                # Infer the density of the current block as the average of neighbors
                inferred_density_grid[i, j, k] = np.mean(neighbors)
    
    print("Density inference completed.")
    return inferred_density_grid

# Step 5: Calculate Average Error of the Inference
def calculate_average_error(actual_grid, inferred_grid):
    # Calculate the absolute error between actual and inferred densities
    error_grid = np.abs(actual_grid - inferred_grid)
    
    # Calculate the average error, ignoring the edges
    depth, height, width = error_grid.shape
    average_error = np.mean(error_grid[1:depth-1, 1:height-1, 1:width-1])
    
    print(f"Average inference error: {average_error:.4f}")
    return average_error

# Visualization function to compare inferred and actual densities (Optional)
def visualize_inferred_vs_actual(actual_grid, inferred_grid, slice_idx=66):
    # Compare a slice of actual and inferred densities
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # Plot actual densities for a specific slice
    axes[0].imshow(actual_grid[slice_idx, :, :], cmap='Greys', interpolation='nearest')
    axes[0].set_title("Actual Density (Slice)")
    axes[0].axis('off')
    
    # Plot inferred densities for the same slice
    axes[1].imshow(inferred_grid[slice_idx, :, :], cmap='Greys', interpolation='nearest')
    axes[1].set_title("Inferred Density (Slice)")
    axes[1].axis('off')
    
    plt.suptitle(f"Comparison of Actual and Inferred Densities at Depth Slice {slice_idx}")
    plt.show()

# Main function to execute the steps and visualize density distribution
if __name__ == "__main__":
    # Path to your point cloud file (PCD)
    file_path = "video_processing/point_clouds/@011 255 2024-10-04 03-20-37.pcd"
    
    # Step 1: Load the PCD file
    points = load_pcd(file_path)
    
    # Step 2: Subdivide space and count points in each subregion
    count_grid, total_points = subdivide_and_count(points)
    
    # Step 3: Calculate density levels based on point counts
    density_grid = calculate_density(count_grid, total_points)
    
    # Step 4: Infer density based on neighboring blocks
    inferred_density_grid = infer_density(density_grid)
    
    # Step 5: Calculate average inference error
    average_error = calculate_average_error(density_grid, inferred_density_grid)
    
    # Optional visualization to compare inferred and actual densities for a particular slice
    visualize_inferred_vs_actual(density_grid, inferred_density_grid, slice_idx=66)
