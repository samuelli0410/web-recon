# import open3d as o3d
# import numpy as np

# def load_point_cloud(file_path):
#     print("Loading point cloud data...")
#     pcd = o3d.io.read_point_cloud(file_path)
#     points = np.asarray(pcd.points)
#     if not points.size:
#         raise ValueError("The point cloud data is empty.")
#     print(f"Loaded point cloud with {len(points)} points.")
#     return points
# def compute_voxel_densities(points, num_subdivisions):
#     print(f"Computing voxel densities with {num_subdivisions} subdivisions per axis...")
    
#     # Compute the bounds of the point cloud
#     min_bound = points.min(axis=0)
#     max_bound = points.max(axis=0)
    
#     # Compute the size of each voxel
#     voxel_size = (max_bound - min_bound) / num_subdivisions
    
#     # Map points to voxel indices
#     voxel_indices = ((points - min_bound) / voxel_size).astype(int)
#     voxel_indices = np.clip(voxel_indices, 0, num_subdivisions - 1)
    
#     # Count the number of points in each voxel
#     densities = np.zeros((num_subdivisions, num_subdivisions, num_subdivisions), dtype=int)
#     for idx in voxel_indices:
#         densities[tuple(idx)] += 1
    
#     print("Voxel densities computed.")
#     return densities
# def discretize_densities(densities, num_levels):
#     print(f"Discretizing densities into {num_levels} levels...")
    
#     # Exclude zero densities (empty voxels) from discretization
#     non_zero_densities = densities[densities > 0]
#     if not non_zero_densities.size:
#         raise ValueError("All voxels are empty.")
    
#     # Compute the density thresholds for discretization
#     min_density = non_zero_densities.min()
#     max_density = non_zero_densities.max()
    
#     # Avoid division by zero
#     if max_density == min_density:
#         # All non-zero densities are the same
#         density_levels = np.zeros_like(densities)
#         density_levels[densities > 0] = num_levels - 1  # Assign the highest level
#     else:
#         # Discretize densities into levels from 1 to num_levels - 1
#         density_levels = np.zeros_like(densities)
#         density_levels[densities > 0] = np.ceil(
#             (densities[densities > 0] - min_density) / (max_density - min_density) * (num_levels - 1)
#         ).astype(int)
#         # Ensure levels are within [1, num_levels - 1]
#         density_levels = np.clip(density_levels, 1, num_levels - 1)
    
#     print("Densities discretized.")
#     return density_levels
# def compute_joint_marginal_distributions(density_levels):
#     print("Computing joint and marginal distributions...")
#     num_subdivisions = density_levels.shape[0]
    
#     # Lists to collect the density pairs
#     voxel_densities = []
#     neighbor_avg_densities = []
    
#     # Iterate over voxels, excluding the boundary voxels
#     for x in range(1, num_subdivisions - 1):
#         for y in range(1, num_subdivisions - 1):
#             for z in range(1, num_subdivisions - 1):
#                 voxel_density = density_levels[x, y, z]
#                 if voxel_density == 0:
#                     continue  # Skip empty voxels
                
#                 # Get the density levels of the 6 neighbors
#                 neighbors = [
#                     density_levels[x - 1, y, z],
#                     density_levels[x + 1, y, z],
#                     density_levels[x, y - 1, z],
#                     density_levels[x, y + 1, z],
#                     density_levels[x, y, z - 1],
#                     density_levels[x, y, z + 1],
#                 ]
                
#                 # Exclude empty neighbors
#                 non_zero_neighbors = [d for d in neighbors if d > 0]
#                 if not non_zero_neighbors:
#                     continue  # Skip if all neighbors are empty
                
#                 # Compute the average density level of the neighbors
#                 neighbor_avg_density = int(round(np.mean(non_zero_neighbors)))
                
#                 # Collect the density pair
#                 voxel_densities.append(voxel_density)
#                 neighbor_avg_densities.append(neighbor_avg_density)
    
#     print(f"Collected {len(voxel_densities)} density pairs.")
    
#     # Convert to NumPy arrays
#     voxel_densities = np.array(voxel_densities)
#     neighbor_avg_densities = np.array(neighbor_avg_densities)
    
#     return voxel_densities, neighbor_avg_densities
# def compute_mutual_information(voxel_densities, neighbor_avg_densities, num_levels):
#     print("Computing mutual information...")
#     # Compute the joint histogram
#     joint_histogram, _, _ = np.histogram2d(
#         voxel_densities, neighbor_avg_densities,
#         bins=num_levels - 1,  # Levels from 1 to num_levels - 1
#         range=[[1, num_levels - 1], [1, num_levels - 1]]
#     )
    
#     # Compute marginal histograms
#     voxel_histogram, _ = np.histogram(
#         voxel_densities,
#         bins=num_levels - 1,
#         range=(1, num_levels - 1)
#     )
#     neighbor_histogram, _ = np.histogram(
#         neighbor_avg_densities,
#         bins=num_levels - 1,
#         range=(1, num_levels - 1)
#     )
    
#     # Convert histograms to probabilities
#     joint_prob = joint_histogram / joint_histogram.sum()
#     voxel_prob = voxel_histogram / voxel_histogram.sum()
#     neighbor_prob = neighbor_histogram / neighbor_histogram.sum()
    
#     # Compute mutual information
#     mutual_info = 0.0
#     for i in range(num_levels - 1):
#         for j in range(num_levels - 1):
#             p_xy = joint_prob[i, j]
#             p_x = voxel_prob[i]
#             p_y = neighbor_prob[j]
#             if p_xy > 0 and p_x > 0 and p_y > 0:
#                 mutual_info += p_xy * np.log2(p_xy / (p_x * p_y))
    
#     print(f"Mutual information computed: {mutual_info:.6f} bits.")
#     return mutual_info
# def main():
#     # File path to your PCD file
#     file_path = 'video_processing/point_clouds/@011 255 2024-10-04 03-20-37.pcd'
    
#     # Parameters
#     num_levels = 10  # Number of density levels
#     subdivisions_range = range(100, 121)  # Subdivisions from 100 to 120
    
#     # Load point cloud data
#     points = load_point_cloud(file_path)
    
#     # List to store mutual information values
#     mi_values = []
    
#     for num_subdivisions in subdivisions_range:
#         print(f"\nProcessing with {num_subdivisions} subdivisions...")
        
#         # Compute voxel densities
#         densities = compute_voxel_densities(points, num_subdivisions)
        
#         # Discretize densities
#         density_levels = discretize_densities(densities, num_levels)
        
#         # Compute joint and marginal distributions
#         voxel_densities, neighbor_avg_densities = compute_joint_marginal_distributions(density_levels)
        
#         if len(voxel_densities) == 0:
#             print("No valid density pairs found; skipping this subdivision.")
#             continue
        
#         # Compute mutual information
#         mutual_info = compute_mutual_information(voxel_densities, neighbor_avg_densities, num_levels)
#         mi_values.append(mutual_info)
    
#     if mi_values:
#         average_mi = np.mean(mi_values)
#         print(f"\nAverage mutual information over subdivisions {subdivisions_range.start} to {subdivisions_range.stop - 1}: {average_mi:.6f} bits.")
#     else:
#         print("No mutual information values computed.")
# if __name__ == '__main__':
#     main()

import open3d as o3d
import numpy as np

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
    bin_size = (max_bound - min_bound) / 20.0  # 20 segments per dimension
    
    # Initialize a 3D array to count points in each of the 8000 subregions
    count_grid = np.zeros((20, 20, 20))
    
    # For each point, determine which subregion (bin) it belongs to
    for point in points:
        indices = np.floor((point - min_bound) / bin_size).astype(int)
        indices = np.clip(indices, 0, 19)  # Ensure indices stay within bounds
        count_grid[tuple(indices)] += 1
    
    total_points = len(points)
    print("Finished counting points in each subregion.")
    return count_grid, total_points

# Step 3: Classify density levels based on the counts
def classify_density(count_grid, total_points):
    print("Classifying density levels...")
    
    # Flatten the 3D count grid for easier processing
    flattened_grid = count_grid.flatten()
    
    # Normalize the counts by dividing by the total number of points to get the density
    densities = flattened_grid / total_points
    
    # Initialize the classification array for density levels (0-9)
    density_levels = np.zeros_like(densities, dtype=int)
    
    # Classify density into levels (0-9), where level 0 is for zero density and levels 1-9 for non-zero densities
    for i, density in enumerate(densities):
        if density == 0:
            density_levels[i] = 0  # Level 0 for density exactly 0
        elif density > 0.005:
            density_levels[i] = 9  # Level 9 for densities greater than 0.005
        else:
            density_levels[i] = int(np.ceil(density / (0.005 / 9)))  # Levels 1 to 9
    
    # Reshape density levels back to 3D for adjacency analysis
    density_levels_3d = density_levels.reshape((20, 20, 20))
    print("Density classification completed.")
    return density_levels_3d

# Step 4: Compute Mutual Information for Adjacent Regions (Skip if Self Level is 0)
def compute_mutual_information(density_levels_3d):
    print("Computing mutual information between adjacent blocks, skipping if self level is 0...")

    # Initialize a dictionary to store mutual information along each axis
    mi_values = {'x': 0, 'y': 0, 'z': 0}

    # Loop over each axis (x, y, z) to compute MI between adjacent blocks
    for axis in range(3):
        axis_name = ['x', 'y', 'z'][axis]
        
        # Initialize joint counts for pairs of density levels (0-9)
        joint_counts = np.zeros((10, 10))

        # Loop over the 3D grid to populate joint counts for adjacent pairs
        for i in range(20):
            for j in range(20):
                for k in range(20):
                    # Get the density level of the current block
                    current_level = int(density_levels_3d[i, j, k])
                    
                    # Skip if the current (self) level is 0
                    if current_level == 0:
                        continue

                    # Determine the adjacent block based on the current axis
                    if axis == 0 and i < 19:  # X-axis (adjacent in i+1)
                        adjacent_level = int(density_levels_3d[i+1, j, k])
                    elif axis == 1 and j < 19:  # Y-axis (adjacent in j+1)
                        adjacent_level = int(density_levels_3d[i, j+1, k])
                    elif axis == 2 and k < 19:  # Z-axis (adjacent in k+1)
                        adjacent_level = int(density_levels_3d[i, j, k+1])
                    else:
                        continue
                    
                    # Update joint count for the pair (current_level, adjacent_level)
                    joint_counts[current_level, adjacent_level] += 1

        # Normalize to get joint probabilities
        joint_probabilities = joint_counts / joint_counts.sum()
        
        # Compute marginal probabilities from joint distribution
        marginal_prob_x = np.sum(joint_probabilities, axis=1)
        marginal_prob_y = np.sum(joint_probabilities, axis=0)

        # Calculate mutual information for the current axis
        mi = 0
        for x in range(10):
            for y in range(10):
                if joint_probabilities[x, y] > 0:
                    mi += joint_probabilities[x, y] * np.log2(joint_probabilities[x, y] / (marginal_prob_x[x] * marginal_prob_y[y]))

        mi_values[axis_name] = mi
        print(f"Mutual Information along {axis_name}-axis (skipping if self level is 0): {mi:.4f}")

    return mi_values

# Main function to execute the steps and compute mutual information
if __name__ == "__main__":
    # Path to your point cloud file (PCD)
    file_path = "video_processing/point_clouds/@020 255 2024-10-19 15-36-41.pcd"
    
    # Step 1: Load the PCD file
    points = load_pcd(file_path)
    
    # Step 2: Subdivide space and count points in each subregion
    count_grid, total_points = subdivide_and_count(points)
    
    # Step 3: Classify density levels based on point counts (into levels 0-9)
    density_levels_3d = classify_density(count_grid, total_points)
    
    # Step 4: Compute mutual information for adjacent blocks along each axis, skipping if self level is 0
    mutual_info = compute_mutual_information(density_levels_3d)
    
    # Print final mutual information values
    print(f"Mutual information (x-axis): {mutual_info['x']:.4f}")
    print(f"Mutual information (y-axis): {mutual_info['y']:.4f}")
    print(f"Mutual information (z-axis): {mutual_info['z']:.4f}")