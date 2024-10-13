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
        # Ensure the indices stay within bounds
        indices = np.clip(indices, 0, 19)
        count_grid[tuple(indices)] += 1
    
    total_points = len(points)
    print("Finished counting points in each subregion.")
    return count_grid, total_points

# Step 3: Calculate the density for each subregion and classify into levels 0-9
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
    
    print("Density classification completed.")
    return density_levels

# Step 4: Record the distribution of the density levels
def record_distribution(density_levels):
    print("Recording the distribution of density levels...")
    
    # Count how many subregions fall into each density level (0-9)
    distribution = np.zeros(10, dtype=int)  # 10 levels, from 0 to 9
    for level in density_levels:
        distribution[level] += 1
    
    print(f"Distribution of density levels: {distribution}")
    return distribution

# Step 5: Compute entropy based on the distribution of density levels
def compute_entropy(distribution, exclude_zero=False):
    print(f"Computing entropy (excluding level 0: {exclude_zero})...")
    
    # Normalize the distribution to get probabilities
    if exclude_zero:
        # Exclude level 0
        non_zero_distribution = distribution[1:]
        total_regions = np.sum(non_zero_distribution)
        probabilities = non_zero_distribution / total_regions
    else:
        # Include all levels
        total_regions = np.sum(distribution)
        probabilities = distribution / total_regions
    
    # Remove zero probabilities to avoid log(0) issues
    non_zero_probs = probabilities[probabilities > 0]
    
    # Compute entropy using Shannon's formula
    entropy = -np.sum(non_zero_probs * np.log2(non_zero_probs))
    
    print(f"Computed entropy: {entropy:.4f}")
    return entropy

# Main function to execute the steps
if __name__ == "__main__":
    # Path to your point cloud file (PCD)
    file_path = "coordinates.pcd"
    
    # Step 1: Load the PCD file
    points = load_pcd(file_path)
    
    # Step 2: Subdivide space and count points in each subregion
    count_grid, total_points = subdivide_and_count(points)
    
    # Step 3: Classify density levels based on point counts (into levels 0-9)
    density_levels = classify_density(count_grid, total_points)
    
    # Step 4: Record the distribution of the density levels
    distribution = record_distribution(density_levels)
    
    # Step 5: Compute entropy considering all levels (including level 0)
    entropy_including_zero = compute_entropy(distribution, exclude_zero=False)
    
    # Step 6: Compute entropy excluding level 0
    entropy_excluding_zero = compute_entropy(distribution, exclude_zero=True)
    
    print(f"Final entropy (including level 0): {entropy_including_zero:.4f}")
    print(f"Final entropy (excluding level 0): {entropy_excluding_zero:.4f}")
