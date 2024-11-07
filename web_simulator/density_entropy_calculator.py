import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt

# File path
num_levels = 10  # Specify the number of density levels
fp = 'video_processing/point_clouds/@011 255 2024-10-04 03-20-37.pcd'

# Step 1: Load the Point Cloud Data (PCD)
def load_pcd(file_path):
    print("Loading the point cloud...")
    pcd = o3d.io.read_point_cloud(file_path)
    points = np.asarray(pcd.points)
    if points.size == 0:
        raise ValueError("The point cloud data is empty.")
    print(f"Loaded point cloud with {len(points)} points.")
    return points

# Step 2: Subdivide the space and calculate density levels
def calculate_density_levels(points, num_subdivisions):
    print(f"Calculating density levels with {num_subdivisions} subdivisions...")
    
    # Get the bounds of the point cloud
    min_bound = points.min(axis=0)
    max_bound = points.max(axis=0)
    
    # Calculate the size of each subdivision
    subdivision_size = (max_bound - min_bound) / num_subdivisions
    
    # Initialize the density levels array
    density_levels = np.zeros(num_subdivisions**3, dtype=int)
    
    # Calculate the density levels
    indices = ((points - min_bound) / subdivision_size).astype(int)
    indices = np.clip(indices, 0, num_subdivisions - 1)  # Ensure indices are within bounds
    flat_indices = np.ravel_multi_index(indices.T, (num_subdivisions, num_subdivisions, num_subdivisions))
    for idx in flat_indices:
        density_levels[idx] += 1
    
    # Set the 0th level to regions with exactly 0 density, normalize other levels to fit remaining range
    max_density = density_levels.max()
    if max_density > 0:
        nonzero_indices = density_levels > 0
        density_levels[nonzero_indices] = (density_levels[nonzero_indices] * (num_levels - 1) / max_density).astype(int) + 1
    
    # Clip density levels to ensure they fall within [0, num_levels - 1]
    density_levels = np.clip(density_levels, 0, num_levels - 1)
    
    print("Density levels:", density_levels)
    return density_levels

# Step 3: Record the distribution of the density levels
def record_distribution(density_levels):
    print("Recording the distribution of density levels...")
    
    # Count how many subregions fall into each density level
    distribution = np.zeros(num_levels, dtype=int)
    for level in density_levels:
        distribution[level] += 1
    
    print(f"Distribution of density levels: {distribution}")
    return distribution

# Step 4: Compute entropy based on the distribution of density levels
def compute_entropy(distribution, exclude_zero=False):
    print(f"Computing entropy (excluding level 0: {exclude_zero})...")
    
    # Normalize the distribution to get probabilities
    if exclude_zero:
        distribution = distribution[1:]  # Exclude level 0
    probabilities = distribution / distribution.sum()
    
    # Compute entropy
    entropy = -np.sum(probabilities[probabilities > 0] * np.log2(probabilities[probabilities > 0]))
    
    print(f"Computed entropy: {entropy}")
    return entropy

def plot_distribution(distribution, exclude_zero=False, label_every_n=5):
    if exclude_zero:
        distribution = distribution[1:]  # Exclude level 0 from the plot
    plt.figure(figsize=(12, 6))
    x_values = range(1, len(distribution) + 1)  # Start x-axis from level 1
    bars = plt.bar(x_values, distribution, color="skyblue")
    plt.xlabel("Density Level (excluding level 0)" if exclude_zero else "Density Level")
    plt.ylabel("Number of Subregions")
    plt.title("Distribution of Density Levels (Excluding Level 0)" if exclude_zero else "Distribution of Density Levels")

    # Label every nth bar
    for i, (bar, count) in enumerate(zip(bars, distribution)):
        height = bar.get_height()
        if height > 0 and i % label_every_n == 0:  # Only label every nth bar
            plt.text(bar.get_x() + bar.get_width() / 2, height, f'{int(count)}', ha='center', va='bottom')

    plt.show()

if __name__ == "__main__":
    # Example usage
    file_path = fp 
    points = load_pcd(file_path)
    density_levels = calculate_density_levels(points, 100)
    distribution = record_distribution(density_levels)
    print(f"Distribution of density levels: {distribution}")
    entropy_including_zero = compute_entropy(distribution, exclude_zero=False)
    entropy_excluding_zero = compute_entropy(distribution, exclude_zero=True)
    print(f"Final entropy (including level 0): {entropy_including_zero:.4f}")
    print(f"Final entropy (excluding level 0): {entropy_excluding_zero:.4f}")

    plot_distribution(distribution, exclude_zero=True)
