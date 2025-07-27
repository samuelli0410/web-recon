import numpy as np
import open3d as o3d


def load_pcd_array(file_path):
    """
    Loads a PCD file and returns a NumPy array of points.
    
    Parameters:
        file_path (str): Path to the PCD file.
    
    Returns:
        numpy.ndarray: A 2D array of shape (N, 3) containing the (x, y, z) coordinates of the point cloud.
    """
    pcd = o3d.io.read_point_cloud(file_path)
    points = np.asarray(pcd.points)
    return points


def calculate_density(points, M):
    """
    Calculate the density of points in a 3D grid.
    
    Parameters:
        points (numpy.ndarray): A 2D array of shape (N, 3) containing (x, y, z) coordinates of the point cloud.
        M (int): The number of subdivisions along each axis.
    
    Returns:
        numpy.ndarray: A 3D array of shape (M, M, M) where each entry (i, j, k) counts the number of points in the corresponding voxel.
    """
    assert M > 0, "Number of subdivisions must be positive"
    print(f"Calculating density with {M} subdivisions...")

    # Compute the bounding box of the point cloud
    min_bounds = points.min(axis=0)
    max_bounds = points.max(axis=0)
    box_size = max_bounds - min_bounds

    # Compute the size of each subdivision (voxel)
    subdivision_size = box_size / M
    voxel_volume = np.prod(subdivision_size)

    # Map points to voxel indices
    scaled_indices = ((points - min_bounds) / subdivision_size).astype(int)
    scaled_indices = np.clip(scaled_indices, 0, M - 1)  # Ensure indices are within grid bounds

    # Compute density
    flat_indices = np.ravel_multi_index(scaled_indices.T, (M, M, M))
    density_counts_flat = np.bincount(flat_indices, minlength=M**3)
    density = density_counts_flat.reshape((M, M, M)) / voxel_volume
    return density


def calculate_density_levels(points, M, num_levels=10, max_density=None):
    """
    Calculate density levels by quantizing the density into discrete levels.
    
    Parameters:
        points (numpy.ndarray): A 2D array of shape (N, 3) containing (x, y, z) coordinates of the point cloud.
        M (int): The number of subdivisions along each axis.
        num_levels (int): Number of density levels. Default is 10.
        max_density (float): Maximum density value for normalization. Default is None.
    
    Returns:
        tuple: 
            - density (numpy.ndarray): The raw density values in the grid.
            - density_levels (numpy.ndarray): The quantized density levels in the grid.
    """
    print(f"Calculating density levels with {M} subdivisions...")

    density = calculate_density(points, M)
    density_levels = density.copy()
    
    # Scale non-zero density levels to fit within [1, num_levels - 1]
    if max_density is None:
        max_density = density.max()
    print("Normalize with max density:", max_density)
    if max_density > 0:
        nonzero_indices = density > 0  # Identify non-zero density voxels
        density_levels[nonzero_indices] = (
            (density_levels[nonzero_indices] * (num_levels) / max_density).astype(int) + 1
        )
    else:
        density_levels = density_levels.astype(int)  # Convert to int if all densities are zero
    
    # Ensure density levels are within [0, num_levels]
    density_levels = np.clip(density_levels, 0, num_levels).astype(int)
    return density, density_levels


def record_distribution(density_levels, num_levels=10):
    """
    Record the distribution of density levels.
    
    Parameters:
        density_levels (numpy.ndarray): A 3D array of density levels.
        num_levels (int): Number of density levels. Default is 10.
    
    Returns:
        numpy.ndarray: A 1D array where each entry corresponds to the count of subregions in a density level.
    """
    print("Recording the distribution of density levels...")
    
    # Count how many subregions fall into each density level
    distribution = np.zeros(num_levels + 1, dtype=int)
    for level in range(num_levels + 1):
        l_indices = density_levels == level
        distribution[level] = np.sum(l_indices)
    
    print(f"Distribution of density levels: {distribution}")
    return distribution


if __name__ == "__main__":
    # Path to the PCD file
    file_path = "../point_clouds/@011.pcd"
    
    # Load the point cloud data
    points = load_pcd_array(file_path)
    
    # Calculate density and density levels
    M = 20  # Number of subdivisions along each axis
    density, density_levels = calculate_density_levels(points, M=M, num_levels=10)
    
    # Print results
    print("Density:")
    print(density)
    print("Density shape:", density.shape)
    
    # Record the distribution of density levels
    distribution = record_distribution(density_levels)
    print("Distribution of Density Levels:")
    print(distribution)
