import open3d as o3d
import numpy as np
import scipy.ndimage
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


# Step 2: Subdivide the space and calculate density
def calculate_density(points, num_subdivisions):
    print(f"Calculating density with {num_subdivisions} subdivisions...")
    
    # Get the bounds of the point cloud
    min_bound = points.min(axis=0)
    max_bound = points.max(axis=0)
    
    # Adjust subdivisions for the second axis (index 1) to be 3/4 the original subdivisions
    adjusted_subdivisions = np.array([num_subdivisions, int(num_subdivisions * 0.75), num_subdivisions])
    
    # Calculate the size of each subdivision (voxel)
    subdivision_size = (max_bound - min_bound) / adjusted_subdivisions
    voxel_volume = np.prod(subdivision_size)  # Volume of each voxel in 3D space

    # Initialize array to store density counts per voxel
    density_counts = np.zeros(adjusted_subdivisions, dtype=float)
    
    # Calculate voxel indices for each point
    indices = ((points - min_bound) / subdivision_size).astype(int)
    indices = np.clip(indices, 0, adjusted_subdivisions - 1)  # Ensure indices are within grid bounds
    flat_indices = np.ravel_multi_index(indices.T, adjusted_subdivisions)
    
    # Count points in each voxel
    np.add.at(density_counts, np.unravel_index(flat_indices, adjusted_subdivisions), 1)
    
    # Convert counts to densities by dividing by voxel volume
    density = density_counts / voxel_volume
    return density, adjusted_subdivisions


# Step 3: Compute the \( p \)-Average
def p_average(density, p=1):
    print(f"Inferring density for each block based on its neighbors using p-power mean with p = {p:.4f}")
    
    # Kernel for averaging densities of 6 neighboring blocks (3D cross-shaped kernel)
    kernel = np.zeros((3, 3, 3))
    kernel[0, 1, 1] = 1/6  # Neighbor in -x direction
    kernel[2, 1, 1] = 1/6  # Neighbor in +x direction
    kernel[1, 0, 1] = 1/6  # Neighbor in -y direction
    kernel[1, 2, 1] = 1/6  # Neighbor in +y direction
    kernel[1, 1, 0] = 1/6  # Neighbor in -z direction
    kernel[1, 1, 2] = 1/6  # Neighbor in +z direction

    if p > 0:
        density_p = np.power(density, p)
        inferred_density_p = scipy.ndimage.convolve(density_p, kernel, mode="constant", cval=0.0)
        inferred_density = np.clip(inferred_density_p, 0, None) ** (1/p)
    else:
        # Geometric mean when p = 0: exp(avg(log(density)))
        eps = 1e-6  # Small value to avoid log(0)
        density_log = np.log(density + eps)
        inferred_density_log = scipy.ndimage.convolve(density_log, kernel, mode="constant", cval=0.0)
        inferred_density = np.exp(inferred_density_log)
    return inferred_density


# Step 4: Compute Harmonicity
def p_harmonicity(density, p=1, q=1):
    """
    Computes the error between density and \( p \)-averaged density using averaged \( L^q \)-norm.
    
    Parameters:
        density (numpy.ndarray): The original density array.
        p (float): The power for the \( p \)-power mean.
        q (float): The power for the \( L^q \)-norm.
    
    Returns:
        float: The harmonicity value (averaged \( L^q \)-norm of the error).
    """
    assert q > 0, "q must be positive"
    inferred_density = p_average(density, p)
    err = np.abs(density - inferred_density)
    return np.mean(err ** q) ** (1/q)


# Step 5: Plot Error vs. \( p \)
def plot_error_vs_p(density, q=1, delta=0.01, max_p=3.0):
    print("Plotting error vs. p...")
    p_range = np.arange(delta, max_p + delta, delta)
    errors = [p_harmonicity(density, p, q) for p in p_range]
    best_p = p_range[np.argmin(errors)]
    print(f"Best p: {best_p}")
    
    plt.figure(figsize=(10, 6))
    plt.plot(p_range, errors, label=f"L^{q} Error")
    plt.xlabel("p")
    plt.ylabel(f"L^{q} Error")
    plt.title(f"L^{q} Error vs. p")
    plt.legend()
    plt.grid()
    plt.show()
    return best_p


if __name__ == "__main__":
    # Example usage
    file_path = 'video_processing/point_clouds/@020 255 2024-10-19 15-36-41.pcd'
    points = load_pcd(file_path)
    
    # Calculate density
    num_subdivisions = 20  # Adjust this value as needed
    density, adjusted_subdivisions = calculate_density(points, num_subdivisions)
    
    # Compute harmonicity for a given p and q
    p = 1.0
    q = 1.0
    harmonicity = p_harmonicity(density, p, q)
    
    
    # Plot error vs. p
    best_p = plot_error_vs_p(density, q=q)
    print(f"Optimal p value: {best_p}")
    print(f"Harmonicity (L^{q} error) for p = {p}: {harmonicity:.4f}")