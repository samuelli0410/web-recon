import open3d as o3d
import numpy as np
from skimage.morphology import skeletonize_3d
from scipy.spatial import cKDTree


def denoise(pcd, tolerance=5):
    # ---- Step 1: Load the point cloud ----
    points = np.asarray(pcd.points)

    # ---- Step 2: Convert point cloud to voxel grid ----
    # Determine the bounds and voxel resolution
    voxel_size = 0.05  # Adjust voxel size based on your data
    min_bound = points.min(axis=0)
    max_bound = points.max(axis=0)
    dims = np.ceil((max_bound - min_bound) / voxel_size).astype(int) + 1

    # Create an empty occupancy grid
    occupancy_grid = np.zeros(dims, dtype=bool)

    # Map each point to its voxel index and mark as occupied
    indices = np.floor((points - min_bound) / voxel_size).astype(int)
    occupancy_grid[indices[:, 0], indices[:, 1], indices[:, 2]] = True

    # ---- Step 3: Skeletonization using scikit-image ----
    # The skeletonize_3d function expects a binary volume.
    skeleton_grid = skeletonize_3d(occupancy_grid)

    # Extract skeleton voxel indices
    skeleton_voxel_indices = np.argwhere(skeleton_grid)

    # Convert voxel indices back to real-world coordinates
    skeleton_points = skeleton_voxel_indices * voxel_size + min_bound

    # ---- Step 4: Filter original points by distance to the skeleton ----
    threshold = 0.1  # Distance threshold; adjust as needed

    # Build a KD-tree for efficient nearest-neighbor queries on the skeleton
    skeleton_tree = cKDTree(skeleton_points)

    # For each original point, check the distance to the nearest skeleton point
    distances, _ = skeleton_tree.query(points)
    mask = distances <= threshold
    filtered_points = points[mask]

    # Convert filtered points back to an Open3D point cloud
    filtered_pcd = o3d.geometry.PointCloud()
    filtered_pcd.points = o3d.utility.Vector3dVector(filtered_points)

    # ---- Optional: Visualize and save the results ----
    o3d.visualization.draw_geometries([filtered_pcd])


pcd = o3d.io.read_point_cloud("video_processing/point_clouds/@062 255 2024-12-05 13-12-57.pcd")
o3d.visualization.draw_geometries([pcd])
denoised_pcd = denoise(pcd, tolerance=5)
