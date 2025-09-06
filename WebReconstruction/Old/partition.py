#!/usr/bin/env python3
"""
split_pcd.py

Loads a PCD file, splits its bounding box into n³ equally-sized cubes,
and writes out one PCD per cube containing the points inside it.

Dependencies:
    pip install open3d numpy
"""

import os
import numpy as np
import open3d as o3d

def split_pcd_into_cubes(input_path: str, n: int, output_dir: str):
    # Load the point cloud
    pcd = o3d.io.read_point_cloud(input_path)
    points = np.asarray(pcd.points)
    if points.size == 0:
        raise ValueError(f"No points found in '{input_path}'")

    # Compute the overall bounds
    min_bound = points.min(axis=0)
    max_bound = points.max(axis=0)
    dims = max_bound - min_bound

    # Make sure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Compute the size of each sub‐cube
    step = dims / n

    # Optionally carry over colors or other attributes
    has_colors = (pcd.has_colors())
    colors = np.asarray(pcd.colors) if has_colors else None

    count = 0
    # Iterate over i, j, k to define each cube
    for i in range(n):
        for j in range(n):
            for k in range(n):
                # Define bounds for this cube
                low  = min_bound + np.array([i, j, k]) * step
                high = low + step

                # Find points within the cube (inclusive of lower, exclusive of upper)
                mask = np.all((points >= low) & (points < high), axis=1)
                sub_pts = points[mask]
                if sub_pts.shape[0] == 0:
                    continue  # skip empty cubes

                # Build a new PointCloud for the subset
                sub_pcd = o3d.geometry.PointCloud()
                sub_pcd.points = o3d.utility.Vector3dVector(sub_pts)
                if has_colors:
                    sub_pcd.colors = o3d.utility.Vector3dVector(colors[mask])

                # Write it out
                filename = f"cube_{i}_{j}_{k}.pcd"
                out_path = os.path.join(output_dir, filename)
                o3d.io.write_point_cloud(out_path, sub_pcd)
                count += 1

    print(f"Done! Wrote {count} non-empty cubes to '{output_dir}'.")


if __name__ == "__main__":
    # ———– EDIT THESE ———–
    INPUT_PCD_PATH = "video_processing/point_clouds/cube_1_0_1.pcd"
    N = 3                     # will split into 3×3×3 = 27 cubes
    OUTPUT_FOLDER = "video_processing/point_clouds/46_cubes_cubes"
    # ————————————————

    split_pcd_into_cubes(INPUT_PCD_PATH, N, OUTPUT_FOLDER)
