#!/usr/bin/env python3
"""
spider_web_skeleton_to_pcd.py

Loads a spider‐web PCD (or PLY), voxelizes it, skeletonizes it,
then writes out a binary PCD (v0.7) containing only the skeleton points.
Manually writes the PCD to avoid Open3D writer issues.

Usage:
    python spider_web_skeleton_to_pcd.py

Dependencies:
    pip install numpy scikit-image open3d
"""

import numpy as np
from skimage.morphology import skeletonize_3d
import open3d as o3d
import os

# -------------------- USER PARAMETERS --------------------
INPUT_PCD_PATH = "video_processing/point_clouds/@011 255 2024-10-04 03-20-37.pcd"
VOXEL_SIZE = 2.0
OUTPUT_PCD_PATH = "video_processing/point_clouds/11_skeleton.pcd"
# ----------------------------------------------------------


def load_pcd_numpy(pcd_path):
    with open(pcd_path, 'rb') as f:
        data = f.read()
    marker = b"DATA binary\n"
    idx = data.find(marker)
    if idx == -1:
        raise RuntimeError("Could not find 'DATA binary' in PCD header.")
    header = data[:idx].decode('utf-8', errors='ignore').splitlines()
    pts = None
    for line in header:
        if line.startswith("POINTS"):
            pts = int(line.split()[1]); break
    if pts is None:
        raise RuntimeError("Missing POINTS count in header.")
    start = idx + len(marker)
    floats = np.frombuffer(data[start:], dtype=np.float32, count=pts*3)
    return floats.reshape((pts, 3))


def load_point_cloud(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pcd":
        return load_pcd_numpy(path)
    pcd = o3d.io.read_point_cloud(path)
    if pcd.is_empty():
        raise RuntimeError(f"Failed to load or empty point cloud: {path}")
    return np.asarray(pcd.points)


def voxelize_point_cloud(points, voxel_size):
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    dims = np.ceil((maxs - mins) / voxel_size).astype(int) + 2
    origin = mins - voxel_size
    occ = np.zeros(dims, bool)
    idx = np.floor((points - origin) / voxel_size).astype(int)
    idx = np.clip(idx, 0, dims - 1)
    occ[idx[:,0], idx[:,1], idx[:,2]] = True
    return occ, origin, 1.0/voxel_size


def extract_skeleton(volume):
    sk = skeletonize_3d(volume.astype(np.uint8))
    return sk.astype(bool)


def skeleton_voxels_to_points(sk, origin, inv_vs):
    ijk = np.argwhere(sk)
    vs = 1.0 / inv_vs
    pts = origin + (ijk + 0.5) * vs
    return pts.astype(np.float32)


def write_pcd(path, points):
    """
    Write a binary PCD v0.7 with FIELDS x y z for a Nx3 float32 array.
    """
    n = points.shape[0]
    header = (
        "# .PCD v0.7 - Point Cloud Data file format\n"
        "VERSION 0.7\n"
        "FIELDS x y z\n"
        "SIZE 4 4 4\n"
        "TYPE F F F\n"
        "COUNT 1 1 1\n"
        f"WIDTH {n}\n"
        "HEIGHT 1\n"
        "VIEWPOINT 0 0 0 1 0 0 0\n"
        f"POINTS {n}\n"
        "DATA binary\n"
    )
    with open(path, 'wb') as f:
        f.write(header.encode('utf-8'))
        f.write(points.tobytes())


def main():
    print(f"Loading point cloud: {INPUT_PCD_PATH}")
    raw = load_point_cloud(INPUT_PCD_PATH)
    print(f"Loaded {len(raw):,} points.")

    print(f"Voxelizing (size={VOXEL_SIZE})…")
    occ, origin, inv_vs = voxelize_point_cloud(raw, VOXEL_SIZE)
    print(f"Grid {occ.shape}, occupied: {occ.sum():,}")

    print("Skeletonizing…")
    sk = extract_skeleton(occ)
    print(f"Skeleton voxels: {sk.sum():,}")

    print("Converting voxels → world points…")
    sk_pts = skeleton_voxels_to_points(sk, origin, inv_vs)
    print(f"Skeleton points: {len(sk_pts):,}")

    print(f"Writing skeleton PCD to: {OUTPUT_PCD_PATH}")
    write_pcd(OUTPUT_PCD_PATH, sk_pts)
    print("Done.")

if __name__ == "__main__":
    main()
