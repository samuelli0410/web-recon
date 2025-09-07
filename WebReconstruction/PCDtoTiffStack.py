import argparse
import numpy as np
import tifffile
import os
import open3d as o3d


def pcd_to_tiff_slices(
    pcd_path: str,
    out_dir: str,
    grid=(256, 256, 256),       # (nx, ny, nz)
    mode: str = "occupancy",    # "occupancy" | "density"
    normalize: bool = False,    # scale nonzero voxel values to 0–255
    padding: float = 0.0,       # fraction of bbox diagonal to pad
    prefix: str = "slice",      # filename prefix
    start_index: int = 0,       # first slice index
    overwrite: bool = True,     # create dir if needed
):
    pcd = o3d.io.read_point_cloud(pcd_path)
    pts = np.asarray(pcd.points, dtype=np.float64)
    if pts.size == 0:
        raise ValueError("Empty point cloud.")

    # --- Compute bounds + optional padding ---
    mins, maxs = pts.min(axis=0), pts.max(axis=0)
    diag = np.linalg.norm(maxs - mins)
    pad = padding * diag
    mins -= pad
    maxs += pad
    span = np.maximum(maxs - mins, 1e-9)

    nx, ny, nz = map(int, grid)

    uvw = (pts - mins) / span  # [0,1]^3
    ix = np.clip(np.floor(uvw[:, 0] * nx), 0, nx - 1).astype(np.int64)
    iy = np.clip(np.floor(uvw[:, 1] * ny), 0, ny - 1).astype(np.int64)
    iz = np.clip(np.floor(uvw[:, 2] * nz), 0, nz - 1).astype(np.int64)

    # --- Allocate volume as (Z, Y, X) ---
    if mode == "occupancy":
        vol = np.zeros((nz, ny, nx), dtype=np.uint8)
        vol[iz, iy, ix] = 1
        if normalize:
            vol *= 255
    elif mode == "density":
        vol = np.zeros((nz, ny, nx), dtype=np.uint32)
        np.add.at(vol, (iz, iy, ix), 1)
        if normalize:
            nonzero = vol > 0
            if nonzero.any():
                vol = (vol / vol[nonzero].max() * 255.0).round().astype(np.uint8)
            else:
                vol = vol.astype(np.uint8)
        else:
            if vol.max() <= np.iinfo(np.uint16).max:
                vol = vol.astype(np.uint16)
    else:
        raise ValueError("mode must be 'occupancy' or 'density'")
    

    if not os.path.exists(out_dir):
        if overwrite:
            os.makedirs(out_dir, exist_ok=True)
        else:
            raise FileExistsError(f"Output dir {out_dir!r} does not exist and overwrite=False.")
        
    z_digits = max(4, len(str(start_index + vol.shape[0] - 1)))
    paths = []
    for z in range(vol.shape[0]):
        fname = f"{prefix}_z{start_index + z:0{z_digits}d}.tif"
        fpath = os.path.join(out_dir, fname)
        tifffile.imwrite(fpath, vol[z])  # 2D TIFF (Y, X)
        paths.append(fpath)

    return paths, vol


import glob, numpy as np, tifffile, napari

def visualize(filepath):
    files = sorted(glob.glob(f"{filepath}/*.tif"))
    vol = np.stack([tifffile.imread(f) for f in files], axis=0)  # (Z, Y, X)
    viewer = napari.Viewer(ndisplay=3)
    viewer.add_image(
        vol,
        name="stack",
        rendering="attenuated_mip",  # options: 'mip', 'translucent', 'iso', 'additive'
        contrast_limits=[vol.min(), vol.max()],
        scale=(1.0, 1.0, 1.0),       # set to voxel size (z, y, x) if you know it
    )
    napari.run()







if __name__ == "__main__":
    # paths, vol = pcd_to_tiff_slices( "C:/Users/samue/Downloads/Research/Spider/PCD Files/TestingPCD/sparse3 255 2024-11-30 11-29-33.pcd", "tiff_slices/", grid=(256, 256, 180), mode="density", normalize=True, padding=0.02, prefix="pcd", start_index=0)

    # print(f"Wrote {len(paths)} slices to tiff_slices/")

    visualize("tiff_slices/")