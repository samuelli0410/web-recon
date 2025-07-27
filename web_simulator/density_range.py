import open3d as o3d
import numpy as np
import pandas as pd
import os

# --- 1) Put all your file paths here ---
PCD_FILES = [
    'video_processing/point_clouds/@012 255 2024-10-04 05-06-11.pcd',
    'video_processing/point_clouds/@014(!005) 255 2024-10-10 06-44-38.pcd',
    'video_processing/point_clouds/@015 255 2024-10-08 04-46-18.pcd',
    'video_processing/point_clouds/@017 255 2024-10-08 05-43-54.pcd',
    'video_processing/point_clouds/@018 255 2024-10-18 04-31-07.pcd',
    'video_processing/point_clouds/@019 255 2024-10-17 22-05-50.pcd',
    'video_processing/point_clouds/@020 255 2024-10-19 15-36-41.pcd',
    'video_processing/point_clouds/@021 255 2024-10-20 02-20-12.pcd',
    'video_processing/point_clouds/@023 255 2024-10-26 20-16-38.pcd',
    'video_processing/point_clouds/@024 255 2024-10-30 04-10-19.pcd',
    'video_processing/point_clouds/!010 255 2024-10-31 18-38-44.pcd',
    'video_processing/point_clouds/@026 255 2024-11-02 16-15-44.pcd',
    'video_processing/point_clouds/@027 255 2024-11-02 17-11-00.pcd',
    'video_processing/point_clouds/@029 255 2024-11-05 04-44-38.pcd',
    'video_processing/point_clouds/@030 255 2024-11-07 22-02-20.pcd',
    'video_processing/point_clouds/@031 255 2024-11-07 23-54-31.pcd',
    'video_processing/point_clouds/@033 255 2024-11-09 18-48-38.pcd',
    'video_processing/point_clouds/@034 255 2024-11-12 04-58-39.pcd',
    'video_processing/point_clouds/@035 255 2024-11-12 06-01-06.pcd',
    'video_processing/point_clouds/@036 255 2024-11-12 22-10-38.pcd',
    'video_processing/point_clouds/@037 255 2024-11-14 00-23-35.pcd',
    'video_processing/point_clouds/@038 255 2024-11-14 01-04-59.pcd',
    'video_processing/point_clouds/@039 255 2024-11-14 04-36-34.pcd',
    'video_processing/point_clouds/@040 255 2024-11-14 05-34-00.pcd',
    'video_processing/point_clouds/@041 255 2024-11-16 19-12-55.pcd',
    'video_processing/point_clouds/@042 255 2024-11-16 20-21-32.pcd',
    'video_processing/point_clouds/@043 255 2024-11-17 00-36-56.pcd',
    'video_processing/point_clouds/@044 255 2024-11-19 03-16-53.pcd',
    'video_processing/point_clouds/@045 255 2024-11-21 05-43-29.pcd',
    'video_processing/point_clouds/@046 255 2024-11-25 19-25-56.pcd',
    'video_processing/point_clouds/@047 255 2024-11-25 20-44-12.pcd',
    'video_processing/point_clouds/@048 255 2024-11-25 21-33-22.pcd',
    'video_processing/point_clouds/@049 255 2024-11-28 13-30-55.pcd',
    'video_processing/point_clouds/@050 255 2024-11-28 14-17-29.pcd',
    'video_processing/point_clouds/@051 255 2024-11-29 15-42-41.pcd',
    'video_processing/point_clouds/@052 255 2024-12-01 14-08-30.pcd',
    'video_processing/point_clouds/@053 255 2024-12-01 15-02-37.pcd',
    'video_processing/point_clouds/@054 255 2024-12-01 15-54-16.pcd',
    'video_processing/point_clouds/@055 255 2024-12-01 18-10-53.pcd',
    'video_processing/point_clouds/@056 255 2024-12-02 23-26-35.pcd',
    'video_processing/point_clouds/@057 255 2024-12-04 23-54-51.pcd',
    'video_processing/point_clouds/@058 255 2024-12-05 04-26-14.pcd',
    'video_processing/point_clouds/@059 255 2024-12-05 04-50-27.pcd',
    'video_processing/point_clouds/@060 255 2024-12-05 12-45-49.pcd',
    'video_processing/point_clouds/@061 255 2024-12-05 13-00-48.pcd',
    'video_processing/point_clouds/@062 255 2024-12-05 13-12-57.pcd',
    'video_processing/point_clouds/@063 255 2024-12-09 22-20-57.pcd',
    'video_processing/point_clouds/@064 255 2024-12-10 17-49-41.pcd',
]

# --- 2) Utility to compute max-fraction in a voxel grid ---
def compute_max_fraction(pcd_path, subdivisions=(100, 100, 75)):
    pcd = o3d.io.read_point_cloud(pcd_path)
    pts = np.asarray(pcd.points)
    total = len(pts)
    
    # bounding box → voxel grid
    mn, mx = pts.min(axis=0), pts.max(axis=0)
    spans = mx - mn
    subs  = np.array(subdivisions)
    voxel_size = spans / subs
    
    # assign to voxels
    idx = ((pts - mn) / voxel_size).astype(int)
    idx = np.clip(idx, 0, subs - 1)
    flat = np.ravel_multi_index(idx.T, subs)
    
    # fraction per voxel
    counts   = np.bincount(flat, minlength=subs.prod()).astype(float)
    fractions = counts / total
    return fractions.max()

if __name__ == "__main__":
    SUBS = (100, 100, 75)
    records = []
    
    for path in PCD_FILES:
        rho_max_frac = compute_max_fraction(path, SUBS)
        fname   = os.path.basename(path)
        date    = fname.split()[1]       # adjust to your naming scheme
        species = fname.split()[0]       # likewise
        records.append({
            'Date of Scan': date,
            'Species': species,
            'Subdivisions': f'{SUBS[0]}×{SUBS[1]}×{SUBS[2]}',
            'rho_max_fraction': rho_max_frac
        })
    
    df = pd.DataFrame(records)
    # show per-web results
    print(df.to_string(index=False))
    
    # summary stats
    mean_frac = df['rho_max_fraction'].mean()
    std_frac  = df['rho_max_fraction'].std()
    print(f"\nMean ρ_max fraction: {mean_frac:.4f}")
    print(f"Std  ρ_max fraction: {std_frac:.4f}")
    
    # optional: export to CSV for LaTeX
    df.to_csv('density_range_summary.csv', index=False)
