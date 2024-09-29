from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import List, Optional, Tuple
import os
import pandas as pd

import cv2
import numpy as np
import open3d as o3d

from sklearn.linear_model import LinearRegression
from skimage.morphology import skeletonize


def box_boundary():
    pass

cut_front_frames = 400
cut_back_frames = 200

left_border = 530
right_border = -550
top_border = 500 #310
bottom_border = -30

pixel_threshold = 0.4


px_per_mm = 4.6
distance_data = pd.read_csv("video_processing/distance_records/@006r 255 distance data 2024-09-12 22-59-36.csv")

def camera_speed_factor(distance_data: pd.DataFrame):
    X = distance_data[['Time']].values
    y = distance_data['Distance'].values

    model = LinearRegression()
    model.fit(X, y)

    return model.coef_[0]

m = camera_speed_factor(distance_data)


def removespider(points: List[Tuple[int, int, float]], threshold: int = 7):
#    points = [(sub[0], sub[2], sub[1]) for sub in pons]
    points.sort(key=lambda p: p[1])
    points.sort(key=lambda p: p[0])
    points.sort(key=lambda p: p[2])

    prevpointx = 0
    prevpointy = 0
    prevpointz = 0
    currentrowrunning = 0

    removecolumn = set()


    for point in points:
        x, y, z = point
        if z == prevpointz:
            if x == prevpointx:
                #check that it is still the same column
                if  y >= prevpointy & y <= prevpointy +1: #check that the y is "consectutive"
                    currentrowrunning += 1
                    if currentrowrunning >= threshold: #if the consecutive count passes threshhold, add them to the removed list
                        removecolumn.add((x,z))
                        currentrowrunning = 0
                else:
                    currentrowrunning = 0
            else:
                    currentrowrunning = 0
        else:
                    currentrowrunning = 0
        prevpointz = z
        prevpointx = x
        prevpointy = y

    def removesomecolumn(points: List[Tuple[int, int, float]], removed: set[Tuple[int, float]]):
        result = []

        for point in points:
            x, y, z = point
            dead = False
            for remove in removed:
                a, b = remove
                if x == a and z == b:
                    dead = True
            if not dead:
                result.append(point)
            
        return result
    

    return removesomecolumn(points, removecolumn)


def findspider(points: List[Tuple[int, int, float]], threshold: int = 0):
    points.sort(key=lambda p: p[0])
    for point in points:
        x, y, z = point

def process_frame_grey(frame_data):
    frame, frame_count, _ = frame_data
    print(f"Processing frame {frame_count}")

    timestamp = frame_count / 60

    distance = m * timestamp * 1000 * px_per_mm
    # # Assuming the video moves away at a constant rate, calculate border width
    # # Adjust the scale factor according to the rate of moving away
    frame = frame[top_border:bottom_border, left_border:right_border, :] # [top:bottom, left:right] 175

    # Split the frame into RGB channels
    blue_channel, green_channel, red_channel = cv2.split(frame)

    # Apply noise reduction or other processing to the green channel
    green_channel = cv2.GaussianBlur(green_channel, (0, 0), sigmaX=1)

    # Merge the processed channels back into an RGB frame
    processed_frame = cv2.merge([blue_channel, green_channel, red_channel])

    # Convert the processed frame to grayscale
    grayscale_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2GRAY)

    # Enhance contrast selectively using CLAHE
    #clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    #contrast_enhanced = clahe.apply(grayscale_frame)
    contrast_enhanced = grayscale_frame
    # Convert to binary image using a normalized threshold of 0.75
    max_pixel_value = np.max(contrast_enhanced)
    threshold_value = max_pixel_value * pixel_threshold
    _, binary_image = cv2.threshold(contrast_enhanced, threshold_value, 255, cv2.THRESH_BINARY)

    imagelen = len(frame)
    while True:
        if imagelen <= 0:
            break
        last_row = binary_image[-1]
        if np.sum(last_row == 255) / len(last_row) > 0.10:
            break
        binary_image = np.roll(binary_image, 1, axis=0)
        binary_image[0] = 0
        imagelen -= 1
    binary_image = binary_image[::-1, :]
    binary_image = binary_image[20:, :]
    # Find bright points
    ys, xs = np.where(binary_image == 255)
    points = [(x, y, -distance) for x, y in zip(xs, ys)]
    points = removespider(points)
    return points

def create_and_visualize_point_cloud(video_path: str, dst_dir: Optional[str], depth_scale) -> None:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    all_points = []
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_count = 0
    ignore_first_frames = cut_front_frames
    ignore_last_frames = cut_back_frames

    with ProcessPoolExecutor() as executor:
        futures = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break  # No more frames to process
            if frame_count < ignore_first_frames or frame_count >= total_frames - ignore_last_frames:
                frame_count += 1
                continue  

            # Submit each frame to be processed as soon as it's read
            future = executor.submit(process_frame_grey, (frame, frame_count, depth_scale))
            futures.append(future)
            frame_count += 1
        
        # Collect results as they become available
        for future in futures:
            points = future.result()  # Blocks until the future is done
            all_points.extend(points)

    cap.release()

    if all_points:
        points_np = np.array(all_points)
        if points_np.ndim == 2 and points_np.shape[1] == 3:
            print(f"Number of points: {points_np.shape[0]}")
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points_np)
            pcd = normalize_pcd(pcd)
            video_name = Path(video_path).stem
            if dst_dir is None: # save to the same directory as video
                file_name = str(Path(video_path).parent / f"{video_name}.pcd")
            else:
                dst_dir = Path(dst_dir)
                dst_dir.mkdir(exist_ok=True)
                file_name = str(dst_dir / f"{video_name}.pcd")
            o3d.io.write_point_cloud(file_name, pcd)
            print(f"Saved point cloud to {dst_dir}.")
            axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1000.0, origin=[0, 0, -1200])
            o3d.visualization.draw_geometries([axes, pcd])
        else:
            print("Error: Points array is not in the expected N by 3 shape.")
    else:
        print("No points were added to the point cloud. Check the frame processing logic.")

def voxelize(pcd, voxel_size=1):
    points = np.asarray(pcd.points)
    min_bound = points.min(axis=0)
    max_bound = points.max(axis=0)

    grid_shape = ((max_bound - min_bound) / voxel_size).astype(int) + 1
    voxel_grid = np.zeros(grid_shape, dtype=int)
    voxel_indices = ((points - min_bound) / voxel_size).astype(int)
    voxel_grid[voxel_indices[:, 0], voxel_indices[:, 1], voxel_indices[:, 2]] = 1
    return voxel_grid

def voxel_to_pcd(voxel_grid: np.ndarray):
    """Convert 3D voxel grid to point cloud object. ASSUMES VOXEL SIZE OF 1.

    Args:
        voxel_grid (np.ndarray): 3D voxel grid.

    Returns:
        o3d.geometry.PointCloud: pcd object.
    """
    voxel_indices = np.argwhere(voxel_grid == 1)
    points = voxel_indices.astype(float)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    return pcd

def normalize_pcd(pcd):
    points = np.asarray(pcd.points)
    min_bound = points.min(axis=0)
    points_normalized = points - min_bound
    pcd_normalized = o3d.geometry.PointCloud()
    pcd_normalized.points = o3d.utility.Vector3dVector(points_normalized)
    return pcd_normalized


if __name__ == '__main__':

    # create_and_visualize_point_cloud(video_path=os.path.expanduser("video_processing/spider_videos/@006r 255 2024-09-12 22-59-36.mp4"),
    #                                  dst_dir=os.path.expanduser("video_processing/point_clouds"), depth_scale=0.2)
    pcd = o3d.io.read_point_cloud("video_processing/point_clouds/@006r 255 2024-09-12 22-59-36.pcd")
    voxel = voxelize(pcd)
    skeleton = skeletonize(voxel)
    thin_pcd = voxel_to_pcd(skeleton)
    o3d.visualization.draw_geometries([pcd])
    o3d.visualization.draw_geometries([thin_pcd])
    print(len(pcd.points), len(thin_pcd.points))
    o3d.io.write_point_cloud("video_processing/point_clouds/thin_test.pcd", thin_pcd)
    

    