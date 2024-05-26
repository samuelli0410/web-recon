import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import List, Optional, Tuple
import os

import cv2
import numpy as np
import open3d as o3d


def process_frame_grey(frame_data):
    frame, frame_count, depth_scale = frame_data
    print(f"Processing frame {frame_count}")

    # # Assuming the video moves away at a constant rate, calculate border width
    # # Adjust the scale factor according to the rate of moving away

    frame = frame[300:850, 550:1350]

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
    threshold_value = max_pixel_value * 0.65
    _, binary_image = cv2.threshold(contrast_enhanced, threshold_value, 255, cv2.THRESH_BINARY)

    # Find bright points
    ys, xs = np.where(binary_image == 255)
    points = [(x, y, frame_count * depth_scale) for x, y in zip(xs, ys)]

    return points

def create_and_visualize_point_cloud(video_path: str, dst_dir: Optional[str], depth_scale) -> None:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    all_points = []
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_count = 0
    ignore_first_frames = 400
    ignore_last_frames = 150

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
        # Convert all_points to a NumPy array and visualize
        points_np = np.array(all_points)
        if points_np.ndim == 2 and points_np.shape[1] == 3:
            print(f"Number of points: {points_np.shape[0]}")
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points_np)
            # o3d.io.write_point_cloud(f"{file_name}.pcd", pcd)
            video_name = Path(video_path).stem
            if dst_dir is None: # save to the same directory as video
                file_name = str(Path(video_path).parent / f"{video_name}.pcd")
            else:
                dst_dir = Path(dst_dir)
                dst_dir.mkdir(exist_ok=True)
                file_name = str(dst_dir / f"{video_name}.pcd")
            o3d.io.write_point_cloud(file_name, pcd)
            print(f"Saved point cloud to {dst_dir}.")
            o3d.visualization.draw_geometries([pcd])
        else:
            print("Error: Points array is not in the expected N by 3 shape.")
    else:
        print("No points were added to the point cloud. Check the frame processing logic.")

if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--src_file", help="Path to the source video file.")
    # parser.add_argument("--dst_dir", help="Directory to save PointCloud (.pcd) files", default=None)
    # parser.add_argument("--depth_scale", type=float, default=1.0)
    # args = parser.parse_args()
    # create_and_visualize_point_cloud(video_path=args.src_file, dst_dir=args.dst_dir, depth_scale=args.depth_scale)

    create_and_visualize_point_cloud(video_path=os.path.expanduser("~/Downloads/2024-05-25 18-20-08.mp4"),
                                     dst_dir=os.path.expanduser("~/Documents/spider-recordings"), depth_scale=0.2)
