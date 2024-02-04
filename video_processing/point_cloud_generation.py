import cv2
import numpy as np
import open3d as o3d
from concurrent.futures import ProcessPoolExecutor
import os

def process_frame_gpu(frame_data):
    frame, frame_count, depth_scale = frame_data
    # Ensure OpenCV is configured with CUDA support
    if not cv2.cuda.getCudaEnabledDeviceCount():
        print("CUDA device not found")
        return []

    # Upload frame to GPU
    gpu_frame = cv2.cuda_GpuMat()
    gpu_frame.upload(frame)

    # Split channels on GPU
    gpu_channels = cv2.cuda.split(gpu_frame)

    # Apply Gaussian blur to each channel with std deviation 1 using CUDA
    gpu_blurred_blue = cv2.cuda.GaussianBlur(gpu_channels[0], (0, 0), 1)
    gpu_blurred_red = cv2.cuda.GaussianBlur(gpu_channels[2], (0, 0), 1)

    # Merge the minimum values of the blurred red and blue channels
    # CUDA does not have a direct min function, so download to CPU for this step
    blurred_blue = gpu_blurred_blue.download()
    blurred_red = gpu_blurred_red.download()
    min_rb_blurred = cv2.min(blurred_red, blurred_blue)

    # Increase contrast using histogram equalization on CPU (no direct CUDA support in OpenCV)
    contrast_enhanced = cv2.equalizeHist(min_rb_blurred)

    # Convert to binary image using a normalized threshold of 0.75
    max_pixel_value = np.max(contrast_enhanced)
    threshold_value = max_pixel_value * 0.75
    _, binary_image = cv2.threshold(contrast_enhanced, threshold_value, 255, cv2.THRESH_BINARY)

    # Find bright points
    ys, xs = np.where(binary_image == 255)
    points = [[x, y, frame_count * depth_scale] for x, y in zip(xs, ys)]
    
    return points


def process_frame(frame_data):
    frame, frame_count, depth_scale = frame_data
    print(f"Processing frame {frame_count}")
    blue_channel, _, red_channel = cv2.split(frame)
    
    # Apply Gaussian blur to each channel with std deviation 1
    blurred_blue = cv2.GaussianBlur(blue_channel, (0, 0), sigmaX=1)
    blurred_red = cv2.GaussianBlur(red_channel, (0, 0), sigmaX=1)

    # Merge the minimum values of the blurred red and blue channels
    min_rb_blurred = cv2.min(blurred_red, blurred_blue)

    # Convert to grayscale for contrast enhancement
    grayscale_image = cv2.cvtColor(min_rb_blurred, cv2.COLOR_BGR2GRAY)

    # Enhance contrast selectively in brighter regions using CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    contrast_enhanced = clahe.apply(grayscale_image)

    # Convert to binary image using a normalized threshold of 0.75
    max_pixel_value = np.max(contrast_enhanced)
    threshold_value = max_pixel_value * 0.75
    _, binary_image = cv2.threshold(contrast_enhanced, threshold_value, 255, cv2.THRESH_BINARY)

    cv2.imshow(f"Processed Frame {frame_count}", binary_image)
    cv2.waitKey(0)

    # Find bright points
    ys, xs = np.where(binary_image == 255)
    points = [[x, y, frame_count * depth_scale] for x, y in zip(xs, ys)]
    
    return points

def process_frame_grey(frame_data):
    frame, frame_count, depth_scale = frame_data
    print(f"Processing frame {frame_count}")

    # Define the percentage of the image to keep (e.g., 90%)
    keep_percentage = 90

    # Get the height of the frame
    height, _ = frame.shape[:2]

    # Calculate the number of rows to keep based on the percentage
    rows_to_keep = int(keep_percentage / 100 * height)

    # Crop the frame to keep only the upper portion
    cropped_frame = frame[:rows_to_keep, :]

    # Convert the cropped frame to grayscale
    grayscale_frame = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur with std deviation 1
    blurred_frame = cv2.GaussianBlur(grayscale_frame, (0, 0), sigmaX=1)

    # Enhance contrast selectively using CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    contrast_enhanced = clahe.apply(blurred_frame)

    # Convert to binary image using a normalized threshold of 0.75
    max_pixel_value = np.max(contrast_enhanced)
    threshold_value = max_pixel_value * 0.75
    _, binary_image = cv2.threshold(contrast_enhanced, threshold_value, 255, cv2.THRESH_BINARY)

    # Find bright points
    ys, xs = np.where(binary_image == 255)
    points = [[x, y, frame_count * depth_scale] for x, y in zip(xs, ys)]
    
    return points

def create_and_visualize_point_cloud(video_path, depth_scale):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    all_points = []
    
    with ProcessPoolExecutor() as executor:
        futures = []
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break  # No more frames to process
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
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points_np)
            o3d.io.write_point_cloud("test_web.pcd", pcd)
            o3d.visualization.draw_geometries([pcd])
        else:
            print("Error: Points array is not in the expected Nx3 shape.")
    else:
        print("No points were added to the point cloud. Check the frame processing logic.")

if __name__ == '__main__':
    create_and_visualize_point_cloud(os.path.expanduser("~/Downloads/2024-01-27_15-46-18 (online-video-cutter.com) (1).mp4"), 1)
