import cv2
import numpy as np
import os

# Global variable to control pausing
paused = False

def process_frame(frame_data):
    global paused
    frame, frame_count, depth_scale = frame_data
    print(f"Processing frame {frame_count}")
    blue_channel, _, red_channel = cv2.split(frame)
    
    # Apply Gaussian blur to each channel with std deviation 1
    blurred_blue = blue_channel
    blurred_red = red_channel

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

    print(max_pixel_value)
    cv2.imshow(f"Processed Frame {frame_count}", binary_image)
    
    while paused:
        # Wait for a key press to resume processing
        key = cv2.waitKey(0)
        if key == ord('q'):
            cv2.destroyAllWindows()  # Close OpenCV windows and exit
            return []
        elif key == ord('c'):
            paused = False  # Resume processing

    # Find bright points
    ys, xs = np.where(binary_image == 255)
    points = [[x, y, frame_count * depth_scale] for x, y in zip(xs, ys)]
    
    return points

# Example usage
if __name__ == '__main__':
    video_path = os.path.expanduser("~/Downloads/2024-01-27_15-46-18 (online-video-cutter.com) (1).mp4") # Replace with your video path
    depth_scale = 1  # Adjust depth scale as needed
    cap = cv2.VideoCapture(video_path)
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        points = process_frame((frame, frame_count, depth_scale))
        
        frame_count += 1

    cap.release()
    cv2.destroyAllWindows()  # Close all OpenCV windows
