import argparse
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
from skimage.morphology import skeletonize
from tqdm import tqdm


def linear_interpolate_int(i: int, n: int, first: int, last: int) -> int:
    r = (i / (n - 1)) * last + (1 - i / (n - 1)) * first
    r = int(np.round(r))
    return r


class VideoToImages:
    """Class which contains functionalities to transform saved MP4 video to sequence of image frames."""
    def __init__(self, video_file: str, dst_dir: Optional[str]) -> None:
        self.video_file = video_file
        self.dst_dir = dst_dir
        self.recorder = cv2.VideoCapture(video_file)

    def crop_and_resize(self, img: np.ndarray, x_min: float, x_max: float, y_min: float, y_max: float) -> np.ndarray:
        # crop and resize an image
        orig_h, orig_w = img.shape[0], img.shape[1]
        crop_img = img[y_min: y_max, x_min: x_max]
        new_img = cv2.resize(crop_img, (orig_w, orig_h))
        return new_img

    def detect_edge(self, img: np.ndarray) -> Tuple[float, float, float, float]:
        """Detect edge. Take horizontal/vertical sum of (green) pixel values and find the positions with largest/smallest values.

        Args:
            img (np.ndarray): image in numpy array. Values represent pixel values, from 0 to 255 (BRG).

        Returns:
            Tuple[float, float, float, float]: x_min, x_max, y_min, y_max
        """
        height, width, _ = img.shape
        N = 10
        adjust_h = int(height * 0.02)
        adjust_v = int(width * 0.02)

        # horizontal lines
        h_thres = 0.0
        h_sum = np.sum(img, axis=1)[:, 2]
        sorted_h_ind = np.argsort(h_sum)
        large_h_ind = sorted_h_ind[-N:]
        sorted_large_h_ind = np.sort(large_h_ind)
        
        if sorted_large_h_ind[0] < 0.5 * height and h_sum[sorted_large_h_ind[0]] > h_thres:
            y_min = sorted_large_h_ind[0] + adjust_h
        else:
            y_min = 0
        if sorted_large_h_ind[-1] > 0.5 * height and h_sum[sorted_large_h_ind[-1]] > h_thres:
            y_max = sorted_large_h_ind[-1] - adjust_h
        else:
            y_max = height - 1

        # vertical lines
        v_thres = 0.0
        v_sum = np.sum(img, axis=0)[:, 2]
        sorted_v_ind = np.argsort(v_sum)
        large_v_ind = sorted_v_ind[-N:]
        sorted_large_v_ind = np.sort(large_v_ind)

        if sorted_large_v_ind[0] < 0.5 * width and v_sum[sorted_large_v_ind[0]] > v_thres:
            x_min = sorted_large_v_ind[0] + adjust_v
        else:
            x_min = 0
        if sorted_large_v_ind[-1] > 0.5 * width and v_sum[sorted_large_v_ind[-1]] > v_thres:
            x_max = sorted_large_v_ind[-1] - adjust_v
        else:
            x_max = width - 1
        
        return x_min, x_max, y_min, y_max

    def sharpen_edges(self, img: np.ndarray) -> np.ndarray:
        """Sharpening edges by applying Canny edge detection and skeletonize.

        Returns:
            np.ndarray: New image with sharpened edges. Black and white.
        """
        img[0] = 0
        img[1] = 0
        avg_val = img.sum() / (img.shape[0] * img.shape[1])
        img = cv2.Canny(img, avg_val * 0.9, 255)
        img = skeletonize(img, method="lee")
        return img

    def generate_every_n_frames(self, num_frames: int) -> str:
        """Converts this video to a series of images with one frame for every n frames in the video.

        num_frames (int): number of frames such that the video is sampled every num_frames frames.

        Creates new folder with name of video to hold image frames in same directory.
        Returns name of folder.
        """
        try:
            if self.dst_dir is not None:
                dst_dir = Path(self.dst_dir)
            else:
                src_dir = Path(self.video_file).parent
                dst_dir = src_dir.joinpath() / f"{Path(self.video_file).stem}_every_{num_frames}_frames"
            dst_dir.mkdir(exist_ok=True)
        except:
            print("Unable to create folder.")

        frame_counter = 0
        while True:
            for _ in range(num_frames):
                able_to_read, image_frame = self.recorder.read() # skip over num_frames frames
            
            if not able_to_read: # if end of video reached, exit loop
                break

            orig_h, orig_w, _ = image_frame.shape
            new_image_name = str(dst_dir / f"{frame_counter}_image.png") # create new numbered image name in new folder
            cv2.imwrite(new_image_name, image_frame) # write the new image to the specified folder

            frame_counter += 1
        
        # go over all the frames again and crop & resize to remove edges
        # assumes the laser getting closer and we need to crop larger and larger
        # TODO: handle both directions
        # check the last image's edge first
        img_first = cv2.imread(str(dst_dir / "0_image.png"))  # use the first image
        img_last = cv2.imread(str(dst_dir / f"{frame_counter - 1}_image.png"))  # use the last iamge
        first_x_min, first_x_max, first_y_min, first_y_max = self.detect_edge(img_first)
        last_x_min, last_x_max, last_y_min, last_y_max = self.detect_edge(img_last)

        print("first", first_x_min, first_x_max, first_y_min, first_y_max)
        print("last", last_x_min, last_x_max, last_y_min, last_y_max)
        
        print(f"original shape: H={orig_h}, W={orig_w}")
        for i in tqdm(range(frame_counter), desc="remove edges"):
            img_path = str(dst_dir / f"{i}_image.png")
            img = cv2.imread(img_path)
            x_min = linear_interpolate_int(i, frame_counter, first_x_min, last_x_min)
            x_max = linear_interpolate_int(i, frame_counter, first_x_max, last_x_max)
            y_min = linear_interpolate_int(i, frame_counter, first_y_min, last_y_min)
            y_max = linear_interpolate_int(i, frame_counter, first_y_max, last_y_max)
            img = self.crop_and_resize(img, x_min, x_max, y_min, y_max)
            img = self.sharpen_edges(img)
            cv2.imwrite(img_path, img)

        # return the directory where images are saved
        return str(dst_dir)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_file", help="Path to the source video file.")
    parser.add_argument("--dst_dir", help="Target directory to save image files.", default=None)
    parser.add_argument("--n_frames", help="Number of frames such that the video is sampled every `n_frames` frames.", type=int, default=5)
    args = parser.parse_args()

    converter = VideoToImages(video_file=args.src_file, dst_dir=args.dst_dir)
    converter.generate_every_n_frames(args.n_frames)
