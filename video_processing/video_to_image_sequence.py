import argparse
from pathlib import Path
from typing import Optional

import cv2


class VideoToImages:
    """Class which contains functionalities to transform saved MP4 video to sequence of image frames."""
    def __init__(self, video_file: str, dst_dir: Optional[str]) -> None:
        self.video_file = video_file
        self.dst_dir = dst_dir
        self.recorder = cv2.VideoCapture(video_file)

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

            new_image_name = str(dst_dir / f"{frame_counter}_image.png") # create new numbered image name in new folder
            cv2.imwrite(new_image_name, image_frame) # write the new image to the specified folder

            frame_counter += 1
        
        # return the directory where images are saved
        return str(dst_dir)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_file", help="Path to the source video file.")
    parser.add_argument("--dst_dir", help="Target directory to save image files.", default=None)
    parser.add_argument("--n_frames", help="Number of frames such that the video is sampled every num_frames frames.", type=int, default=5)
    args = parser.parse_args()

    converter = VideoToImages(video_file=args.src_file, dst_dir=args.dst_dir)
    converter.generate_every_n_frames(args.n_frames)
