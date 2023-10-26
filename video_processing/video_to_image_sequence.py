import cv2
import os


class VideoToImages:
    """Class which contains functionalities to transform saved MP4 video to sequence of image frames."""
    def __init__(self, video_file: str) -> None:
        self.video_file = video_file
        self.recorder = cv2.VideoCapture(video_file)

    def generate_every_n_frames(self, num_frames: int):
        """Converts this video to a series of images with one frame for every n frames in the video.

        num_frames (int): number of frames such that the video is sampled every num_frames frames.

        Creates new folder with name of video to hold image frames in same directory.
        Returns None.
        """
        try:
            folder_data_path = f"{self.video_file}_every_{num_frames}_frames" # create new folder associated with video name
            if not os.path.exists(folder_data_path): # if already exists, go to next step
                os.makedirs(folder_data_path)
        
        except:
            print("Unable to create folder.")

        frame_counter = 0
        while True:
            for _ in range(num_frames):
                able_to_read, image_frame = self.recorder.read() # skip over num_frames frames
            
            if not able_to_read: # if end of video reached, exit loop
                break;

            new_image_name = f"./{folder_data_path}/{frame_counter}_image" # create new numbered image name in new folder
            cv2.imwrite(new_image_name, image_frame) # write the new image to the specified folder

            frame_counter += 1
    


        




            