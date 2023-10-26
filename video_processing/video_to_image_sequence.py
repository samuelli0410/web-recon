import cv2
import os



class VideoToImages:
    """
    Class which contains functionalities to transform saved MP4 video to sequence of image frames.
    """
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
            folder_data_path = f"{self.video_file}_every_{num_frames}_frames"
            if not os.path.exists(folder_data_path):
                os.makedirs(folder_data_path)
        
        except:
            print("Unable to create folder.")

        frame_counter = 0
        while True:
            for _ in range(num_frames):
                able_to_read, image_frame = self.recorder.read()
            
            if not able_to_read:
                break;

            new_image_name = f"./{folder_data_path}/{frame_counter}_image"
            cv2.imwrite(new_image_name, image_frame)

            frame_counter += 1
    

    
        




            