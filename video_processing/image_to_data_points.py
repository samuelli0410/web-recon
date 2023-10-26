from video_to_image_sequence import VideoToImages
from web_simulator.web import Point3D

class PointGenerator:
    """
    Class which allows for processing of images and generation of 3D points corresponding to sequence of image frames.
    """

    CONTAINER_LENGTH = 50
    CONTAINER_WIDTH = 50
    CONTAINER_DEPTH = 50

    def __init__(self, image_folder: str) -> None:
        self.image_folder = image_folder

    



