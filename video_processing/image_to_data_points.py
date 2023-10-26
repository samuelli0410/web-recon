from video_to_image_sequence import VideoToImages
from web_simulator.web import Point3D
import numpy as np
import cv2
from huggingface_hub import from_pretrained_keras


class PointGenerator:
    """Class which allows for processing of images and generation of 3D points corresponding to sequence of image frames."""

    CONTAINER_LENGTH = 50
    CONTAINER_WIDTH = 50
    CONTAINER_DEPTH = 50

    MIN_BRIGHTNESS = 150

    def __init__(self, image_folder: str) -> None:
        self.image_folder = image_folder
        self.model = from_pretrained_keras("keras-io/deeplabv3p-resnet50")


    def find_points(self, image: cv2.Mat):
        """Locates bright (greater or equal to MIN_BRIGHTNESS) pixels in the image and converts them to a list of Point3D points.
            
        image (cv2.Mat): image object returned from imread

        Returns list of coordinates identified as web based on brightness values in the image.
        """
        brightness_values = np.array(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
        brightest_points = np.where(brightness_values >= PointGenerator.MIN_BRIGHTNESS)

        return list(zip(brightest_points[1], brightest_points[0]))
    
    

