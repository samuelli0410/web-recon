from web_simulator.web import Point3D
import numpy as np
import cv2
from huggingface_hub import from_pretrained_keras
import os


class PointGenerator:
    """Class which allows for processing of images and generation of 3D points corresponding to sequence of image frames."""

    CONTAINER_LENGTH = 50
    CONTAINER_HEIGHT = 50
    CONTAINER_DEPTH = 50

    MIN_BRIGHTNESS = 150

    def __init__(self, image_folder: str) -> None:
        self.image_folder = image_folder
        self.model = from_pretrained_keras("keras-io/deeplabv3p-resnet50")


    def find_points(self, image: cv2.Mat):
        """Locates bright (greater or equal to MIN_BRIGHTNESS) pixels in the image and converts them to a list of Point3D points.
            
        image (cv2.Mat): image object returned from imread

        Returns list of 2D coordinates identified as web based on brightness values in the image, values are pixel locations, not distance.
        """
        brightness_values = np.array(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)) # convert image to black and white array of brightness values
        brightest_points = np.where(brightness_values >= PointGenerator.MIN_BRIGHTNESS) # identify coordinates >= to min brightness

        return list(zip(brightest_points[1], brightest_points[0])) # return as list of (x, y) pairs
    

    def process_folder(self):
        """Iterates through the image folder of this object, finding the points in each image and adding a time (z) coordinate to each.
        The z coordinate is defined as time (image_number) divided by predefined container depth. Using a different number of frames 
        will require adjustments to the time variable.

        Returns list of 3D coordinates for the image sequence in the folder, based on box dimensions.
        """
        all_points = [] # new list to store the results of each frame scan and all points in the 3D space

        for time, frame in enumerate(os.listdir(self.image_folder)): # iterate through each image in the folder, new time value for each
            # read the image and find the xy points in the slice, then add the estimated z coordinate from time / depth
            curr = [Point3D(x_coord / PointGenerator.CONTAINER_LENGTH, 
                            y_coord / PointGenerator.CONTAINER_HEIGHT, 
                            time / PointGenerator.CONTAINER_DEPTH) for x_coord, y_coord in self.find_points(cv2.imread(frame))]
            all_points.extend(curr) # add to total points

        return all_points # return as list of (x, y, z) Point3D objects
        
