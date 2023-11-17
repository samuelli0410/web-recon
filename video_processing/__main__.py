from video_to_image_sequence import *
from image_to_data_points import *
from fitting_points_to_line import *

import seaborn as sns
import matplotlib.pyplot as plt

if __name__ == "__main__":
    #video_reader = VideoToImages("test.mp4")
    #image_folder = video_reader.generate_every_n_frames(1)

    image_folder = os.path.expanduser("~/Documents/python_projects/spider_web_project/test_image_folder")

    point_generator = PointGenerator(image_folder)
    raw_points = point_generator.process_folder()

    #fitter = CollinearPointsFitting([])

    print(raw_points)
