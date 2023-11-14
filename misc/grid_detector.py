import cv2
import numpy as np

# Load the image
image = cv2.imread('test.png')

# Height and width are pixels, grid_dimension is user set. Sample image is 10x10.
# We will probably want to up the resolution. 
height, width, _ = image.shape
rows, cols = 10, 10

# Cell size represents the width and height of a single cell
cell_size = (width / rows, height / cols)

# Definitely needs work, but we really don't need a super complicated one for our project.
# Most likely a binary yes/no to plot the point will be good enough.
# This specific one is very contrived for this specific image.
def color_detector(r,g,b):
    if abs(r - g) < 10 and b < 10:
        return 'yellow'
    elif max(r,g,b) == r:
        return 'red'
    elif max(r,g,b) == b:
        return 'blue'


averages = [[None for _ in range(cols)] for _ in range(rows)]


for i in range(rows):
    for j in range(cols):
        y = i*cell_size[1]
        x = j*cell_size[0]
        cell = image[int(y):int(y + cell_size[1]), int(x):int(x + cell_size[0])]
        b,g,r = np.mean(cell, axis=(0, 1))
        averages[i][j] = color_detector(r,g,b)

for row in averages:
    print(row)
