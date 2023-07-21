import random
import PlaneStructure
import math


# Generates a random slope between the indicated range
def randomNum (start, final):
    # Input: Int
    # Output: Int
    return random.randint(start, final)



def intersection(x, y, z, currentSpace, lfunc):
    for t in range(-10, 10):
        startX = x*t
        startY = y*t
        currentSpace.nextSecond(startX, startY, lfunc(startX,startY))



def pointGenerator(lfunc, slope_iterations, x_iterations, starting_S, final_S):
#lfunc: lambda function that takes in x, y, z and generates point based off of them
#iterations: number of lines and x values  we want to find the intersect ot

    spacetime = PlaneStructure.SpaceAndTime(lfunc)

    for slope_number in range(slope_iterations):
        #iterate over the number of lines that we want to create
        startingX = randomNum(starting_S, final_S)
        startingY = randomNum(starting_S, final_S)
        startingZ = randomNum(starting_S, final_S)
        intersection(startingX, startingY, startingZ, spacetime, lfunc)


    spacetime.plot_points_3d()
norm = lambda x, y: math.sqrt(x ** 2 + y ** 2)


pointGenerator(norm, 10, 10, 0, 10)




