import random
# Returns a list holding a new 3D equation
def pluggingInSlope (coeff_X, coeff_Y, exponent_x, exponent_y, d, slope):
    newcoeff_y = coeff_Y * (slope ** exponent_y)
    return [coeff_X, exponent_x, newcoeff_y, exponent_y, d]

# Generates a random slope between the indicated range
def randomSlopeGenerator (start, final):
    # Input: Int
    # Output: Int
    return random.randint(start, final)

# Generation of points based on 3D shape and line
# Returns a list of coordinates
def points(coeff_X, coeff_Y, exponent_x, exponent_y, d, starting_S, final_S, iterations):
    coordinates = []
    for x_value in range(iterations):
        slope = randomSlopeGenerator(starting_S, final_S)
        y_value = slope * x_value
        shape_3d = pluggingInSlope(coeff_X, coeff_Y, exponent_x, exponent_y, d, slope)
        z_value = shape_3d[0] * (x_value**shape_3d[1]) + shape_3d[2] * (x_value**shape_3d[3]) + shape_3d[4]
        coordinates.append((x_value, y_value, z_value))
    return coordinates