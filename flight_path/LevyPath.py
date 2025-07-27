import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import time



def levyStep(position=[0,0,0],alpha = 2, step_scale=1):
    #Pick a random unit vector in 3D, and multiply it by levy distance
    direction = np.random.normal(size=3)
    direction /= np.linalg.norm(direction)


    #find levy distance
    u = np.random.uniform(0, 1)
    step_length = step_scale * (u ** (-1/alpha))

    step = direction*step_length

    return position + step

def simulate_levy_walk(starting_pos=[0,0,0], timesteps=10,alpha = 2, step_scale=1):
    pos = starting_pos
    res = {}
    res[0] = starting_pos
    for i in range(timesteps):
        pos = levyStep(pos, alpha, step_scale)
        res[i+1] = pos

    return res


print(simulate_levy_walk())