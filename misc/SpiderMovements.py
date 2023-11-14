import PlaneStructure
import numpy as np


spiderMovements = PlaneStructure.SpaceAndTime()


#Drawing the Web

#Start @ 0,0

spiderMovements.nextSecond(0,0)

t = 0
while t <= 35:
    spiderMovements.nextSecond(t*np.cos(t), t*np.sin(t))
    t += 0.25


spiderMovements.draw2D()

