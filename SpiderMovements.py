import PlaneStructure
import numpy as np


spiderMovements = PlaneStructure.SpaceAndTime()


#Drawing the Web

#Start @ 0,0

spiderMovements.nextSecond(0,0)


#Draw the first pattern:

for x in range(10):
    spiderMovements.nextSecond(x, np.sin(x*(np.pi/2)))


#Draw the second pattern
for x in range(5):
    spiderMovements.nextSecond(x + 10, 10*np.sin(10*np.pi) +x**2)

#Draw the last pattern

for x in range (11):
    spiderMovements.nextSecond(15-x, np.sin(10*np.pi) +5**2 + x)



spiderMovements.draw2D()

