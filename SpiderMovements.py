import PlaneStructure


spiderMovements = PlaneStructure.SpaceAndTime()


for x in range(10):
    spiderMovements.nextSecond(x,10)

spiderMovements.draw2D()

