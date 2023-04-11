# This file data structure that tracks the spiders movement
import matplotlib.pyplot as plt



# Location Class to Find Position
class Position:
    def __init__(self, x, y):
        self.position = (x, y)

    def __str__(self):
        return self.position

    def getX(self):
        return self.position[0]

    def getY(self):
        return self.position[1]


# DataStructure that Gives Time Information

class SpaceAndTime:

    def __init__(self):
        # Represent full location set as a list of positions
        # Start time = 0 at index 0, each next point is a second
        self.spiderMovement = []

    def nextSecond(self, x, y):
        self.spiderMovement.append(Position(x, y))

    def get(self, second):
        if second > len(self.spiderMovement):
            return "Out of bounds"
        else:
            return self.spiderMovement[second]

    def __str__(self):
        return self.spiderMovement

    def entries(self):
        return len(self.spiderMovement)

    def draw2D(self):
        x = [self.get(i).getX() for i in range(self.entries())]
        y = [self.get(i).getY() for i in range(self.entries())]
        plotted = plt.plot(x,y)
        plt.show()



    def toSheets(self):
        return


