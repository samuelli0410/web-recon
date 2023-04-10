#This file data structure that tracks the spiders movement


#Location Class to Find Position
class Postion:
    def __init__(self, x, y):
        self.position = (x,y)

    def __str__(self):
        return self.position

    def getX(self):
        return self.position[0]
    def getY(self):
        return self.position[1]

#DataStructure that Gives Time Information

class SpaceAndTime:

    def __init__(self):
        #Represent full location set as a list of positions
        #Start time = 0 at index 0, each next point is a second
        spiderMovement = []

    def nextSecond(x, y):
        spiderMovement.append(Position(x,y))

    def get(second):
        if second>spiderMovement.length:
            return "Out of bounds"
        else:
            return spiderMovement[second]

    def __str__(self):
        return spiderMovement

    def entries(self):
        return spiderMovement.length
