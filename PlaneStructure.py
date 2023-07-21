# This file data structure that tracks the spiders movement
import matplotlib.pyplot as plt
import pandas as pd

# Class for points
class Point:
    def __init__(self, x, y, z):
        #Points are defined in terms of immutable x,y,z tuple
        self.position = (x, y,z)


    def __str__(self):
        return self.position

    def getX(self):
        return self.position[0]

    def getY(self):
        return self.position[1]
    
    def getY(self):
        return self.position[2]


# DataStructure that Stores Information About Points and time they were entered
class SpaceAndTime:

    def __init__(self):
        # Represent full location set as a list of positions
        # Start time = 0 at index 0, each next point is a second
        self.spiderMovement = []

    def nextSecond(self, x, y, z):
        #Adds a point given it's x,yz and coordinates
        self.spiderMovement.append(Position(x, y, z))

    def get(self, second):
        if second > len(self.spiderMovement):
            return "Out of bounds"
        else:
            return self.spiderMovement[second]

    def __str__(self):
        #Returning this as a string returns it as a list!

        return self.spiderMovement

    def entries(self):
        return len(self.spiderMovement)


    def getLists(self):
        x = [self.get(i).getX() for i in range(self.entries())]
        y = [self.get(i).getY() for i in range(self.entries())]
        z = [self.get(i).getZ() for i in range(self.entries())]

        t = [i for i in range(self.entries())]
        return x, y, z, t
    

    #TODO: Jason -- graphing

    def draw2D(self):
        t, x, y = self.getLists()
        plotted = plt.plot(x,y)
        plt.show()



    def lists_to_excel(self, file_name):
        #MadeByChatGPT

        t, x, y = self.getLists()

        # create a dataframe from the three lists
        df = pd.DataFrame({'Time': t, 'X': x, 'Y': y})

        # create a writer object for Excel
        writer = pd.ExcelWriter(file_name, engine='xlsxwriter')

        # write the dataframe to Excel
        df.to_excel(writer, index=False, sheet_name='Sheet1')

        # save the Excel file
        writer.save()


