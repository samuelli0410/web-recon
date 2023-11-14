import plotly.graph_objects as go

import pandas as pd
import csv
import random

# make some data
f = lambda x, y: x**2 + y**2
a = [[f(x, y) + random.random() * (x+y) for x in range(-10, 10)] for y in range(-10, 10)]

# make a csv with the data 
with open("new_file.csv","w+") as my_csv:
    csvWriter = csv.writer(my_csv,delimiter=',')
    csvWriter.writerows(a)

# Read data from a csv
z_data = pd.read_csv('new_file.csv')

fig = go.Figure(data=[go.Surface(z=z_data.values)])

fig.update_layout(title='Mt Bruno Elevation', autosize=False,
                  width=500, height=500,
                  margin=dict(l=65, r=50, b=65, t=90))

fig.show()