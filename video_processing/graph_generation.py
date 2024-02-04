import networkx as nx
import matplotlib.pyplot as plt
import open3d as o3d
import numpy as np

# Load your point cloud (replace 'your_point_cloud.pcd' with the actual file path)
point_cloud = o3d.io.read_point_cloud('test_web.pcd')

# Define a distance threshold for connecting close points
distance_threshold = 0.1  # Adjust as needed

# Create a graph to represent the connectivity
graph = nx.Graph()

# Get the XYZ coordinates of the points in the point cloud
points_xyz = np.asarray(point_cloud.points)

# Iterate through the points to connect close points
num_points = points_xyz.shape[0]
for i in range(num_points):
    for j in range(i + 1, num_points):
        point1 = points_xyz[i]
        point2 = points_xyz[j]
        # Calculate the Euclidean distance between point1 and point2
        distance = np.linalg.norm(point1 - point2)
        if distance <= distance_threshold:
            # Add an edge to connect the close points in the graph
            graph.add_edge(tuple(point1), tuple(point2))

# Visualize the graph
pos = nx.spring_layout(graph)  # You can choose a different layout algorithm
nx.draw(graph, pos, node_size=10, font_size=6)
plt.show()
