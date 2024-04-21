import numpy as np
from scipy.spatial import KDTree
import open3d as o3d
import igraph as ig
import plotly.graph_objects as go
import time



pcd = o3d.io.read_point_cloud("test_web.pcd")

points = np.asarray(pcd.points)



def adaptive_threshold_connect(points: np.ndarray, base_threshold: float, density_factor: float, num_neighbors: int):

    tree = KDTree(points)
    edges = []
    thresholds = np.full(len(points), base_threshold)


    for i, point in enumerate(points):
        # determine thresholds for local regions
        distances, indices = tree.query(point, k=num_neighbors)
        local_threshold = np.mean(distances) * density_factor
        thresholds[i] = local_threshold

        # determine edges
        neighbors = tree.query_ball_point(point, r=thresholds[i])
        for neighbor in neighbors:
            if neighbor != i and [i, neighbor] not in edges and [neighbor, i] not in edges:
                edge = [i, neighbor]
                edges.append(edge)
                print(edge)

    
    graph = ig.Graph(edges=edges)

    return graph



start_time = time.time()
g = adaptive_threshold_connect(points=points, base_threshold=10.0, density_factor=1.0, num_neighbors=10)
end_time = time.time()

print(f"Edge connection took {end_time - start_time} to complete.")
g.summary(verbosity=1)

g.write_graphml("test_graph.graphml")



x = points[:, 0]
y = points[:, 1]
z = points[:, 2]

#trace_vertices = go.Scatter3d(x=x, y=y, z=z, mode='markers', marker=dict(size=5, color='blue'))

edge_list = g.get_edgelist()
x_edges = []
y_edges = []
z_edges = []


for edge in edge_list:
    x_coords = [points[edge[0]][0], points[edge[1]][0], None]  
    y_coords = [points[edge[0]][1], points[edge[1]][1], None]
    z_coords = [points[edge[0]][2], points[edge[1]][2], None]
    x_edges += x_coords
    y_edges += y_coords
    z_edges += z_coords

trace_edges = go.Scatter3d(x=x_edges, y=y_edges, z=z_edges, mode='lines', line=dict(color='black', width=2))

layout = go.Layout(
    title='3D Graph Visualization',
    scene=dict(
        xaxis=dict(title='X'),
        yaxis=dict(title='Y'),
        zaxis=dict(title='Z'),
        aspectmode='cube'
    )
)

# Combine traces and layout
#fig = go.Figure(data=[trace_vertices, trace_edges], layout=layout)
fig = go.Figure(data=trace_edges, layout=layout)

# Show the plot
fig.show()