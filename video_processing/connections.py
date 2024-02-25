import numpy as np
import igraph as ig
from scipy.spatial import KDTree


fps = 60

initial_distance_to_box = 1000
box_depth = 100

adjustment_factor = (initial_distance_to_box + box_depth) / initial_distance_to_box

def real_distance(point1: np.array, point2: np.array, adjustment_factor: float):
    return np.linalg.norm(point1 * adjustment_factor)










#TEST

# Sample 3D points
points = np.random.rand(100, 3)  # 100 random 3D points

# Fixed distance threshold
distance_threshold = 0.2

# Build a KDTree
tree = KDTree(points)

# Query the KDTree for points within the distance threshold
connections = []
for i, point in enumerate(points):
    # Query includes the point itself, so we exclude it by setting k=None and distance_upper_bound
    indices = tree.query_ball_point(point, r=distance_threshold)
    for j in indices:
        if i != j:  # Exclude self-connections
            connections.append((i, j))

# Remove duplicates: Since (i, j) and (j, i) represent the same connection
connections = list(set(tuple(sorted(pair)) for pair in connections))

print(f"Number of connections: {len(connections)}")
# Optionally, print connections
print("Connections (indices):", connections)
print(points)








