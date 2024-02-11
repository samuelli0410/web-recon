import argparse

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import open3d as o3d
from tqdm import tqdm


def generate_graph(file_path: str, distance_threshold: float):
    # Load your point cloud (replace 'your_point_cloud.pcd' with the actual file path)
    # point_cloud = o3d.io.read_point_cloud('test_web.pcd')
    point_cloud = o3d.io.read_point_cloud(file_path)

    # Create a graph to represent the connectivity
    graph = nx.Graph()

    # Get the XYZ coordinates of the points in the point cloud
    points_xyz = np.asarray(point_cloud.points)

    # Iterate through the points to connect close points
    num_points = points_xyz.shape[0]
    print(f"Number of points: {num_points}")

    # Sort points lexicographically
    # https://stackoverflow.com/questions/38277143/sort-2d-numpy-array-lexicographically
    points_xyz = points_xyz[np.lexsort(points_xyz.T[::-1])]
    for i in tqdm(range(num_points)):
        for j in tqdm(range(i + 1, num_points), leave=False):
            point1 = points_xyz[i]
            point2 = points_xyz[j]
            # Calculate the Euclidean distance between point1 and point2
            distance = np.linalg.norm(point1 - point2)
            if distance <= distance_threshold:
                # Add an edge to connect the close points in the graph
                graph.add_edge(tuple(point1), tuple(point2))
            elif point2[0] - point1[0] > distance_threshold: # Break inner loop, since it is lexicographically sorted
                break

    # Visualize the graph
    pos = nx.spring_layout(graph)  # You can choose a different layout algorithm
    nx.draw(graph, pos, node_size=10, font_size=6)
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_file", help="Source pointcloud file (.pcd).")
    parser.add_argument("--threshold", help="Distance threshold for connecting close points", type=float, default=0.1)
    args = parser.parse_args()

    generate_graph(file_path=args.src_file, distance_threshold=args.threshold)
