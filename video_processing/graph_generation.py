import argparse

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import open3d as o3d
from tqdm import tqdm
import pyvista as pv
import pickle
from scipy.spatial import distance


def generate_graph(file_path: str, distance_threshold: float, out_dir: str):
    # Load your point cloud
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
    for i in tqdm(range(num_points), desc="Creating graph..."):
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
    graph = connect_disjoint_subsets(graph, threshold=15)
    print(f"Number of edges: {len(graph.edges)}")
    with open(out_dir, mode="wb") as f:
        pickle.dump(graph, f)


def generate_graph_2(file_path: str):
    pcd = o3d.io.read_point_cloud(file_path)

    # Assuming pcd is your Open3D point cloud object loaded from earlier
    pcd.estimate_normals()

    # Poisson surface reconstruction
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9)

    # Optionally remove low density vertices
    vertices_to_remove = densities < np.quantile(densities, 0.01)
    mesh.remove_vertices_by_mask(vertices_to_remove)

    # Visualize the mesh
    o3d.visualization.draw_geometries([mesh])

def visualize_graph(graph):
    plotter = pv.Plotter()
    plotter.add_points(np.array(list(graph.nodes)), color='blue', point_size=3)

    center = np.array([200, 200, 200])
    radius = 150
    count = 0
    def is_inside_sphere(point, center, radius):
        return np.linalg.norm(point - center) <= radius
    
    for edge in tqdm(graph.edges, desc="Drawing edges..."):
        if is_inside_sphere(edge[0], center, radius) and is_inside_sphere(edge[1], center, radius):
            count += 1
            line = pv.Line(edge[0], edge[1])
            plotter.add_mesh(line, color='red', line_width=3)
    print(count)
    plotter.show()


def find_closest_points_between_subsets(subset1, subset2):
    min_dist = float('inf')
    closest_pair = (None, None)
    for point1 in subset1:
        for point2 in subset2:
            dist = distance.euclidean(point1, point2)
            if dist < min_dist:
                min_dist = dist
                closest_pair = (point1, point2)
    
    return closest_pair, min_dist

def connect_disjoint_subsets(graph, threshold):
    connected_components = list(nx.connected_components(graph))

    components = [list(comp) for comp in connected_components]

    for i in tqdm(range(len(components)), desc="Connecting disjoint subsets..."):
        for j in range(i + 1, len(components)):
            subset1 = components[i]
            subset2 = components[j]
    
            (point1, point2), dist = find_closest_points_between_subsets(subset1, subset2)
            
            if dist <= threshold:
                graph.add_edge(point1, point2)
                print(f"Added edge between {point1} and {point2} (distance: {dist})")
    return graph

if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--src_file", help="Source pointcloud file (.pcd).")
    # parser.add_argument("--threshold", help="Distance threshold for connecting close points", type=float, default=0.1)
    # args = parser.parse_args()

    # generate_graph(file_path=args.src_file, distance_threshold=args.threshold)
    #generate_graph(file_path="video_processing/point_clouds/thin_test.pcd", distance_threshold=2, out_dir="video_processing/graphs/thin_test.pkl")
    with open("video_processing/graphs/thin_test.pkl", "rb") as f:
        graph = pickle.load(f)
    # visualize_graph(graph)
    degrees = [deg for _, deg in graph.degree()]

    # Plot degree distribution as a histogram
    plt.hist(degrees, bins=range(min(degrees), max(degrees) + 1), edgecolor='black', alpha=0.7)
    plt.title('Degree Distribution')
    plt.xlabel('Degree')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()



