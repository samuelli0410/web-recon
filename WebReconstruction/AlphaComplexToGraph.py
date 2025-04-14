import gudhi
import networkx as nx
import numpy as np
import open3d as o3d
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.spatial import KDTree
import matplotlib.pyplot as plt

from scipy.spatial.distance import euclidean

def splitIntoQuadrants(points, eps):
    print("Starting quadrant splitting...")
    # Rotate points (optional, adjust as needed)
    rotationmatrix = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
    points = np.dot(points, rotationmatrix.T)

    # Center the points around the mean
    mean = np.mean(points, axis=0)
    points -= mean

    # Initialize quadrant lists
    quad1, quad2, quad3, quad4 = [], [], [], []

    # Assign points to quadrants
    print("Assigning points to quadrants...")
    for x, y, z in points:
        if x >= -eps and y >= -eps:
            quad1.append([x, y, z])
        if x >= -eps and y <= eps:
            quad4.append([x, y, z])
        if x <= eps and y >= -eps:
            quad2.append([x, y, z])
        if x <= eps and y <= -eps:
            quad3.append([x, y, z])

    # Restore original position by adding the mean
    print("Restoring original positions...")
    quad1 = np.array(quad1) + mean if quad1 else np.array([])
    quad2 = np.array(quad2) + mean if quad2 else np.array([])
    quad3 = np.array(quad3) + mean if quad3 else np.array([])
    quad4 = np.array(quad4) + mean if quad4 else np.array([])

    rotationmatrix_inv = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]])
    if len(quad1) > 0:
        quad1 = np.dot(quad1, rotationmatrix_inv.T)
    if len(quad2) > 0:
        quad2 = np.dot(quad2, rotationmatrix_inv.T)
    if len(quad3) > 0:
        quad3 = np.dot(quad3, rotationmatrix_inv.T)
    if len(quad4) > 0:
        quad4 = np.dot(quad4, rotationmatrix_inv.T)

    print(f"Quadrant sizes: {len(quad1)}, {len(quad2)}, {len(quad3)}, {len(quad4)}")
    return quad1, quad2, quad3, quad4

def compute_laplacian(points, k=10):
    print("Computing Laplacian matrix...")
    n = len(points)
    tree = KDTree(points)
    neighbors = tree.query(points, k=k + 1)[1][:, 1:]
    
    row, col, data = [], [], []
    for i in range(n):
        nbrs = points[neighbors[i]]
        center = points[i]
        
        # Compute weights based on distances
        weights = np.exp(-np.linalg.norm(nbrs - center, axis=1))
        weights /= np.sum(weights)
        
        for j, neighbor in enumerate(neighbors[i]):
            row.append(i)
            col.append(neighbor)
            data.append(-weights[j])
        row.append(i)
        col.append(i)
        data.append(1.0)
    
    L = sp.coo_matrix((data, (row, col)), shape=(n, n))
    print("Laplacian matrix computed")
    return L.tocsr()

def laplacian_contraction(points, iterations=5, k=10, sL=3.0):
    print(f"Starting Laplacian contraction with {iterations} iterations...")
    points = points.copy()
    
    for iter in range(iterations):
        print(f"Iteration {iter + 1}/{iterations}")
        L = compute_laplacian(points, k)
        W_L = sp.diags([sL] * len(points))  # Contraction weight matrix
        W_H = sp.diags([1.0] * len(points))  # Attraction weight matrix
        
        # Solve the system: (W_L * L) * P' = W_H * P
        A = (W_L @ L + W_H).tocsc()  # Convert to CSC format for efficiency
        B = W_H @ points

        # Solve for each coordinate separately using Conjugate Gradient
        points = np.column_stack([spla.cg(A, B[:, i])[0] for i in range(3)])
    
    print("Laplacian contraction completed")
    return points


def distance(p1, p2):
    """Calculate Euclidean distance between two points"""
    return np.linalg.norm(p1 - p2)

def plot_3d_graph(graph, points):
    """Visualize a 3D graph using matplotlib"""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot edges
    for u, v in graph.edges():
        x = [points[u][0], points[v][0]]
        y = [points[u][1], points[v][1]]
        z = [points[u][2], points[v][2]]
        ax.plot(x, y, z, 'b-', linewidth=0.5, alpha=0.8)
    
    # Plot nodes
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='r', s=10)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.title("3D Web Reconstruction")
    plt.tight_layout()
    plt.show()

# Main execution
print("Starting web reconstruction...")
print("Loading point cloud...")
cloud = o3d.io.read_point_cloud("C:/Users/samue/Downloads/Research/Spider/WebReconstruction/LargeWebConnectTest/quadrant_14.pcd")
points = np.asarray(cloud.points)
print(f"Loaded {len(points)} points")

# Compute Alpha Complex
print("Building alpha complex...")
alpha_complex = gudhi.AlphaComplex(points=points)
simplex_tree = alpha_complex.create_simplex_tree()
G = nx.Graph()
for simplex, _ in simplex_tree.get_skeleton(1):
    if len(simplex) == 2:  # It's an edge
        G.add_edge(simplex[0], simplex[1])

# Filter edges (e.g., by length)
max_strand_length = 1.0  # Adjust based on web scale
for u, v in list(G.edges()):
    if distance(points[u], points[v]) > max_strand_length:
        G.remove_edge(u, v)

# Further processing (MST, planarization, etc.)
web_graph = nx.minimum_spanning_tree(G)

# Visualize
plot_3d_graph(web_graph, points)