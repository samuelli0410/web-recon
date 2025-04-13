import gudhi
import networkx as nx
import numpy as np
import open3d as o3d
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.spatial import KDTree
import matplotlib.pyplot as plt


def splitIntoQuadrants(points,eps):
    # Rotate points (optional, adjust as needed)
    rotationmatrix = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
    points = np.dot(points, rotationmatrix.T)

    # Center the points around the mean
    mean = np.mean(points, axis=0)
    points -= mean

    # Initialize quadrant lists
    quad1, quad2, quad3, quad4 = [], [], [], []

    # Assign points to quadrants
    for x, y, z in points:
        if x >= -eps and y >= -eps:
            quad1.append([x, y, z])
        if x >= -eps and y <= eps:
            quad4.append([x, y, z])
        if x <= eps and y >= -eps:
            quad2.append([x, y, z])
        if x <= eps  and y <= -eps:
            quad3.append([x, y, z])

    # Restore original position by adding the mean, only if quadrant is not empty
    quad1 = np.array(quad1) + mean if quad1 else np.array([])
    quad2 = np.array(quad2) + mean if quad2 else np.array([])
    quad3 = np.array(quad3) + mean if quad3 else np.array([])
    quad4 = np.array(quad4) + mean if quad4 else np.array([])

    rotationmatrix_inv = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]])
    if quad1 is not None:
        quad1 = np.dot(quad1, rotationmatrix_inv.T)
    if quad2 is not None:
        quad2 = np.dot(quad2, rotationmatrix_inv.T)
    if quad3 is not None:
        quad3 = np.dot(quad3, rotationmatrix_inv.T)
    if quad4 is not None:
        quad4 = np.dot(quad4, rotationmatrix_inv.T)

    return quad1, quad2, quad3, quad4


def compute_laplacian(points, k=10):
    """Computes the Laplacian matrix using cotangent weights for a point cloud."""
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
    return L.tocsr()
def laplacian_contraction(points, iterations=5, k=10, sL=3.0):
    """Performs Laplacian-based contraction on a point cloud."""
    points = points.copy()
    
    for _ in range(iterations):
        L = compute_laplacian(points, k)
        W_L = sp.diags([sL] * len(points))  # Contraction weight matrix
        W_H = sp.diags([1.0] * len(points))  # Attraction weight matrix
        
        # Solve the system: (W_L * L) * P' = W_H * P
        A = (W_L @ L + W_H).tocsc()  # Convert to CSC format for efficiency
        B = W_H @ points

        # Solve for each coordinate separately using Conjugate Gradient
        points = np.column_stack([spla.cg(A, B[:, i])[0] for i in range(3)])
    
    return points


cloud = o3d.io.read_point_cloud("C:/Users/samue/Downloads/Research/Spider/WebReconstruction/LargeWebConnectTest/quadrant_14.pcd")
points = np.asarray(cloud.points)

# Compute Alpha Complex
alpha_complex = gudhi.AlphaComplex(points=points)
simplex_tree = alpha_complex.create_simplex_tree()

# Compute persistence and plot (optional)
persistence = simplex_tree.persistence()
gudhi.plot_persistence_diagram(persistence)
plt.show()

# Filter edges by persistence
persistence_threshold = 0.1  # Adjust based on your data
G = nx.Graph()

# Iterate over all edges in the complex
for simplex, filtration_value in simplex_tree.get_skeleton(1):
    if len(simplex) == 2:  # Only edges (not vertices)
        # Check persistence: find when this edge dies
        for interval in simplex_tree.persistence_intervals_in_dimension(1):
            if filtration_value >= interval[0] and filtration_value < interval[1]:
                persistence = interval[1] - interval[0]
                if persistence > persistence_threshold:
                    G.add_edge(simplex[0], simplex[1])

# Visualize the graph
pos = {i: points[i] for i in range(len(points))}
nx.draw(G, pos, node_size=5, with_labels=False)
plt.title("Reconstructed Web (Persistence Filtered)")
plt.show()