import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.spatial import KDTree
import time
from scipy.spatial import Voronoi, voronoi_plot_2d
from collections import defaultdict
import math
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


start = time.time()



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



def voronoi_diagram_3d(samples, ax):
    vor = Voronoi(samples[:, :2], qhull_options="QJ") 
    n = len(vor.vertices)

    vor_ridges = {
        min(edges) * n + max(edges): (
            (centers[0], vor.points[centers[0]]),
            (centers[1], vor.points[centers[1]])
        )
        for edges, centers in zip(vor.ridge_vertices, vor.ridge_points)
    }

    adjacency = defaultdict(list)
    vertices = {}
    for (ip, p), (iq, q) in vor_ridges.values():
        vertices[ip] = p
        vertices[iq] = q
        adjacency[min(ip, iq)].append(max(ip, iq))

    triangles = []
    adjacency_items = list(adjacency.items())  # Convert to list to avoid dictionary change during iteration
    for p, neighbours in adjacency_items:
        auxp = set(adjacency[p])
        for i, q in enumerate(neighbours):
            auxq = auxp & set(adjacency[q])
            for r in neighbours[i+1:]:
                if max(q, r) in adjacency[min(q, r)] and len(auxq.intersection(adjacency[r])) == 0:
                    try:
                        a = vertices[p]
                        b = vertices[q]
                        c = vertices[r]

                        # Get z-values from nearest neighbors in original point cloud
                        def get_z(xy):
                            dists = np.linalg.norm(samples[:, :2] - xy[:2], axis=1)
                            return samples[np.argmin(dists)][2]

                        a3 = [*a, get_z(a)]
                        b3 = [*b, get_z(b)]
                        c3 = [*c, get_z(c)]

                        triangles.append([a3, b3, c3])
                    except KeyError:
                        continue
    # Create 3D triangle collection
    tri_collection = Poly3DCollection(triangles, alpha=0.5, facecolor='orange', edgecolor='black')
    ax.add_collection3d(tri_collection)

    # Scatter plot of points
    ax.scatter(samples[:, 0], samples[:, 1], samples[:, 2], s=1, c='k')
    ax.set_title("3D Voronoi Triangle Approximation")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.auto_scale_xyz(samples[:, 0], samples[:, 1], samples[:, 2])


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

def is_in_circle(point, center, r):
    
    return (point[0] - center[0])**2 + (point[1] - center[1])**2 <= r**2

def find_intersection(p, q, center, r):
    # Case 1: If p and q are both in the circle, clip to [p, q]
    if is_in_circle(p, center, r) and is_in_circle(q, center, r):
        return [(p, q)], True, True

    # Case 2: Compute intersection with the circle
    slope = (q[1] - p[1]) / (q[0] - p[0]) if q[0] != p[0] else float('inf')  # Handle vertical lines
    intercept = p[1] - slope * p[0] if slope != float('inf') else p[0]  # Use x-intercept for vertical lines

    a = slope**2 + 1
    b = 2 * (slope * (intercept - center[1]) - center[0])
    c = center[0]**2 + (intercept - center[1])**2 - r**2
    delta = b**2 - 4*a*c

    # Case 2a: No intersection (discriminant <= 0)
    if delta <= 0:
        return [], False, False

    sqrt_delta = np.sqrt(delta)
    x1 = (-b + sqrt_delta) / (2 * a)
    x2 = (-b - sqrt_delta) / (2 * a)

    # Case 3: p is not in the circle
    pt1 = np.array([x1, slope * x1 + intercept])
    pt2 = np.array([x2, slope * x2 + intercept])

    is_in_pq = lambda z: (z >= p[0]) and (z <= q[0])
    check = False 

    if not is_in_circle(p, center, r):
        x = (-b - np.sqrt(delta)) / (2 * a)
        pt1 = np.array([x, slope * x + intercept])
        check = not is_in_pq(x)

    # Case 4: q is not in the circle
    if not is_in_circle(q, center, r):
        x = (-b + np.sqrt(delta)) / (2 * a)
        pt2 = np.array([x, slope * x + intercept])
        check = (check or (not is_in_pq(x)))

    # Case 5: Neither p nor q are inside the circle
    if check:
        return ([], False, False)
    
    return [(pt1, pt2)], is_in_circle(p, center, r), is_in_circle(q, center, r)






#1/64th of a web
cloud = o3d.io.read_point_cloud("C:/Users/samue/Downloads/Research/Spider/WebReconstruction/LargeWebConnectTest/quadrant_14.pcd")

#split into quadrants for smaller sample
points, _,_,_ = splitIntoQuadrants(cloud.points, 0)
# print(len(points))

points = laplacian_contraction(points, k=20, sL=2)


fig = plt.figure(figsize=(10, 10))

ax = fig.add_subplot(111, projection='3d')
voronoi_diagram_3d(points,ax)
plt.show()


# pcd = o3d.geometry.PointCloud()
# pcd.points = o3d.utility.Vector3dVector(points)
# o3d.visualization.draw_geometries([pcd], window_name="3D Point Cloud")