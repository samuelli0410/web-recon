import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.spatial import KDTree
import time
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from scipy.spatial import Voronoi, Delaunay
from collections import defaultdict
import math
import matplotlib.patches as mp
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




def is_in_circle(point, center, radius):
    return np.linalg.norm(point - center) <= radius

def line_circle_intersection(p, q, center, r):
    # Case 1: If p and q are both in the circle -> clip to [p, q]
    if is_in_circle(p, center, r) and is_in_circle(q, center, r):
        return [(p, q)], True, True

    # Intersection with line y = ax + b
    # Handle vertical line case
    if q[0] == p[0]:
        # Vertical line x = p[0]
        a = 1
        b = -2 * center[0]
        c = center[0]**2 + (p[1] - center[1])**2 - r**2
        delta = b**2 - 4*a*c
        if delta <= 0:
            return [], False, False
        
        y1 = (-b - np.sqrt(delta)) / (2*a)
        y2 = (-b + np.sqrt(delta)) / (2*a)
        pt1 = np.array([p[0], y1])
        pt2 = np.array([p[0], y2])
        
        # Determine which points are within the segment
        y_min, y_max = sorted([p[1], q[1]])
        pt1_in = y_min <= y1 <= y_max
        pt2_in = y_min <= y2 <= y_max
        
        if not pt1_in and not pt2_in:
            return [], False, False
        
        result = []
        if pt1_in and pt2_in:
            result.append((pt1, pt2))
        elif pt1_in:
            result.append((pt1, q) if is_in_circle(q, center, r) else (pt1, pt1))
        else:
            result.append((p, pt2) if is_in_circle(p, center, r) else (pt2, pt2))
            
        return result, is_in_circle(p, center, r), is_in_circle(q, center, r)
    
    # Non-vertical line case
    slope = (q[1] - p[1]) / (q[0] - p[0])
    intersect = q[1] - slope * q[0]
    
    # Quadratic equation coefficients
    a = slope**2 + 1
    b = 2 * (slope * (intersect - center[1]) - center[0])
    c = center[0]**2 + (intersect - center[1])**2 - r**2

    # Case 2: No intersection
    delta = b**2 - 4*a*c
    if delta <= 0:
        return [], False, False

    # Case 3/4: One or both points outside circle
    pt1 = p
    pt2 = q
    x_min, x_max = sorted([p[0], q[0]])
    check = False
    
    if not is_in_circle(p, center, r):
        x = (-b - np.sqrt(delta)) / (2*a)
        pt1 = np.array([x, slope*x + intersect])
        check = not (x_min <= x <= x_max)
    
    if not is_in_circle(q, center, r):
        x = (-b + np.sqrt(delta)) / (2*a)
        pt2 = np.array([x, slope*x + intersect])
        check = check or not (x_min <= x <= x_max)
    
    # Case 5: neither p or q are inside the circle and no valid intersection
    if check:
        return [], False, False
    
    return [(pt1, pt2)], is_in_circle(p, center, r), is_in_circle(q, center, r)

def compute_restricted_voronoi(samples, radius, ax=None):
    # Compute Voronoi diagram
    vor = Voronoi(samples, qhull_options="Q0")
    
    # Build ridge information
    n = len(vor.vertices)
    vor_ridges = {
        min(edges) * n + max(edges): (
            (centers[0], vor.points[centers[0]]),
            (centers[1], vor.points[centers[1]])
        )
        for edges, centers in zip(vor.ridge_vertices, vor.ridge_points)
        if -1 not in edges  # Skip infinite ridges
    }
    
    # Build adjacency and vertices dictionary
    adjacency = defaultdict(list)
    vertices = {}
    for (ip, p), (iq, q) in vor_ridges.values():
        vertices[ip] = p
        vertices[iq] = q
        adjacency[min(ip, iq)].append(max(ip, iq))
    
    # Build triangles for adjacent cells - FIXED: Create copy of items for iteration
    triangles = []
    adjacency_items = list(adjacency.items())  # Create static copy for iteration
    for p, neighbours in adjacency_items:
        auxp = set(adjacency[p])  # Safe to access original dict
        for i, q in enumerate(neighbours):
            auxq = auxp & set(adjacency.get(q, []))
            for r in neighbours[i+1:]:
                if max(q, r) in adjacency.get(min(q, r), []) and len(auxq.intersection(adjacency.get(r, []))) == 0:
                    triangles.append(mp.Polygon(
                        [vertices[p], vertices[q], vertices[r]], closed=True))
    
    # Process each Voronoi region
    restricted_voronoi_cells = []
    alpha_complex_cells = defaultdict(list)
    triangles_dict = defaultdict(list)
    
    for point_idx, region_idx in enumerate(vor.point_region):
        vertices_in_region = vor.regions[region_idx]
        if not vertices_in_region or -1 in vertices_in_region:
            continue
        
        region = [vor.vertices[i] for i in vertices_in_region]
        center = vor.points[point_idx]
        restr_region = []
        
        for i in range(len(region)):
            p = region[i]
            q = region[(i + 1) % len(region)]
            inter, clip_p, clip_q = line_circle_intersection(p, q, center, radius)
            restr_region.extend(inter)
            
            if clip_p:
                triangles_dict[point_idx].append(i)
            if clip_q:
                triangles_dict[point_idx].append((i + 1) % len(region))
            else:
                alpha_complex_cells[0].append(i)
        
        restricted_voronoi_cells.append((center, restr_region))
    
    # Form triangles for alpha complex - FIXED: Create copy of items for iteration
    triangles_dict_items = list(triangles_dict.items())  # Static copy
    for vertex, incident_edges in triangles_dict_items:
        if len(incident_edges) == 3:
            vertices = [vor.vertices[i] for i in incident_edges]
            alpha_complex_cells[1].append(vertices)
    
    return restricted_voronoi_cells, alpha_complex_cells, triangles


def project_3d_to_2d(points_3d, method='orthographic'):
    if method == 'xy':
        return points_3d[:, :2]
    elif method == 'xz':
        return points_3d[:, [0, 2]]
    elif method == 'yz':
        return points_3d[:, 1:]
    elif method == 'pca':
        # Use PCA to find most informative projection
        
        pca = PCA(n_components=2)
        return pca.fit_transform(points_3d)
    else:  # orthographic (default)
        # Simple orthographic projection (drop z-coordinate)
        return points_3d[:, :2]

def compute_3d_alpha_complex_via_projection(points_3d, radius, projection_method='pca'):
    points_2d = project_3d_to_2d(points_3d, method=projection_method)
    return compute_restricted_voronoi(points_2d, radius)

def visualize_projected_alpha_complex(points_3d, results, projection_method='pca'):
    """
    Robust visualization of projected alpha complex
    
    Parameters:
        points_3d: Original 3D points (n, 3)
        results: Tuple from compute_3d_alpha_complex_via_projection
        projection_method: Which projection was used
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    
    restricted_cells, alpha_complex, triangles = results
    points_2d = project_3d_to_2d(points_3d, projection_method)
    
    # Create figure
    fig = plt.figure(figsize=(12, 6))
    
    # Plot 1: 2D Projection View
    ax1 = fig.add_subplot(121)
    ax1.scatter(points_2d[:, 0], points_2d[:, 1], c='b', s=10)
    
    # Plot 2D boundaries
    for center, boundary in restricted_cells:
        if len(boundary) > 0:
            # Convert boundary segments to continuous lines
            boundary_array = np.array(boundary)
            if boundary_array.ndim == 3:  # If we have multiple segments
                for segment in boundary_array:
                    ax1.plot(segment[:, 0], segment[:, 1], 'r-', linewidth=0.5)
            else:
                ax1.plot(boundary_array[:, 0], boundary_array[:, 1], 'r-', linewidth=0.5)
    ax1.set_title(f'2D Projection ({projection_method} view)')
    
    # Plot 2: 3D View
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.scatter(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2], c='b', s=10)
    
    # Plot 3D boundaries
    for center_2d, boundary_2d in restricted_cells:
        if len(boundary_2d) == 0:
            continue
            
        # Find original 3D center point
        center_2d = np.array(center_2d).reshape(1, -1)  # Ensure proper shape
        dists = np.linalg.norm(points_2d - center_2d, axis=1)
        center_3d = points_3d[np.argmin(dists)]
        
        # Process boundary segments
        boundary_3d = []
        if isinstance(boundary_2d, list):
            if len(boundary_2d) > 0 and isinstance(boundary_2d[0], tuple):
                # Handle list of segments
                for segment in boundary_2d:
                    segment_array = np.array(segment)
                    for bp in segment_array:
                        bp_2d = bp.reshape(1, -1)
                        dists = np.linalg.norm(points_2d - bp_2d, axis=1)
                        boundary_3d.append(points_3d[np.argmin(dists)])
            else:
                # Handle single segment
                boundary_array = np.array(boundary_2d)
                for bp in boundary_array:
                    bp_2d = bp.reshape(1, -1)
                    dists = np.linalg.norm(points_2d - bp_2d, axis=1)
                    boundary_3d.append(points_3d[np.argmin(dists)])
        
        if boundary_3d:
            boundary_3d = np.array(boundary_3d)
            ax2.plot(boundary_3d[:, 0], boundary_3d[:, 1], boundary_3d[:, 2], 
                    'r-', linewidth=0.5)
    
    ax2.set_title('3D View with Projected Boundaries')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    cloud = o3d.io.read_point_cloud("C:/Users/samue/Downloads/Research/Spider/WebReconstruction/LargeWebConnectTest/quadrant_14.pcd")
    points, _,_,_ = splitIntoQuadrants(cloud.points, 0)
    points = laplacian_contraction(points, k=20, sL=2)
    radius = 0.5
    results = compute_3d_alpha_complex_via_projection(points, radius=5.0, 
                                                     projection_method='pca')
    visualize_projected_alpha_complex(points, results)
