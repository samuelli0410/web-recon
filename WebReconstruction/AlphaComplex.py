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



# def voronoi_diagram_3d(samples, ax):
#     vor = Voronoi(samples[:, :2], qhull_options="QJ") 
#     n = len(vor.vertices)

#     vor_ridges = {
#         min(edges) * n + max(edges): (
#             (centers[0], vor.points[centers[0]]),
#             (centers[1], vor.points[centers[1]])
#         )
#         for edges, centers in zip(vor.ridge_vertices, vor.ridge_points)
#     }

#     adjacency = defaultdict(list)
#     vertices = {}
#     for (ip, p), (iq, q) in vor_ridges.values():
#         vertices[ip] = p
#         vertices[iq] = q
#         adjacency[min(ip, iq)].append(max(ip, iq))

#     triangles = []
#     adjacency_items = list(adjacency.items())  # Convert to list to avoid dictionary change during iteration
#     for p, neighbours in adjacency_items:
#         auxp = set(adjacency[p])
#         for i, q in enumerate(neighbours):
#             auxq = auxp & set(adjacency[q])
#             for r in neighbours[i+1:]:
#                 if max(q, r) in adjacency[min(q, r)] and len(auxq.intersection(adjacency[r])) == 0:
#                     try:
#                         a = vertices[p]
#                         b = vertices[q]
#                         c = vertices[r]

#                         # Get z-values from nearest neighbors in original point cloud
#                         def get_z(xy):
#                             dists = np.linalg.norm(samples[:, :2] - xy[:2], axis=1)
#                             return samples[np.argmin(dists)][2]

#                         a3 = [*a, get_z(a)]
#                         b3 = [*b, get_z(b)]
#                         c3 = [*c, get_z(c)]

#                         triangles.append([a3, b3, c3])
#                     except KeyError:
#                         continue
#     # Create 3D triangle collection
#     tri_collection = Poly3DCollection(triangles, alpha=0.5, facecolor='orange', edgecolor='black')
#     ax.add_collection3d(tri_collection)

#     # Scatter plot of points
#     ax.scatter(samples[:, 0], samples[:, 1], samples[:, 2], s=1, c='k')
#     ax.set_title("3D Voronoi Triangle Approximation")
#     ax.set_xlabel("X")
#     ax.set_ylabel("Y")
#     ax.set_zlabel("Z")
#     ax.auto_scale_xyz(samples[:, 0], samples[:, 1], samples[:, 2])


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

# def is_in_circle(point, center, r):
    
#     return (point[0] - center[0])**2 + (point[1] - center[1])**2 <= r**2

# def find_intersection(p, q, center, r):
#     # Case 1: If p and q are both in the circle, clip to [p, q]
#     if is_in_circle(p, center, r) and is_in_circle(q, center, r):
#         return [(p, q)], True, True

#     # Case 2: Compute intersection with the circle
#     slope = (q[1] - p[1]) / (q[0] - p[0]) if q[0] != p[0] else float('inf')  # Handle vertical lines
#     intercept = p[1] - slope * p[0] if slope != float('inf') else p[0]  # Use x-intercept for vertical lines

#     a = slope**2 + 1
#     b = 2 * (slope * (intercept - center[1]) - center[0])
#     c = center[0]**2 + (intercept - center[1])**2 - r**2
#     delta = b**2 - 4*a*c

#     # Case 2a: No intersection (discriminant <= 0)
#     if delta <= 0:
#         return [], False, False

#     sqrt_delta = np.sqrt(delta)
#     x1 = (-b + sqrt_delta) / (2 * a)
#     x2 = (-b - sqrt_delta) / (2 * a)

#     # Case 3: p is not in the circle
#     pt1 = np.array([x1, slope * x1 + intercept])
#     pt2 = np.array([x2, slope * x2 + intercept])

#     is_in_pq = lambda z: (z >= p[0]) and (z <= q[0])
#     check = False 

#     if not is_in_circle(p, center, r):
#         x = (-b - np.sqrt(delta)) / (2 * a)
#         pt1 = np.array([x, slope * x + intercept])
#         check = not is_in_pq(x)

#     # Case 4: q is not in the circle
#     if not is_in_circle(q, center, r):
#         x = (-b + np.sqrt(delta)) / (2 * a)
#         pt2 = np.array([x, slope * x + intercept])
#         check = (check or (not is_in_pq(x)))

#     # Case 5: Neither p nor q are inside the circle
#     if check:
#         return ([], False, False)
    
#     return [(pt1, pt2)], is_in_circle(p, center, r), is_in_circle(q, center, r)

def is_in_sphere(point, center, radius):
    return np.linalg.norm(point - center) <= radius
def line_sphere_intersection(p, q, center, r):
    # Vector from p to q
    d = q - p
    # Vector from center to p
    f = p - center
    
    a = np.dot(d, d)
    b = 2 * np.dot(f, d)
    c = np.dot(f, f) - r**2
    
    discriminant = b**2 - 4*a*c
    
    # Case 1: Both points inside sphere
    if is_in_sphere(p, center, r) and is_in_sphere(q, center, r):
        return [(p, q)], True, True
    
    # Case 2: No intersection
    if discriminant <= 0:
        return [], False, False
    
    discriminant = np.sqrt(discriminant)
    t1 = (-b - discriminant) / (2*a)
    t2 = (-b + discriminant) / (2*a)
    
    # Check if intersections are within the segment
    intersections = []
    if 0 <= t1 <= 1:
        intersections.append(p + t1 * d)
    if 0 <= t2 <= 1:
        intersections.append(p + t2 * d)
    
    # Case 3: One intersection (tangent or one point inside)
    if len(intersections) == 1:
        if is_in_sphere(p, center, r):
            return [(p, intersections[0])], True, False
        else:
            return [(intersections[0], q)], False, True
    
    # Case 4: Two intersections
    if len(intersections) == 2:
        return [(intersections[0], intersections[1])], False, False
    
    # Case 5: No valid intersections
    return [], False, False


def plane_sphere_intersection(normal, point, center, radius):
    # Distance from center to plane
    distance = np.abs(np.dot(normal, center - point)) / np.linalg.norm(normal)
    
    if distance > radius:
        return None 
    
    # Circle center is projection of sphere center onto plane
    circle_center = center - distance * normal / np.linalg.norm(normal)
    circle_radius = np.sqrt(radius**2 - distance**2)
    
    return circle_center, circle_radius


def compute_3d_restricted_voronoi(samples, radius, ax=None):
    # Compute 3D Voronoi diagram
    vor = Voronoi(samples, qhull_options="Qz")
    
    # Build ridge information - modified for 3D
    vor_ridges = {}
    for ridge_idx, (p1, p2) in enumerate(vor.ridge_points):
        ridge_verts = vor.ridge_vertices[ridge_idx]
        if -1 in ridge_verts:
            continue  # Skip infinite ridges
        
        # In 3D, a ridge is a polygon (list of vertices)
        ridge_polygon = [vor.vertices[v] for v in ridge_verts]
        key = tuple(sorted((p1, p2)))
        vor_ridges[key] = (vor.points[p1], vor.points[p2], ridge_polygon)
    
    # Build adjacency and vertices dictionary
    adjacency = defaultdict(list)
    vertices = {}
    for (ip, iq), (p, q, ridge) in vor_ridges.items():
        vertices[ip] = p
        vertices[iq] = q
        adjacency[min(ip, iq)].append(max(ip, iq))
    
    # Process each Voronoi cell
    restricted_voronoi_cells = []
    alpha_complex_cells = defaultdict(list)
    
    for point_idx, region_idx in enumerate(vor.point_region):
        vertices_in_region = vor.regions[region_idx]
        if not vertices_in_region or -1 in vertices_in_region:
            continue
        
        region = [vor.vertices[i] for i in vertices_in_region]
        center = vor.points[point_idx]
        
        # Compute convex hull of the region vertices
        from scipy.spatial import ConvexHull
        try:
            hull = ConvexHull(region)
        except:
            continue 
        
        restricted_faces = []
        
        # Process each face of the convex hull
        for simplex in hull.simplices:
            face_vertices = [region[i] for i in simplex]
            normal = np.cross(face_vertices[1] - face_vertices[0], 
                             face_vertices[2] - face_vertices[0])
            if np.linalg.norm(normal) < 1e-10:
                continue  # Skip degenerate faces
            normal = normal / np.linalg.norm(normal)
            
            # Get intersection with sphere (a circle)
            circle = plane_sphere_intersection(normal, face_vertices[0], center, radius)
            if circle is None:
                continue  # No intersection
            
            circle_center, circle_radius = circle
            
            # Project face vertices onto the plane
            projected_vertices = []
            for v in face_vertices:
                vec = v - circle_center
                proj = v - np.dot(vec, normal) * normal
                projected_vertices.append(proj)
            
            # Clip the face to the circle
            clipped_face = []
            for i in range(len(projected_vertices)):
                p = projected_vertices[i]
                q = projected_vertices[(i+1)%len(projected_vertices)]
                inter, _, _ = line_sphere_intersection(p, q, circle_center, circle_radius)
                clipped_face.extend(inter)
            
            if clipped_face:
                restricted_faces.append(clipped_face)
        
        if restricted_faces:
            restricted_voronoi_cells.append((center, restricted_faces))
    
    # Build alpha complex
    alpha_complex_cells[2] = restricted_voronoi_cells
    
    return restricted_voronoi_cells, alpha_complex_cells


if __name__ == "__main__":

    #1/64th of a web
    cloud = o3d.io.read_point_cloud("C:/Users/samue/Downloads/Research/Spider/WebReconstruction/LargeWebConnectTest/quadrant_14.pcd")

    #split into quadrants for smaller sample
    points, _,_,_ = splitIntoQuadrants(cloud.points, 0)
    # print(len(points))

    points = laplacian_contraction(points, k=20, sL=2)
    radius = 0.5
    
    restricted_cells, alpha_complex = compute_3d_restricted_voronoi(points, radius)

    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot original points
    ax.scatter(points[:,0], points[:,1], points[:,2], c='r', marker='o')
    
    # Plot restricted Voronoi cells
    for center, faces in restricted_cells:
        for face in faces:
            if len(face) >= 2:
                # Convert list of line segments to polygon vertices
                vertices = []
                for segment in face:
                    vertices.extend(segment)
                vertices = np.array(vertices)
                
                # Create a polygon for each face
                try:
                    poly = Poly3DCollection([vertices], alpha=0.3)
                    poly.set_facecolor(np.random.rand(3))
                    ax.add_collection3d(poly)
                except:
                    pass
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.title('3D Restricted Voronoi Diagram')
    plt.show()


# pcd = o3d.geometry.PointCloud()
# pcd.points = o3d.utility.Vector3dVector(points)
# o3d.visualization.draw_geometries([pcd], window_name="3D Point Cloud")