import open3d as o3d
import numpy as np
from sklearn.cluster import DBSCAN
from scipy.spatial import Delaunay
from scipy.spatial import KDTree
import random
from collections import defaultdict


def cluster_points_in_xz(points, radius):
    """
    Cluster points projected onto the XZ plane and use cluster centroids for triangulation.

    Parameters:
        points (numpy.ndarray): Nx3 array of 3D point cloud coordinates.
        radius (float): Clustering radius in the XZ plane.

    Returns:
        numpy.ndarray: Cluster centroids in 3D (preserving y-values).
    """
    points_xz = points[:, [0, 2]]  # Only use x and z
    clustering = DBSCAN(eps=radius, min_samples=1).fit(points_xz)
    labels = clustering.labels_

    unique_labels = np.unique(labels)
    centroids = []
    for label in unique_labels:
        cluster_points = points[labels == label]
        centroid = np.mean(cluster_points, axis=0)  # Preserve original y-values
        centroids.append(centroid)

    return np.array(centroids)


def delaunay_triangulation_with_edge_filter(points, max_edge_length):
    """
    Perform Delaunay triangulation on 2D projected points (xz-plane),
    and filter triangles based on maximum edge length.

    Parameters:
        points (numpy.ndarray): Nx3 array of 3D point cloud coordinates.
        max_edge_length (float): Maximum allowed edge length for triangles.

    Returns:
        list: Valid triangles and leftover points.
    """
    points_2d = points[:, [0, 2]]  # Project onto XZ-plane
    delaunay = Delaunay(points_2d)

    valid_triangles = []
    leftover_points = set(range(len(points)))

    for simplex in delaunay.simplices:
        p1, p2, p3 = points[simplex]
        edge_lengths = [
            np.linalg.norm(p1 - p2),
            np.linalg.norm(p2 - p3),
            np.linalg.norm(p3 - p1),
        ]
        if all(length <= max_edge_length for length in edge_lengths):
            valid_triangles.append(simplex)
            leftover_points -= set(simplex)

    return valid_triangles, leftover_points


def fill_gaps_with_kdtree(points, leftover_points, valid_triangles, max_attempts=100):
    """
    Fill gaps by iteratively finding the three closest points and triangulating them using a KDTree.

    Parameters:
        points (numpy.ndarray): Nx3 array of 3D point cloud coordinates.
        leftover_points (set): Indices of points without enough triangles.
        valid_triangles (list): List of valid triangles.
        max_attempts (int): Maximum number of attempts to fill gaps.

    Returns:
        list: Updated list of triangles.
    """
    leftover_points = list(leftover_points)
    new_triangles = []
    attempts = 0

    while len(leftover_points) >= 3 and attempts < max_attempts:
        # Build a KDTree from the leftover points
        kdtree = KDTree(points[leftover_points])

        # Find the three closest points
        point_idx = leftover_points[0]
        distances, indices = kdtree.query(points[point_idx], k=3)

        if len(indices) == 3:  # Ensure at least 3 points are found
            triangle_indices = [leftover_points[idx] for idx in indices]
            new_triangles.append(triangle_indices)

            # Remove used points from the leftover set
            leftover_points = [idx for idx in leftover_points if idx not in triangle_indices]

        attempts += 1

    return valid_triangles + new_triangles


def find_boundary_edges(mesh):
    """
    Identify boundary edges in a TriangleMesh.

    Parameters:
        mesh (o3d.geometry.TriangleMesh): Input mesh.

    Returns:
        list: A list of boundary edges.
    """
    edge_dict = {}
    triangles = np.asarray(mesh.triangles)

    for triangle in triangles:
        edges = [
            tuple(sorted([triangle[0], triangle[1]])),
            tuple(sorted([triangle[1], triangle[2]])),
            tuple(sorted([triangle[2], triangle[0]])),
        ]
        for edge in edges:
            if edge in edge_dict:
                edge_dict[edge] += 1
            else:
                edge_dict[edge] = 1

    # Boundary edges appear only once
    boundary_edges = [edge for edge, count in edge_dict.items() if count == 1]
    return boundary_edges



def group_edges_into_loops(boundary_edges, vertices, distance_threshold=5.0):
    """
    Group boundary edges into loops using a recursive DFS approach.

    Parameters:
        boundary_edges (list): List of boundary edges (vertex index pairs).
        vertices (np.ndarray): Mesh vertices (Nx3 array).
        distance_threshold (float): Maximum distance to close gaps.

    Returns:
        list: List of loops, where each loop is a list of vertex indices.
    """
    # Create a mapping of vertex to connected vertices
    edge_map = {}
    for edge in boundary_edges:
        edge_map.setdefault(edge[0], []).append(edge[1])
        edge_map.setdefault(edge[1], []).append(edge[0])

    visited_edges = set()  # To track visited edges
    loops = []  # To store detected loops

    def dfs(current, parent, loop):
        """
        Recursive Depth-First Search to detect loops.

        Parameters:
            current (int): Current vertex being visited.
            parent (int): Parent vertex to avoid backtracking.
            loop (list): Current path of the traversal.
        """
        loop.append(current)

        for neighbor in edge_map[current]:
            edge = tuple(sorted([current, neighbor]))
            if edge in visited_edges:
                continue  # Skip already visited edges

            # Mark the edge as visited
            visited_edges.add(edge)

            if neighbor == parent:
                # Skip backtracking to the parent
                continue

            if neighbor in loop:
                # A loop is detected
                loop_start = loop.index(neighbor)
                loops.append(loop[loop_start:])
                return

            dfs(neighbor, current, loop)

        # Backtrack
        loop.pop()

    # Traverse all vertices
    for start_vertex in edge_map:
        if any((start_vertex, neighbor) in visited_edges or (neighbor, start_vertex) in visited_edges
               for neighbor in edge_map[start_vertex]):
            continue  # Skip if all edges for this vertex are already visited

        dfs(start_vertex, None, [])

    print(f"Total loops detected: {len(loops)}")
    return loops


def fill_holes(mesh, max_edge_length, distance_threshold=5.0):
    """
    Fill holes in a TriangleMesh.

    Parameters:
        mesh (o3d.geometry.TriangleMesh): Input mesh.
        max_edge_length (float): Maximum allowed edge length for triangulation.
        distance_threshold (float): Maximum distance to close gaps in loops.

    Returns:
        o3d.geometry.TriangleMesh: Updated mesh with holes filled.
    """
    boundary_edges = find_boundary_edges(mesh)
    vertices = np.asarray(mesh.vertices)

    # Group edges into logical loops
    loops = group_edges_into_loops(boundary_edges, vertices, distance_threshold)

    triangles = np.asarray(mesh.triangles).tolist()

    for loop in loops:
        if len(loop) < 3:
            continue  # Ignore loops with fewer than 3 vertices

        # Triangulate the loop (simple fan triangulation)
        center = np.mean(vertices[loop], axis=0)
        center_idx = len(vertices)
        vertices = np.vstack([vertices, center])

        for i in range(len(loop)):
            v1 = loop[i]
            v2 = loop[(i + 1) % len(loop)]
            edge_length = np.linalg.norm(vertices[v1] - vertices[v2])
            if edge_length <= max_edge_length:
                triangles.append([v1, v2, center_idx])

    # Update the mesh
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(triangles)
    return mesh

def visualize_loops(mesh, loops):
    """
    Visualize the detected loops using Open3D.

    Parameters:
        mesh (o3d.geometry.TriangleMesh): The input mesh.
        loops (list): List of loops, where each loop is a list of vertex indices.
    """
    vertices = np.asarray(mesh.vertices)
    line_sets = []

    for loop in loops:
        lines = [[loop[i], loop[(i + 1) % len(loop)]] for i in range(len(loop))]
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(vertices)
        line_set.lines = o3d.utility.Vector2iVector(lines)

        # Assign a random color to the loop
        color = [random.random(), random.random(), random.random()]
        line_set.colors = o3d.utility.Vector3dVector([color] * len(lines))
        line_sets.append(line_set)

    return line_sets

def visualize_boundary_edges(mesh, boundary_edges):
    """
    Visualize the detected boundary edges using Open3D.

    Parameters:
        mesh (o3d.geometry.TriangleMesh): The input mesh.
        boundary_edges (list): List of boundary edges.
    """
    vertices = np.asarray(mesh.vertices)
    lines = [[edge[0], edge[1]] for edge in boundary_edges]

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(vertices)
    line_set.lines = o3d.utility.Vector2iVector(lines)

    # Assign a random color to the boundary edges
    line_set.colors = o3d.utility.Vector3dVector([[1, 0, 0]] * len(lines))  # Red color
    return line_set


def filter_triangles_by_edge_length(mesh, max_edge_length):
    """
    Filter triangles in a mesh by removing those with edges exceeding a specified length.

    Parameters:
        mesh (o3d.geometry.TriangleMesh): Input mesh.
        max_edge_length (float): Maximum allowed edge length.

    Returns:
        o3d.geometry.TriangleMesh: Filtered mesh with long-edge triangles removed.
    """
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)

    valid_triangles = []
    for triangle in triangles:
        # Get the vertices of the triangle
        v1, v2, v3 = vertices[triangle[0]], vertices[triangle[1]], vertices[triangle[2]]

        # Calculate edge lengths
        edge_lengths = [
            np.linalg.norm(v1 - v2),
            np.linalg.norm(v2 - v3),
            np.linalg.norm(v3 - v1),
        ]

        # Check if all edges are within the maximum length
        if all(length <= max_edge_length for length in edge_lengths):
            valid_triangles.append(triangle)

    # Update the mesh with valid triangles
    filtered_mesh = mesh
    filtered_mesh.triangles = o3d.utility.Vector3iVector(valid_triangles)
    return filtered_mesh


def remove_isolated_points(points, min_distance=10.0):
    """
    Remove points that have no neighbors within a specified distance.

    Parameters:
        points (np.ndarray): Nx3 array of point coordinates.
        min_distance (float): Minimum distance to consider a neighbor.

    Returns:
        np.ndarray: Filtered points with isolated points removed.
    """
    # Build a KDTree for efficient neighbor search
    kdtree = KDTree(points)

    # Filter points by checking their neighbors
    filtered_points = []
    for i, point in enumerate(points):
        distances, indices = kdtree.query(point, k=2)  # Find the 2 nearest points (self + closest neighbor)
        if distances[1] <= min_distance:  # Check if the closest neighbor is within the min_distance
            filtered_points.append(point)

    return np.array(filtered_points)


def filter_top_n_components(mesh, n):
    # Step 1: Get vertices and triangles
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)

    # Step 2: Build adjacency graph
    adjacency = defaultdict(set)
    for tri in triangles:
        for i in range(3):
            adjacency[tri[i]].update(tri[np.arange(3) != i])

    # Step 3: Find connected components
    visited = set()
    components = []

    def dfs(v, current_component):
        stack = [v]
        while stack:
            node = stack.pop()
            if node not in visited:
                visited.add(node)
                current_component.add(node)
                stack.extend(adjacency[node] - visited)

    for v in range(len(vertices)):
        if v not in visited:
            component = set()
            dfs(v, component)
            components.append(component)

    # Step 4: Rank components by size
    components = sorted(components, key=lambda x: len(x), reverse=True)

    # Step 5: Keep top N components
    top_components = components[:n]
    selected_indices = set.union(*top_components)

    # Step 6: Filter vertices and triangles
    index_map = {old_idx: new_idx for new_idx, old_idx in enumerate(selected_indices)}
    new_vertices = vertices[list(selected_indices)]
    new_triangles = []
    for tri in triangles:
        if set(tri).issubset(selected_indices):
            new_triangles.append([index_map[idx] for idx in tri])
    new_triangles = np.array(new_triangles)

    # Step 7: Create new mesh
    new_mesh = o3d.geometry.TriangleMesh()
    new_mesh.vertices = o3d.utility.Vector3dVector(new_vertices)
    new_mesh.triangles = o3d.utility.Vector3iVector(new_triangles)
    return new_mesh

def filter_overconnected_triangles(mesh):
    """
    Removes triangles from the mesh that share edges with more than two triangles, 
    prioritizing removal of the most isolated triangle.

    Args:
        mesh (o3d.geometry.TriangleMesh): The input triangular mesh.

    Returns:
        o3d.geometry.TriangleMesh: The filtered triangular mesh.
    """
    # Step 1: Extract vertices and triangles
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    
    # Step 2: Build edge-to-triangle map
    edge_to_triangles = defaultdict(list)
    for i, tri in enumerate(triangles):
        edges = [
            tuple(sorted([tri[0], tri[1]])),
            tuple(sorted([tri[1], tri[2]])),
            tuple(sorted([tri[2], tri[0]]))
        ]
        for edge in edges:
            edge_to_triangles[edge].append(i)
    
    # Step 3: Identify edges shared by more than two triangles
    overconnected_edges = {edge: tris for edge, tris in edge_to_triangles.items() if len(tris) > 2}
    if not overconnected_edges:
        print("No overconnected edges found.")
        return mesh  # No overconnected edges; return the original mesh
    print(f"Found {len(overconnected_edges)} overconnected edges.")
    # Step 4: Count neighboring triangles for each triangle
    triangle_neighbors = defaultdict(set)
    for edge, tris in edge_to_triangles.items():
        for i in range(len(tris)):
            for j in range(i + 1, len(tris)):
                triangle_neighbors[tris[i]].add(tris[j])
                triangle_neighbors[tris[j]].add(tris[i])
    
    # Step 5: Find the most isolated triangle among overconnected triangles
    triangles_to_remove = set()
    for edge, tris in overconnected_edges.items():
        # Identify the triangle with the fewest neighbors
        most_isolated_triangle = min(tris, key=lambda tri: len(triangle_neighbors[tri]))
        triangles_to_remove.add(most_isolated_triangle)
    
    # Step 6: Filter out the removed triangles
    remaining_triangles = [tri for i, tri in enumerate(triangles) if i not in triangles_to_remove]
    
    # Step 7: Create new mesh with remaining triangles
    new_mesh = o3d.geometry.TriangleMesh()
    new_mesh.vertices = mesh.vertices
    new_mesh.triangles = o3d.utility.Vector3iVector(remaining_triangles)
    return new_mesh

# Load the original point cloud
pcd_file = "video_processing/point_clouds/@062 255 2024-12-05 13-12-57.pcd"
pcd = o3d.io.read_point_cloud(pcd_file)
points = np.asarray(pcd.points)
print(len(points))
o3d.visualization.draw_geometries([pcd], window_name="Original PCD")

# Cluster points in the XZ plane
radius = 1 # Clustering radius in XZ plane
clustered_points = cluster_points_in_xz(points, radius)

# clustered_points = remove_isolated_points(clustered_points, 9)

clustered_pcd = o3d.geometry.PointCloud()
clustered_pcd.points = o3d.utility.Vector3dVector(clustered_points)

o3d.visualization.draw_geometries([clustered_pcd], window_name="Clustered Points")

# Perform Delaunay triangulation with edge length filtering
max_edge_length = 60.0  # Maximum allowed edge length for triangles
valid_triangles, leftover_points = delaunay_triangulation_with_edge_filter(clustered_points, max_edge_length)

# Fill gaps by iteratively triangulating the closest points with KDTree optimization
print("Filling gaps in the triangulation...")
max_attempts = 1000  # Limit the number of attempts to fill gaps
print(len(valid_triangles))
filled_triangles = fill_gaps_with_kdtree(clustered_points, leftover_points, valid_triangles, max_attempts)
print(len(filled_triangles))
# Create a TriangleMesh in Open3D
mesh = o3d.geometry.TriangleMesh()
mesh.vertices = o3d.utility.Vector3dVector(clustered_points)
mesh.triangles = o3d.utility.Vector3iVector(filled_triangles)

boundary_edges = find_boundary_edges(mesh)

# Check if any boundary edges were detected
if not boundary_edges:
    print("No boundary edges detected. Please check the mesh.")
else:
    print(f"Detected {len(boundary_edges)} boundary edges.")

# Visualize the boundary edges
boundary_visualization = visualize_boundary_edges(mesh, boundary_edges)

# Visualize the mesh and the boundary edges
o3d.visualization.draw_geometries([mesh, boundary_visualization], window_name="Boundary Edges")

distance_threshold = 5.0  # Threshold for closing gaps in loops
loops = group_edges_into_loops(boundary_edges, np.asarray(mesh.vertices), distance_threshold)

# Visualize the loops
print("Visualizing detected loops...")
loop_visualizations = visualize_loops(mesh, loops)
print(len(loop_visualizations))

# Visualize the mesh and the loops
o3d.visualization.draw_geometries([mesh] + loop_visualizations, window_name="Detected Loops")


mesh = fill_holes(mesh, max_edge_length, distance_threshold=5.0)


mesh = filter_triangles_by_edge_length(mesh, 100)

mesh = filter_top_n_components(mesh, n=1)
print("Filtering top n components...")


mesh = filter_overconnected_triangles(mesh)


# DOUBLE SIDED 

original_triangles = np.asarray(mesh.triangles)
reversed_triangles = np.flip(original_triangles, axis=1)  # Reverse vertex order

# Combine original and reversed triangles
double_sided_triangles = np.vstack((original_triangles, reversed_triangles))
mesh.triangles = o3d.utility.Vector3iVector(double_sided_triangles)

# Compute normals for the mesh
mesh.compute_triangle_normals()

# Visualize the updated mesh and points
print("Visualizing the updated mesh and points...")

o3d.visualization.draw_geometries([mesh, clustered_pcd], window_name="Updated Mesh with Filled Gaps")


