import open3d as o3d
import numpy as np
from sklearn.cluster import DBSCAN
from scipy.spatial import Delaunay
from scipy.spatial import KDTree
import random
from collections import defaultdict
from scipy.spatial import ConvexHull, QhullError
from tqdm import tqdm


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


def fill_holes(mesh, max_edge_length, distance_threshold=20.0):
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

    lines = []
    for edge in overconnected_edges:
        lines.append(edge)

    highlighted_triangles = set()
    for tris in overconnected_edges.values():
        highlighted_triangles.update(tris)
    
    # Create a LineSet for edges
    lineset = o3d.geometry.LineSet()
    lineset.points = o3d.utility.Vector3dVector(vertices)
    lineset.lines = o3d.utility.Vector2iVector(np.array(lines, dtype=int))

    # Assign color to edges (red)
    edge_colors = [[1, 0, 0] for _ in lines]
    lineset.colors = o3d.utility.Vector3dVector(edge_colors)

    # Create a new mesh for triangles
    triangle_vertices = vertices
    triangle_indices = [triangles[i] for i in highlighted_triangles]
    triangle_colors = np.array([[1, 0, 0] for _ in highlighted_triangles])  # Red for highlighted triangles
    
    highlighted_mesh = o3d.geometry.TriangleMesh()
    highlighted_mesh.vertices = o3d.utility.Vector3dVector(triangle_vertices)
    highlighted_mesh.triangles = o3d.utility.Vector3iVector(triangle_indices)
    highlighted_mesh.vertex_colors = o3d.utility.Vector3dVector(
        np.zeros_like(triangle_vertices))  # Default to black for unused vertices
    highlighted_mesh.paint_uniform_color([1, 0, 0])

    o3d.visualization.draw_geometries([mesh, lineset, highlighted_mesh])


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

def visualize_mesh_with_fixed_colors(mesh):
    """
    Visualize the mesh with each triangle assigned a color from a fixed set (red, yellow, green, blue, purple).

    Args:
        mesh (o3d.geometry.TriangleMesh): The input triangular mesh.

    Returns:
        o3d.geometry.TriangleMesh: The mesh with colored triangles.
    """
    # Step 1: Extract vertices and triangles
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    
    # Step 2: Define fixed colors
    fixed_colors = np.array([
        [1, 0, 0],   # Red
        [1, 1, 0],   # Yellow
        [0, 1, 0],   # Green
        [0, 0, 1],   # Blue
        [0.5, 0, 0.5]  # Purple
    ])
    num_colors = len(fixed_colors)

    # Step 3: Assign colors to vertices based on triangle colors
    vertex_colors = np.zeros_like(vertices)
    vertex_color_counts = np.zeros(len(vertices))  # To average colors for shared vertices

    for i, tri in enumerate(triangles):
        color = fixed_colors[i % num_colors]  # Cycle through fixed colors
        for vertex_idx in tri:
            vertex_colors[vertex_idx] += color
            vertex_color_counts[vertex_idx] += 1

    # Normalize vertex colors by averaging for shared vertices
    vertex_colors /= vertex_color_counts[:, None]

    # Step 4: Create a new mesh with colored vertices
    colored_mesh = o3d.geometry.TriangleMesh()
    colored_mesh.vertices = mesh.vertices
    colored_mesh.triangles = mesh.triangles
    colored_mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)
    
    return colored_mesh


def remove_negative_y_normals(mesh):
    """
    Removes triangles from the mesh that have normals pointing in the negative y-direction.

    Args:
        mesh (o3d.geometry.TriangleMesh): The input triangular mesh.

    Returns:
        o3d.geometry.TriangleMesh: The filtered triangular mesh.
    """
    # Step 1: Compute triangle normals
    mesh.compute_triangle_normals()
    triangle_normals = np.asarray(mesh.triangle_normals)

    # Step 2: Identify triangles with normals pointing in the negative y-direction
    valid_triangles = [
        i for i, normal in enumerate(triangle_normals) if normal[1] >= 0
    ]

    # Step 3: Filter the triangles and rebuild the mesh
    triangles = np.asarray(mesh.triangles)
    vertices = np.asarray(mesh.vertices)
    filtered_triangles = triangles[valid_triangles]

    # Create a new mesh
    filtered_mesh = o3d.geometry.TriangleMesh()
    filtered_mesh.vertices = o3d.utility.Vector3dVector(vertices)
    filtered_mesh.triangles = o3d.utility.Vector3iVector(filtered_triangles)
    
    return filtered_mesh

def visualize_topographical_map_multi_color(mesh):
    """
    Visualizes the mesh as a topographical map with a smooth gradient transitioning
    through red, yellow, green, blue, and purple based on y-coordinates.

    Args:
        mesh (o3d.geometry.TriangleMesh): The input triangular mesh.

    Returns:
        o3d.geometry.TriangleMesh: The topographically colored mesh with a multi-color gradient.
    """
    # Step 1: Extract vertex coordinates
    vertices = np.asarray(mesh.vertices)

    # Step 2: Normalize y-coordinates to [0, 1]
    y_coords = vertices[:, 1]
    min_y, max_y = np.min(y_coords), np.max(y_coords)
    normalized_y = (y_coords - min_y) / (max_y - min_y)

    # Step 3: Map normalized y-coordinates to colors in the multi-color gradient
    colors = np.zeros((len(vertices), 3))
    for i, value in enumerate(normalized_y):
        if value < 0.25:  # Red to Yellow
            t = value / 0.25
            colors[i] = [1, t, 0]
        elif value < 0.5:  # Yellow to Green
            t = (value - 0.25) / 0.25
            colors[i] = [1 - t, 1, 0]
        elif value < 0.75:  # Green to Blue
            t = (value - 0.5) / 0.25
            colors[i] = [0, 1 - t, t]
        else:  # Blue to Purple
            t = (value - 0.75) / 0.25
            colors[i] = [t, 0, 1]

    # Step 4: Assign vertex colors to the mesh
    multi_color_mesh = o3d.geometry.TriangleMesh()
    multi_color_mesh.vertices = mesh.vertices
    multi_color_mesh.triangles = mesh.triangles
    multi_color_mesh.vertex_colors = o3d.utility.Vector3dVector(colors)

    return multi_color_mesh

def highlight_and_visualize_intersecting_triangles(mesh):
    """
    Highlights intersecting triangles in red and visualizes the mesh.

    Args:
        mesh (o3d.geometry.TriangleMesh): The input triangular mesh.
    """
    # Step 1: Detect intersecting triangles
    intersecting_triangles = mesh.get_self_intersecting_triangles()
    print("Number of intersecting triangles:", len(intersecting_triangles))
    num_vertices = len(mesh.vertices)
    vertex_colors = np.ones((num_vertices, 3))  # Default to white for all vertices

    # Highlight vertices of intersecting triangles in red
    triangles = np.asarray(mesh.triangles)
    for idx in intersecting_triangles:
        triangle = triangles[idx]
        for vertex_idx in triangle:
            vertex_colors[vertex_idx] = [1, 0, 0]  # Red color for vertices of intersecting triangles

    # Step 3: Create a new mesh and assign vertex colors
    colored_mesh = o3d.geometry.TriangleMesh()
    colored_mesh.vertices = mesh.vertices
    colored_mesh.triangles = mesh.triangles
    colored_mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)

    colored_mesh.compute_vertex_normals()

    # Step 4: Visualize the mesh
    o3d.visualization.draw_geometries([colored_mesh], window_name="Intersecting Triangles")


def sort_vertices_around_center(center, vertices):
    """
    Sort vertices around a center point in a consistent order (clockwise or counterclockwise).
    """
    # Calculate the centroid of the vertices to define a reference plane
    centroid = np.mean(vertices, axis=0)
    vectors = vertices - center

    # Project vectors onto a plane orthogonal to the average normal
    normal = np.cross(vectors[0], vectors[1])  # Approximate normal using cross product
    normal = normal / np.linalg.norm(normal)
    projection_matrix = np.eye(3) - np.outer(normal, normal)
    projected_vectors = (projection_matrix @ vectors.T).T

    # Calculate angles around the center in 2D projection
    angles = np.arctan2(projected_vectors[:, 1], projected_vectors[:, 0])
    sorted_indices = np.argsort(angles)
    return vertices[sorted_indices]

def are_points_collinear(points, tolerance=1e-6):
    """
    Check if a set of 3D points are collinear.
    """
    if len(points) < 3:
        return True  # Fewer than 3 points are trivially collinear
    
    # Compute vectors between points
    base_vector = points[1] - points[0]
    for i in range(2, len(points)):
        check_vector = points[i] - points[0]
        cross_product = np.cross(base_vector, check_vector)
        if np.linalg.norm(cross_product) > tolerance:
            return False  # Points are not collinear
    return True  # All points are collinear


def calculate_solid_angle(mesh):
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    solid_angle_results = []
    vertex_surfaces = []

    for i, vertex in enumerate(vertices):
        # Find connected triangles
        connected_triangles = [tri for tri in triangles if i in tri]

        # Find first-order neighbors
        neighbors = set()
        for tri in connected_triangles:
            for v in tri:
                if v != i:  # Skip the current vertex
                    neighbors.add(v)
        neighbor_vertices = vertices[list(neighbors)]

        # Skip vertices with fewer than 2 neighbors
        if len(neighbor_vertices) < 3:
            solid_angle_results.append(0)  # Assign a solid angle of 0
            vertex_surfaces.append((vertex, None, None, None, None))  # No surface info
            continue

        # Check for collinearity
        if are_points_collinear(neighbor_vertices):
            solid_angle_results.append(0)  # Assign a solid angle of 0
            vertex_surfaces.append((vertex, None, None, None, None))  # No surface info
            continue

        # Fit a plane to the neighbors
        plane_normal, plane_point = fit_plane(neighbor_vertices)

        # Sort neighbors to form a valid polygon
        sorted_neighbors = sort_vertices_around_center(vertex, neighbor_vertices)

        # Calculate the polygon area on the plane
        area = calculate_polygon_area_3d(sorted_neighbors, plane_normal)

        # Calculate the distance from the original vertex to the plane
        vector_to_plane = vertex - plane_point
        min_distance = np.abs(np.dot(vector_to_plane, plane_normal))

        # Compute A / r^2
        if min_distance > 0:
            solid_angle_measure = area / (min_distance ** 2)
        else:
            solid_angle_measure = 0  # Avoid division by zero

        solid_angle_results.append(solid_angle_measure)
        vertex_surfaces.append((vertex, neighbor_vertices, sorted_neighbors, plane_normal, plane_point))
    
    return solid_angle_results, vertex_surfaces


def fit_plane(points):
    """
    Fit a plane to a set of 3D points using Singular Value Decomposition (SVD).
    Returns the plane normal and a point on the plane.
    """
    centroid = np.mean(points, axis=0)
    centered_points = points - centroid
    _, _, vh = np.linalg.svd(centered_points)
    normal = vh[-1]  # The last singular vector corresponds to the normal
    return normal, centroid

def project_points_onto_plane(points, plane_normal, plane_point):
    """
    Project 3D points onto a plane defined by a normal and a point on the plane.
    """
    plane_normal = plane_normal / np.linalg.norm(plane_normal)
    projections = []
    for p in points:
        vector_to_plane = p - plane_point
        distance_to_plane = np.dot(vector_to_plane, plane_normal)
        projection = p - distance_to_plane * plane_normal
        projections.append(projection)
    return np.array(projections)

def calculate_polygon_area_3d(points, plane_normal):
    """
    Calculate the area of a polygon defined by a set of 3D points.
    Assumes the points are coplanar and ordered (e.g., via `sort_vertices_around_center`).
    """
    # Project points to 2D for polygon area calculation
    plane_normal = plane_normal / np.linalg.norm(plane_normal)
    projection_matrix = np.eye(3) - np.outer(plane_normal, plane_normal)  # Orthogonal projection
    points_2d = (projection_matrix @ points.T).T[:, :2]  # Drop to 2D by ignoring one dimension

    # Use the shoelace formula to calculate the polygon area
    area = 0.0
    for i in range(len(points_2d)):
        x1, y1 = points_2d[i]
        x2, y2 = points_2d[(i + 1) % len(points_2d)]
        area += x1 * y2 - y1 * x2
    return abs(area) / 2.0

def visualize_vertex_surface(mesh, vertex_surface):
    """
    Visualize the given vertex, its neighbors, the projected polygon, and the plane.
    """
    vertex, neighbor_vertices, projected_neighbors, plane_normal, plane_point = vertex_surface

    # Skip visualization if there are no valid neighbors
    if neighbor_vertices is None or projected_neighbors is None:
        print("No valid neighbors for this vertex. Skipping visualization.")
        return

    # Create Open3D geometries
    mesh.paint_uniform_color([0.7, 0.7, 0.7])  # Light gray for the mesh

    # Highlight the vertex
    vertex_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=1)
    vertex_sphere.translate(vertex)
    vertex_sphere.paint_uniform_color([1, 0, 0])  # Red for the selected vertex

    # Highlight neighbor vertices
    neighbor_spheres = []
    for neighbor in neighbor_vertices:
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.5)
        sphere.translate(neighbor)
        sphere.paint_uniform_color([0, 1, 0])  # Green for neighbors
        neighbor_spheres.append(sphere)

    # Create a line set for the polygon
    polygon_lines = []
    for i in range(len(projected_neighbors)):
        start = projected_neighbors[i]
        end = projected_neighbors[(i + 1) % len(projected_neighbors)]
        line = o3d.geometry.LineSet()
        line.points = o3d.utility.Vector3dVector([start, end])
        line.lines = o3d.utility.Vector2iVector([[0, 1]])
        line.paint_uniform_color([0, 0, 1])  # Blue for the polygon
        polygon_lines.append(line)

    # Visualize all geometries
    o3d.visualization.draw_geometries(
        [mesh, vertex_sphere, *neighbor_spheres, *polygon_lines],
        window_name="Vertex Surface Visualization",
    )


def smooth_solid_angles(mesh, threshold=0.01, max_iterations=5):
    smallest_angle = 0
    solid_angles, vertex_surfaces = calculate_solid_angle(mesh)
    counter = 1
    while smallest_angle <= threshold and counter <= max_iterations:
        print(f"Iteration {counter} smoothing...")
        vertices = np.asarray(mesh.vertices)
        for i in tqdm(range(len(solid_angles))):
            if solid_angles[i] > 0 and solid_angles[i] <= threshold:
                vertex, neighbor_vertices, projected_neighbors, plane_normal, plane_point = vertex_surfaces[i]

                distance = np.dot(vertex - plane_point, plane_normal)
                projected_vertex = vertex - distance * plane_normal

                vertices[i] = projected_vertex

        mesh.vertices = o3d.utility.Vector3dVector(vertices)
        mesh.compute_vertex_normals()
        solid_angles, vertex_surfaces = calculate_solid_angle(mesh)
        smallest_angle = min([s for s in solid_angles if s != 0])
        counter += 1
    return mesh



# Load the original point cloud
pcd_file = "video_processing/point_clouds/@026 255 2024-11-02 16-15-44.pcd"
pcd = o3d.io.read_point_cloud(pcd_file)
points = np.asarray(pcd.points)
print(len(points))
#o3d.visualization.draw_geometries([pcd], window_name="Original PCD")

# Cluster points in the XZ plane
radius = 1 # Clustering radius in XZ plane
clustered_points = cluster_points_in_xz(points, radius)

#clustered_points = remove_isolated_points(clustered_points, 6)

clustered_pcd = o3d.geometry.PointCloud()
clustered_pcd.points = o3d.utility.Vector3dVector(clustered_points)

#o3d.visualization.draw_geometries([clustered_pcd], window_name="Clustered Points")

# Perform Delaunay triangulation with edge length filtering
max_edge_length = 100.0  # Maximum allowed edge length for triangles
valid_triangles, leftover_points = delaunay_triangulation_with_edge_filter(clustered_points, max_edge_length)

# initial_mesh = o3d.geometry.TriangleMesh()
# initial_mesh.vertices = o3d.utility.Vector3dVector(clustered_points)
# initial_mesh.triangles = o3d.utility.Vector3iVector(valid_triangles)
# o3d.visualization.draw_geometries([initial_mesh], window_name="pre y removal")
# initial_mesh = remove_negative_y_normals(initial_mesh)
# o3d.visualization.draw_geometries([initial_mesh], window_name="Removed negative y")
# valid_triangles = initial_mesh.triangles

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


colored_mesh = visualize_mesh_with_fixed_colors(mesh)
#o3d.visualization.draw_geometries([colored_mesh, clustered_pcd], window_name="Colored Mesh")

boundary_edges = find_boundary_edges(mesh)

# Check if any boundary edges were detected
if not boundary_edges:
    print("No boundary edges detected. Please check the mesh.")
else:
    print(f"Detected {len(boundary_edges)} boundary edges.")

# Visualize the boundary edges
boundary_visualization = visualize_boundary_edges(mesh, boundary_edges)

# Visualize the mesh and the boundary edges
#o3d.visualization.draw_geometries([mesh, boundary_visualization], window_name="Boundary Edges")

distance_threshold = 5.0  # Threshold for closing gaps in loops
loops = group_edges_into_loops(boundary_edges, np.asarray(mesh.vertices), distance_threshold)

# Visualize the loops
print("Visualizing detected loops...")
loop_visualizations = visualize_loops(mesh, loops)
print(len(loop_visualizations))

# Visualize the mesh and the loops
#o3d.visualization.draw_geometries([mesh] + loop_visualizations, window_name="Detected Loops")


mesh = fill_holes(mesh, max_edge_length, distance_threshold=5.0)


mesh = filter_triangles_by_edge_length(mesh, 100)

mesh = filter_top_n_components(mesh, n=1)
print("Filtering top n components...")


mesh = filter_overconnected_triangles(mesh)

colored_mesh = visualize_mesh_with_fixed_colors(mesh)
#o3d.visualization.draw_geometries([colored_mesh, clustered_pcd], window_name="Colored Mesh")

# DOUBLE SIDED 

original_triangles = np.asarray(mesh.triangles)
reversed_triangles = np.flip(original_triangles, axis=1)  # Reverse vertex order

# Combine original and reversed triangles
double_sided_triangles = np.vstack((original_triangles, reversed_triangles))
mesh.triangles = o3d.utility.Vector3iVector(double_sided_triangles)

# Compute normals for the mesh
mesh.compute_triangle_normals()
mesh.compute_vertex_normals()



# # Calculate solid angle measures and store vertex surface data
# solid_angles, vertex_surfaces = calculate_solid_angle(mesh)
# print(min([solid_angle for solid_angle in solid_angles if solid_angle != 0]))
# Pick a vertex to visualize

# for vertex_index_to_visualize in np.argsort(solid_angles):
#     if solid_angles[vertex_index_to_visualize] < 100 and solid_angles[vertex_index_to_visualize] != 0:
#         print(f"Visualizing solid angle of value {solid_angles[vertex_index_to_visualize]}")
#         visualize_vertex_surface(mesh, vertex_surfaces[vertex_index_to_visualize])

o3d.visualization.draw_geometries([mesh, clustered_pcd], window_name="Before Mesh with Gaps")

mesh = smooth_solid_angles(mesh, threshold=10)



# Visualize the updated mesh and points
print("Visualizing the updated mesh and points...")

o3d.visualization.draw_geometries([mesh, clustered_pcd], window_name="Updated Mesh with Filled Gaps")

# colored_mesh = visualize_mesh_with_fixed_colors(mesh)
# o3d.visualization.draw_geometries([colored_mesh, clustered_pcd], window_name="Colored Mesh")

# topographical_mesh = visualize_topographical_map_multi_color(mesh)
# axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1000.0, origin=[0, 0, 0])
# o3d.visualization.draw_geometries([topographical_mesh, axes], window_name="Topographical Mesh")

# print("Number of triangles:", len(mesh.triangles))
# simplified_mesh = mesh.simplify_quadric_decimation(target_number_of_triangles=len(mesh.triangles) // 2)
# colored_mesh = visualize_mesh_with_fixed_colors(simplified_mesh)
# o3d.visualization.draw_geometries([colored_mesh, clustered_pcd], window_name="Colored Simplified Mesh")

highlight_and_visualize_intersecting_triangles(mesh)
