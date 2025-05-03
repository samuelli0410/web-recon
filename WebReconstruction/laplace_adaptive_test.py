import numpy as np
import open3d as o3d
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
from scipy.spatial import Delaunay
import heapq

def find_min_coord_point(points):
    """Finds the point with the smallest x, y, and z coordinates."""
    min_index = np.argmin(np.sum(points, axis=1))
    return min_index

def find_max_coord_point(points):
    """Finds the point with the largest x, y, and z coordinates."""
    min_index = np.argmax(np.sum(points, axis=1))
    return min_index

def estimate_local_density(points, k=10):
    """Estimate local density as inverse average k-NN distance (excluding self)."""
    nbrs = NearestNeighbors(n_neighbors=k+1).fit(points)
    distances, _ = nbrs.kneighbors(points)
    avg_distances = np.mean(distances[:, 1:], axis=1)  # skip distance to self
    densities = 1.0 / (avg_distances + 1e-8)
    return densities / np.max(densities)  # Normalize to [0, 1]

def adaptive_laplacian_contraction(points, k=10, base_lambda=3, exponent=1.2, iterations=1):
    """Apply adaptive Laplacian smoothing/contraction."""
    points = points.copy()
    nbrs = NearestNeighbors(n_neighbors=k+1).fit(points)

    for _ in range(iterations):
        distances, indices = nbrs.kneighbors(points)
        densities = estimate_local_density(points, k)

        new_points = points.copy()

        for i in range(len(points)):
            neighbors = indices[i][1:]  # skip self
            laplacian = np.mean(points[neighbors] - points[i], axis=0)
            adaptive_lambda = base_lambda * (densities[i] ** exponent)
            new_points[i] += adaptive_lambda * laplacian

        points = new_points
        nbrs.fit(points)  # Re-fit neighbor structure if needed

    return points

def remove_redundant_points(points, threshold=0.1):
    nbrs = NearestNeighbors(radius=threshold).fit(points)
    visited = np.zeros(len(points), dtype=bool)
    unique_points = []

    for i in range(len(points)):
        if visited[i]:
            continue

        # Mark all neighbors (including self) as visited
        _, indices = nbrs.radius_neighbors([points[i]])
        visited[indices[0]] = True

        # Keep just one representative point (could choose the centroid instead)
        unique_points.append(points[i])

    return np.array(unique_points)

def volexReduction(points, eps = 0.5):
    points = points.copy()
    model = DBSCAN(eps, min_samples = 3)
    clumps = model.fit_predict(points)
    labels = set(clumps)-{-1}
    print(len(labels))
    clump_dict = {}
    clumpPoints = set()
    for i in labels:
        clump_points = points[clumps == i]
        if len(clump_points)> 1:
            clump_center = np.median(clump_points, axis=0)    
            # print(clump_points, clumpcenter)
            for point in clump_points:
                clumpPoints.add(tuple(point))
                clump_dict[tuple(point)] = clump_center
                # print(clump_center)
    newpoints = [x if tuple(x) not in clumpPoints else clump_dict[tuple(x)] for x in points]
    return newpoints

def vis_pcd(contracted_points):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(contracted_points)
    pcd.colors = o3d.utility.Vector3dVector(np.zeros((len(pcd.points), 3)))

    # Write point cloud
    # o3d.io.write_point_cloud("video_processing/point_clouds/@059 255 2024-12-05 04-50-27 skeleton.pcd", pcd)

    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pcd)
    render_option = vis.get_render_option()
    render_option.point_size = 5
    vis.run()
    vis.destroy_window()

def visualize_graph(points, edges):
    """Visualizes the 3D points and their k-NN edges using Open3D."""
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    
    lines = [[e[0], e[1]] for e in edges]
    colors = [[1, 0, 0] for _ in lines]  # Red edges
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(colors)
    
    o3d.visualization.draw_geometries([point_cloud, line_set])

def visualize_graph_points_overlay(points, edges, cloud):
    """Visualizes the 3D points and their k-NN edges using Open3D."""
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    point_cloud.colors = o3d.utility.Vector3dVector([[0, 0, 0] for _ in points])
    
    
    lines = [[e[0], e[1]] for e in edges]
    colors = [[0, 0, 0] for _ in lines]  # Red edges
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(colors)
    
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(point_cloud)
    vis.add_geometry(line_set)
    vis.add_geometry(cloud)
    render_option = vis.get_render_option()
    render_option.point_size = 10
    vis.run()
    vis.destroy_window()

def prim(points, rootidx, k=10):
    n = len(points)
    if n == 0:
        return []
    
    # Keep track of points included in the MST.
    in_mst = [False] * n
    mst_edges = []

    # Start from the given root index.
    in_mst[rootidx] = True
    edge_heap = []
    
    # Initialize the heap with edges from the root to all other points.
    for j in range(n):
        if j == rootidx:
            continue
        weight = np.linalg.norm(points[rootidx] - points[j])
        heapq.heappush(edge_heap, (weight, rootidx, j))


    # Build the MST until we have n-1 edges.
    while edge_heap and len(mst_edges) < n - 1:
        # Uncomment the following line if you need to see timing info.
        # print(time.time() - start)
        weight, i, j = heapq.heappop(edge_heap)
        
        # If j is already in the MST, skip this edge.
        if in_mst[j]:
            continue
        
        # Add the edge to the MST.
        mst_edges.append([i, j])
        in_mst[j] = True
        
        # Add new candidate edges from the newly added vertex j.
        for k in range(n):
            if not in_mst[k]:
                new_weight = np.linalg.norm(points[j] - points[k])
                heapq.heappush(edge_heap, (new_weight, j, k))
    
    return mst_edges

def prim_directional(points, rootidx, k=10, angle_bias_strength=0.5, angle_threshold_degrees=30):
    n = len(points)
    if n == 0:
        return []
    
    in_mst = [False] * n
    mst_edges = []

    in_mst[rootidx] = True
    edge_heap = []
    
    # We'll track the direction vectors for each node after it's connected
    directions = [None] * n

    for j in range(n):
        if j == rootidx:
            continue
        weight = np.linalg.norm(points[rootidx] - points[j])
        heapq.heappush(edge_heap, (weight, rootidx, j))

    while edge_heap and len(mst_edges) < n - 1:
        weight, i, j = heapq.heappop(edge_heap)
        
        if in_mst[j]:
            continue
        
        mst_edges.append([i, j])
        in_mst[j] = True

        # Save the last direction vector for j
        directions[j] = points[j] - points[i]
        directions[j] /= np.linalg.norm(directions[j])  # Normalize!

        for k in range(n):
            if not in_mst[k]:
                new_vec = points[k] - points[j]
                dist = np.linalg.norm(new_vec)
                if dist == 0:
                    continue

                # Normalize new_vec
                new_vec /= dist

                adjusted_dist = dist

                if directions[j] is not None:
                    # Compute the angle (via dot product)
                    dot = np.clip(np.dot(directions[j], new_vec), -1.0, 1.0)
                    angle = np.arccos(dot) * (180.0 / np.pi)  # Angle in degrees

                    if angle < angle_threshold_degrees:
                        # Reduce distance weight if the angle is small
                        adjusted_dist *= angle_bias_strength

                heapq.heappush(edge_heap, (adjusted_dist, j, k))

    return mst_edges

def delaunay_graph(points, distance_threshold=6):
    """
    Creates a graph based on Delaunay triangulation with a distance restriction.
    
    Parameters:
    - points: (n, d) ndarray, where n is the number of points and d is the dimensionality of the points.
    - distance_threshold: The maximum allowed distance for edges to be added to the graph.
    
    Returns:
    - edges: List of tuples representing edges in the graph.
    """
    # Perform Delaunay triangulation
    tri = Delaunay(points)
    
    # Initialize a set to store edges (avoid duplicates)
    edges = set()
    
    # Function to calculate Euclidean distance between two points
    def distance(p1, p2):
        return np.linalg.norm(p1 - p2)

    # Iterate over the triangles (simplices) in the triangulation
    for simplex in tri.simplices:
        # Extract the three edges of the triangle
        for i in range(3):
            # Create edge as a sorted tuple to ensure it's undirected
            edge = tuple(sorted([simplex[i], simplex[(i+1)%3]]))
            
            # Calculate the Euclidean distance between the points in the edge
            p1, p2 = points[edge[0]], points[edge[1]]
            dist = distance(p1, p2)
            
            # Only add the edge if the distance is less than or equal to the threshold
            if dist <= distance_threshold:
                edges.add(edge)
    
    # Convert the set of edges to a list and return
    return list(edges)

# degree 1 fix
# check angle

def euclidean_distance(p1, p2):
    return np.linalg.norm(p1 - p2)

def deg_list(edges, points):
    degree = [0] * len(points)
    for edge in edges:
        degree[edge[0]] += 1
        degree[edge[1]] += 1
    return degree
    
def find_degree_one_vertices(degree):
    return [i for i, deg in enumerate(degree) if deg == 1]

def angle_between_vectors(v1, v2):
    dot_product = np.dot(v1, v2)
    magnitude_v1 = np.linalg.norm(v1)
    magnitude_v2 = np.linalg.norm(v2)
    cos_theta = dot_product / (magnitude_v1 * magnitude_v2)
    return np.arccos(np.clip(cos_theta, -1.0, 1.0))

def is_angle_too_sharp_old(v, closest_vertex, points, mst_edges, degree, threshold_angle_deg=60):
    # Convert angle threshold from degrees to radians
    threshold_angle_rad = np.radians(threshold_angle_deg)

    # Find the neighbors of both v and closest_vertex
    neighbors_v = [i for i in range(len(points)) if (v, i) in mst_edges or (i, v) in mst_edges]
    neighbors_closest_vertex = [i for i in range(len(points)) if (closest_vertex, i) in mst_edges or (i, closest_vertex) in mst_edges]
    
    # Check if both vertices already have neighbors (we don't want to form a sharp angle with any existing edges)
    for neighbor in neighbors_v:
        if neighbor != closest_vertex:
            edge1 = points[neighbor] - points[v]  # Edge from v to its neighbor
            edge2 = points[closest_vertex] - points[v]  # New edge from v to closest_vertex
            angle = angle_between_vectors(edge1, edge2)
            if angle < threshold_angle_rad:
                return True  # Reject this edge because it creates a sharp angle

    for neighbor in neighbors_closest_vertex:
        if neighbor != v:
            edge1 = points[neighbor] - points[closest_vertex]  # Edge from closest_vertex to its neighbor
            edge2 = points[closest_vertex] - points[v]  # New edge from v to closest_vertex
            angle = angle_between_vectors(edge1, edge2)
            if angle < threshold_angle_rad:
                return True  # Reject this edge because it creates a sharp angle

    return False  # Accept the edge if no sharp angles are formed

def is_angle_too_sharp(v, neighbor, points, existing_edges, threshold_angle_deg=60, threshold_angle_deg_n=30):
    threshold_angle_rad = np.radians(threshold_angle_deg)
    threshold_angle_rad_n = np.radians(threshold_angle_deg_n)
    
    # Check at vertex v
    neighbors_v = [i for i in range(len(points)) if (v, i) in existing_edges or (i, v) in existing_edges]
    for n in neighbors_v:
        if n != neighbor:
            edge1 = points[n] - points[v]
            edge2 = points[neighbor] - points[v]
            angle = angle_between_vectors(edge1, edge2)
            if angle < threshold_angle_rad:
                return True  # Sharp angle at v
    
    # Check at vertex neighbor (candidate)
    neighbors_neighbor = [i for i in range(len(points)) if (neighbor, i) in existing_edges or (i, neighbor) in existing_edges]
    for n in neighbors_neighbor:
        if n != v:
            edge1 = points[n] - points[neighbor]
            edge2 = points[v] - points[neighbor]
            angle = angle_between_vectors(edge1, edge2)
            if angle < threshold_angle_rad_n:
                return True  # Sharp angle at neighbor

    return False

def add_edges(mst_edges, points, degree, max_angle_threshold_deg=60):
    existing_edges = set(tuple(sorted(edge)) for edge in mst_edges)
    new_edges = []
    degree_one_vertices = find_degree_one_vertices(degree)

    for v in degree_one_vertices:
        candidates = []
        
        # Build list of all possible candidates and their (penalized) distances
        for i in range(len(points)):
            if i != v and tuple(sorted([v, i])) not in existing_edges:
                distance = euclidean_distance(points[v], points[i])
                if degree[i] == 1:
                    distance *= 1  # Favor connecting degree-1 vertices
                candidates.append((distance, i))
        
        # Sort candidates by penalized distance
        candidates.sort()

        # Now try to connect to the closest valid candidate
        for _, candidate in candidates:
            if not is_angle_too_sharp(v, candidate, points, existing_edges, threshold_angle_deg=max_angle_threshold_deg):
                # If angle check passes, add the edge
                new_edges.append((v, candidate))
                degree[v] += 1
                degree[candidate] += 1
                existing_edges.add(tuple(sorted((v, candidate))))  # Update existing edges immediately
                break  # Only add one edge per degree-1 vertex

    return new_edges

if __name__ == "__main__":
    pcd_o = o3d.io.read_point_cloud("video_processing/point_clouds/059_spiderless_quadrant.pcd")
    points = np.asarray(pcd_o.points)
    print(points.__len__())

    points = remove_redundant_points(points, threshold=1)
    contracted_points = adaptive_laplacian_contraction(points, k=15, iterations=5)
    cleaned_points = remove_redundant_points(contracted_points, threshold=1.5)

    new_pcd = o3d.geometry.PointCloud()
    new_pcd.points = o3d.utility.Vector3dVector(cleaned_points)
    print(cleaned_points.__len__())

    #cleaned_points = volexReduction(cleaned_points, eps=1.1)
    #print(cleaned_points.__len__())

    #vis_pcd(cleaned_points)
    
    root_idx = find_min_coord_point(cleaned_points)
    print("found")
    graph_edges = prim(cleaned_points, root_idx)
    #graph_edges = delaunay_graph(cleaned_points, distance_threshold=6)
    #graph_edges = prim_directional(cleaned_points, root_idx, angle_bias_strength=0.5, angle_threshold_degrees=15)
    print("prims completed")
    visualize_graph_points_overlay(cleaned_points, graph_edges, new_pcd)
    graph_edges = graph_edges + add_edges(graph_edges, cleaned_points, deg_list(graph_edges, cleaned_points))
    print("infill completed")
    visualize_graph_points_overlay(cleaned_points, graph_edges, new_pcd)
    