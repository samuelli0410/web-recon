import numpy as np
import open3d as o3d
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.spatial import KDTree
import heapq
import time
from scipy.spatial import KDTree
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
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





def dijkstra(points, root_index, k=10):
    """Runs Dijkstra's algorithm using a k-nearest neighbors graph for efficiency."""
    num_points = len(points)
    tree = KDTree(points)
    
    # Get k-nearest neighbors (excluding the point itself)
    distances, neighbors = tree.query(points, k=k+1)
    
    # Initialize distances and priority queue
    dist_dict = {i: float('inf') for i in range(num_points)}
    dist_dict[root_index] = 0
    pq = [(0, root_index)]  # (distance, node)
    edges = []
    
    while pq:
        current_dist, current_node = heapq.heappop(pq)
        
        # Skip if we already found a shorter path
        if current_dist > dist_dict[current_node]:
            continue
        
        # Check k-nearest neighbors
        for i in range(1, k+1):  # Skip the first as it's the point itself
            neighbor = neighbors[current_node][i]
            new_dist = current_dist + distances[current_node][i]
            if new_dist < dist_dict[neighbor]:
                dist_dict[neighbor] = new_dist
                heapq.heappush(pq, (new_dist, neighbor))
                edges.append((current_node, neighbor))
    
    return dist_dict, edges


def find_min_coord_point(points):
    """Finds the point with the smallest x, y, and z coordinates."""
    min_index = np.argmin(np.sum(points, axis=1))
    return min_index

def find_max_coord_point(points):
    """Finds the point with the largest x, y, and z coordinates."""
    min_index = np.argmax(np.sum(points, axis=1))
    return min_index


    
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

def datacleanup(points):
    counter = 0
    Trackerset = {}
    cleanupMap = {}
    cleanedupPoints = []
    for i,n in enumerate(points):
        n_ = tuple(n)
        if n_ in Trackerset:
            cleanupMap[i] = Trackerset[n_]
        else: 
            Trackerset[n_] = counter
            counter += 1
            cleanedupPoints.append(n)
            cleanupMap[i] = Trackerset[n_]

    #cleanupMap maps old index to new index
    return cleanedupPoints, cleanupMap, Trackerset
    
def graphCleanup(graph_edges, newMap):
    graph_edges_new = [(newMap[x[0]],newMap[x[1]]) for x in graph_edges]
    graph_edges_new_new = [x for x in graph_edges_new if x[0] != x[1]]
    return graph_edges_new_new


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


def contractionConnection(points):
    contracted_points = laplacian_contraction(points, k=20, sL=2)
    root_idx = find_min_coord_point(contracted_points)
    graph_edges = prim(contracted_points, root_idx)
    return contracted_points, graph_edges

def save_to_pcd(points, filename="output.pcd"):

    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)


    o3d.io.write_point_cloud(filename, point_cloud)
    print(f"Saved {len(points)} points to {filename}")



SUMedges = []
SUMvertices = []


def visualize_graph_points_overlay(points, edges, cloud):
    """Visualizes the 3D points and their k-NN edges using Open3D."""
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    
    
    lines = [[e[0], e[1]] for e in edges]
    colors = [[1, 0, 0] for _ in lines]  # Red edges
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
    render_option.point_size = 0.3
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

def makeGraphs(a,b,c,d):
    for i, quadrant in enumerate([a,b,c,d]):
        if len(quadrant) > 10000:
            print(len(quadrant))
            print("---")
            w, x,y,z = splitIntoQuadrants(quadrant, 5)
            makeGraphs(w,x,y,z)
        else:
            if quadrant.shape[0] == 0:  # Skip empty quadrants
                print(f"Skipping quadrant {i+1} (empty)")
                continue
            points, edges = contractionConnection(quadrant)

            if len(points) == 0 or len(edges) == 0:
                print(f"Skipping quadrant {i+1} (no points/edges after processing)")
                continue

            # point_cloud = o3d.geometry.PointCloud()
            # point_cloud.points = o3d.utility.Vector3dVector(points)


            shfiter = len(SUMvertices)
            
            lines = [[e[0]+shfiter, e[1]+shfiter] for e in edges]
            SUMvertices.extend(points)
            SUMedges.extend(lines)
            print("SUMVERT", len(SUMvertices))
            print("SUMEDGE", len(SUMedges))
            

cloud = o3d.io.read_point_cloud("C:/Users/samue/Downloads/Research/Spider/WebReconstruction/LargeWebConnectTest/@058 255 2024-12-05 04-26-14.pcd")


print(len(cloud.points))
a,b,c,d = splitIntoQuadrants(cloud.points, 15)

makeGraphs(a,b,c,d)

print(find_max_coord_point(SUMvertices))
print(find_min_coord_point(SUMvertices))
print(find_max_coord_point(cloud.points))
print(find_min_coord_point(cloud.points))



visualize_graph_points_overlay(SUMvertices, SUMedges, cloud)
