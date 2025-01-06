# Import libraries
import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import open3d as o3d
from sklearn.cluster import DBSCAN
from matplotlib import colormaps
from matplotlib.widgets import CheckButtons
from scipy.spatial import KDTree
import random
from scipy.spatial import distance
import time
import ipywidgets as widgets
from IPython.display import display


start = time.time()
fig =  plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')







def SafeKDTree(input_array):
    def remove_nan_inf(array):
        finite_mask = np.isfinite(array).all(axis=1)
        cleaned_array = array[finite_mask]
        return cleaned_array
    cleaned_array = remove_nan_inf(input_array)
    return KDTree(cleaned_array)



#Skeletonizes the code via rudimentary nonexpansive clustering
def compactize(input_array, radius):
    print("Starting to compactize")
    # Create a KDTree for efficient neighbor search
    if len(input_array) < 2:
        return
    tree = SafeKDTree(input_array)
    
    # Track indices to remove
    rows_to_remove = set()
    linepoints = []
    

    for idx, point in enumerate(input_array):
        if idx in rows_to_remove:
            continue 
        

        neighbors_idx = tree.query_ball_point(point, radius)
        

        neighbors_idx = [n_idx for n_idx in neighbors_idx]

        neighbors_points = input_array[neighbors_idx]
        
        # If there are enough neighbors (at least 1), perform the calculations
        if len(neighbors_points) >= 1:
            # Mark neighbors for removal
            rows_to_remove.update(neighbors_idx)
            
            # Compute the mean of the neighboring points
            datamean = np.mean(neighbors_points, axis=0)

            linepoints.append(datamean)

    linepoints = np.array(linepoints)
    mask = np.array([i not in rows_to_remove for i in range(len(input_array))])
    filtered_array = np.concatenate((input_array[mask], linepoints), axis=0)
    # ax.scatter(filtered_array[:, 0], filtered_array[:, 1], filtered_array[:, 2], s=10)
    print("compactized.")
    return filtered_array



#Skeletonizes the code via Rodel Niblle technique (very slow)
def rodelNibble(input_array, radius):
    tree = SafeKDTree(np.empty((0, 3)))
    result = np.empty((0,3))
    while (len(input_array) > 0 and input_array is not np.empty):
        index = random.randint(0, len(input_array)-1)
        element = input_array[index]
        np.delete(input_array, index, 0)
        print(time.time() - start)
        if len(tree.query_ball_point(element, radius)) == 0:
            np.append(result, element)
            tree = SafeKDTree(result)
    return result






class WeightedQuickUnionWithPathCompressionUF():
    def __init__(self, n):
        self.items = list(range(n))
        self.sizes = [1] * n
        self.count = len(self.items)

    def __repr__(self):
        return "{'roots': %s, 'count': %d}" % (self.items, self.count)
    def union(self, p, q):
        if p == q:
            return
        p_root = self._root(p)
        q_root = self._root(q)
        if p_root == q_root:
            return

        if self.sizes[p_root] < self.sizes[q_root]:
            self.items[p_root] = q_root
            self.sizes[q_root] += self.sizes[p_root]
        else:
            self.items[q_root] = p_root
            self.sizes[p_root] += self.sizes[q_root]

        self.count -= 1

    def _root(self, p):
        root = self.items[p]
        while not root == self.items[root]:
            self.items[root] = self.items[self.items[root]]
            root = self.items[root]
        self.items[p] = root
        return root

    def find(self, p):
        return self._root(p)

    def connected(self, p, q):
        return p == q or self.find(p) == self.find(q)

    def count(self):
        return self.count

    def size_of_union(self, p):
    # Find the root of the component containing `p`
        root = self._root(p)
    # Return the size of the union (connected component)
        return self.sizes[root]

class Graph:
    def __init__(self, input):
        self.size = len(input)
        self.vertex_data = [''] * self.size
        self.tree = SafeKDTree(input)
        self.WQU = WeightedQuickUnionWithPathCompressionUF(self.size)
        for i in range(self.size):
            self.add_vertex_data(i, input[i])

    def add_vertex_data(self, vertex, data):
        if 0 <= vertex < self.size:
            self.vertex_data[vertex] = data
            

    def connectLines(self):
        return 0



        

def ConnectSegment(pcd_arr, eps):
    print("connecting Segments!")

    #scales down to decentivize prioritzation in either direction
    x_max = (pcd_arr[:, 0].max())
    y_max = (pcd_arr[:, 1].max())
    z_max = pcd_arr[:, 2].max()

    xScale = x_max / 500 if x_max != 0 else 1
    yScale = y_max / 500 if y_max != 0 else 1
    zScale = z_max / 500 if z_max != 0 else 1

    pcd_arr[:, 0] /= xScale
    pcd_arr[:, 1] /= yScale
    pcd_arr[:, 2] /= zScale

    compact = compactize(pcd_arr, eps//8)
    sorted_indices = np.lexsort((compact[:, 2], compact[:, 1], compact[:, 0])) 
    compact = compact[sorted_indices]

    tree = SafeKDTree(compact)

            
    def connectLines(input_array, tree): 
        WQU = WeightedQuickUnionWithPathCompressionUF(len(input_array))

        #calculates the heuristic to force prioritization on more in-line points
        def calcAngle(point1, point2, point3):
            prev_vector = point2 - point1
            curr_vector = point3 - point2
            
            # Calculate norms
            prev_norm = np.linalg.norm(prev_vector)
            curr_norm = np.linalg.norm(curr_vector)
            
            # Handle zero-length vectors
            if prev_norm == 0 or curr_norm == 0:
                print("Warning: One of the vectors has zero length.")
                return 0  # Define a fallback heuristic value
            
            # Compute cosine of the angle
            cosine_angle = np.dot(prev_vector, curr_vector) / (prev_norm * curr_norm)
            
            # Clamp value to [-1, 1] to avoid domain error
            cosine_angle = max(-1, min(1, cosine_angle))
            
            # Compute angle
            prev_angle = math.acos(cosine_angle)
            return prev_angle

        def calcHeuristic(point1, point2, point3, angle, weight):
            
            prev_angle = calcAngle(point1, point2, point3)
            
            # Calculate heuristic
            if prev_angle <= angle :
                return (1 + math.sin(prev_angle)) * weight
            else:
                return 100000

        #recursively looks for closest point by distance + heuristic that it is not already connected to, within range of radius
        #tracks the start of journey as well as end of journeys
        def connectionHelper(query_point_idx, previous_point_idx, count, tracked, starts, ends, radius):
            print(time.time()-start)

            if query_point_idx in tracked:
                return
            
            query_point = input_array[query_point_idx]
            
            tracked.append(query_point_idx)

            

            previous_point = input_array[previous_point_idx]
            nearestPoints = (tree.query_ball_point(query_point,radius))
            nearestPoints.remove(query_point_idx)
            
            if query_point_idx == previous_point_idx:
                distances = [distance.euclidean(query_point, input_array[i]) for i in nearestPoints]
            else: 
                distances = [distance.euclidean(query_point, input_array[i]) + calcHeuristic(previous_point, query_point, input_array[i], math.pi//2, eps//3) for i in nearestPoints]
            
            if len(distances) > 0:
                dist = min(distances)
                if dist > 100000:
                    if previous_point_idx != query_point_idx:
                        ends.append([previous_point_idx, query_point_idx])
                    return
            else: 
                if previous_point_idx != query_point_idx:
                        ends.append([previous_point_idx, query_point_idx])
                return


        
            closest = nearestPoints[distances.index(dist)]
            while WQU.connected(query_point_idx, closest):
                if len(nearestPoints) == 1:
                    return
                nearestPoints.remove(closest)
                distances.remove(min(distances))
                closest = nearestPoints[distances.index(min(distances))]



            WQU.union(closest, query_point_idx)
            nearest_point = input_array[closest]
            linepts = np.vstack([query_point, nearest_point])  # Stack the points vertically for plotting
            linepts[:, 0] *= xScale
            linepts[:, 1] *= yScale
            linepts[:, 2] *= zScale
            ax.plot(linepts[:, 0], linepts[:, 1], linepts[:, 2], color='black')

            if calcAngle(previous_point, query_point, nearest_point) > math.pi/4:
                print()
                ends.append([query_point_idx, previous_point_idx])

            if query_point_idx == previous_point_idx and closest != query_point_idx:
                starts.append(query_point_idx)

            connectionHelper(closest, query_point_idx, count + 1, tracked, starts, ends, radius)
            if closest in starts and query_point_idx in starts:
                starts.remove(query_point_idx)
                ends.append([closest, query_point_idx])
            
        #connects ends to start by looking for nearest point by distance + heuristic, but with much stricter weighing on being in-line and stricter angle tolerances.
        def connectEndsToStarts(starts, ends, radius):
            print("Connecting Holes")

            starts = np.array([input_array[x] for x in starts])
            starts = np.array(starts)
            if len(starts) == 0:
                return
            tree = SafeKDTree(starts)
            counted = set()

            for end in ends:
                killer = 0
                endidx = end[1]
                previdx = end[0]
                prevpoint = input_array[previdx]
                if endidx in counted:
                    continue
                counted.add(endidx)
                endpoint = input_array[endidx]
                nearestPoints = (tree.query_ball_point(endpoint,radius))
                distances = [distance.euclidean(endpoint, starts[i]) + calcHeuristic(prevpoint, endpoint, starts[i], math.pi//2, eps//1.5) for i in nearestPoints]
                if len(distances) > 0:
                    dist = min(distances)
                else: 
                    continue

                closest = nearestPoints[distances.index(dist)]
                nearest_point = starts[closest]
                closest1 = np.where((input_array == nearest_point).all(axis=1))[0][0]
                
                while WQU.connected(endidx, closest1):
                    if len(nearestPoints) == 1:
                        killer = 1
                        break
                    nearestPoints.remove(closest)
                    distances.remove(min(distances))
                    closest = nearestPoints[distances.index(min(distances))]
                    nearest_point = starts[closest]
                    closest1 = np.where((input_array == nearest_point).all(axis=1))[0][0]
                if killer == 1:
                    continue
                
                
                linepts = np.vstack([endpoint, nearest_point])  # Stack the points vertically for plotting
                WQU.union(closest1, endidx)
                linepts[:, 0] *= xScale
                linepts[:, 1] *= yScale
                linepts[:, 2] *= zScale
                ax.plot(linepts[:, 0], linepts[:, 1], linepts[:, 2], color='lightslategrey')
                





        tracked = []
        starts = []
        ends = []
        for i in range(len(input_array)):
            connectionHelper(i,i,0,tracked,starts, ends, eps//2)
        connectEndsToStarts(starts, ends, eps)

    connectLines(compact, tree)

    pcd_arr[:, 0] *= xScale
    pcd_arr[:, 1] *= yScale
    pcd_arr[:, 2] *= zScale



# Load point cloud
def segmenterTrial(EPS, pointcloud):
    cloud = o3d.io.read_point_cloud(pointcloud)
    pcd_arr = np.asarray(cloud.points)
    x_max = (pcd_arr[:, 0].max())
    y_max = (pcd_arr[:, 1].max())
    z_max = pcd_arr[:, 2].max()

    xScale = x_max / 500 if x_max != 0 else 1
    yScale = y_max / 500 if y_max != 0 else 1
    zScale = z_max / 500 if z_max != 0 else 1

    # Cluster using DBSCAN
    print("Start Clustering")
    model = DBSCAN(eps=EPS, min_samples=10).fit(pcd_arr)
    print("Done Clustering")
    labels = model.labels_



    # Connect individual segments
    unique_labels = list(set(labels))
    pcd_arr[:, 0] *= xScale
    pcd_arr[:, 1] *= yScale
    pcd_arr[:, 2] *= zScale

    for label in unique_labels:
        print(f"{label}/{len(unique_labels)}")
        # Points belonging to this cluster
        if label == -1:
            continue
        cluster_points = pcd_arr[labels == label]
        
        # ax.scatter(cluster_points[:, 0], cluster_points[:, 1], cluster_points[:, 2], s=0.1)
        
        ConnectSegment(cluster_points, EPS)

    return pcd_arr

    



pcd_arr = segmenterTrial(120,  "C:/Users/samue/Downloads/Research/PCD Files/TestingPCD/sparse3 255 2024-11-30 11-29-33.pcd")






ax.set_xlabel("X-axis")
ax.set_ylabel("Y-axis")
ax.set_zlabel("Z-axis")
plt.show()
