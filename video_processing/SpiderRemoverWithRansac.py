from typing import List, Tuple
from scipy.interpolate import griddata
from sklearn.linear_model import RANSACRegressor

import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import open3d as o3d

import math

import statistics
import time
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

start = time.time()
print("Starting up")

#change these
cloud = o3d.io.read_point_cloud("C:/Users/samue/Downloads/Research/PCD Files/Good PCD's/@031r 255 2024-11-07 23-21-49 (with Spider).pcd")



#Tolerance is how aggressive you want the parts below RANSAC plane to be cut off --> lower means more cut off
tolerance = 6
tolerance2 = 3
# !009 255 2024-10-26 18-18-59 : 4


#neighborhood is the radius of the sphere around the spider you want to consider --> higher if shed skin, lower if not
neighborhood = 80


pcd_arr = np.asarray(cloud.points)


n = 1600.0
num_clusters = 100

#figuring out the smallest degree clustering in which we just isolate the spider
print("num_points_before: " + str(len(pcd_arr)))
while num_clusters > 2:
# Cluster
    n = n * 2
    model = DBSCAN(eps=10, min_samples=int(n))
    print("working")
    labels = model.fit_predict(pcd_arr)
    num_clusters = len(set(labels))
    print("num_clusters:" + str(num_clusters))
    if num_clusters == 1:
        num_clusters = 3
        n = (n - 0.25*(n))/2

print("num_clusters:" + str(num_clusters))
# Spider removal + locator
spider_arr = pcd_arr[labels != -1]
spider_x, spider_y, spider_z = np.mean(spider_arr[:, 0]), np.mean(spider_arr[:, 1]), np.mean(spider_arr[:, 2])
print(spider_x, spider_z)


# Segment PCD into where spider is and isn't
def distance(point):
    x, y, z = point
    return math.sqrt((x - spider_x)**2 + (y - spider_y)**2 + (z - spider_z)**2)

AreaSeg = []
spider_segment = []
web_segment = []
for point in pcd_arr:
    if distance(point) < neighborhood:
        spider_segment.append(point)
    elif distance(point) < neighborhood*2:
        AreaSeg.append(point)
    else: 
        web_segment.append(point)
spider_segmentOriginal = np.array(spider_segment)


#remove spider features
def removespider(points: List[Tuple[int, int, float]], outputintz: List[float], threshold: int = 7):
    
    points.sort(key=lambda p: (p[2], p[0], p[1])) 

    prevpointx = prevpointy = prevpointz = None
    currentrowrunning = 0
    removecolumn = []

    def addmissedpoints(x: int, y: int, z: float, removed: List[Tuple[int, int, float]]):
        if removed[-1] != (x,y-1,z):
            for i in range(threshold-1):
                removed.append((x,y-1-i,z))


    for point in points:
        x, y, z = point
        
        
        # Check if still in the same column and z-layer
        if z == prevpointz and x == prevpointx and prevpointy is not None:
            # Check for consecutive y-values
            if y == prevpointy + 1:
                currentrowrunning += 1
                if currentrowrunning >= threshold:
                    removecolumn.append((x, y, z))
                    addmissedpoints(x,y,z, removecolumn)
            else:
                currentrowrunning = 1  # Reset consecutive row counter
        else:
            currentrowrunning = 1  # Reset when column or z changes
        
        # Update previous points
        prevpointx, prevpointy, prevpointz = x, y, z

    # Remove columns that pass the threshold
    result = removesomecolumn(points, removecolumn)
    # Compute output intervals based on removed points
    outputinterval(removecolumn, outputintz)

    return result



def removesomecolumn(points: List[Tuple[int, int, float]], removed: set) -> List[Tuple[int, int, float]]:
    # Ensure removed set contains tuples, not ndarrays
    removed_set = {tuple(point) for point in removed}
    # Filter points by checking tuple conversion
    return [point for point in points if tuple(point) not in removed_set]


def outputinterval(removed: List[Tuple[int, int, float]], outputintz: List[float]):
    if not removed:  # Early exit if no points were removed
        return

    xlist = [a for a, _, _ in removed]
    zlist = [c for _, _, c in removed]

    xq1 = np.percentile(xlist, 25)
    xq3 = np.percentile(xlist, 90)
    zq1 = np.percentile(zlist, 25)
    zq3 = np.percentile(zlist, 90)

    xgap = abs(xq3 - xq1)
    zgap = abs(zq3 - zq1)

    # Update outputintz with calculated intervals
    outputintz[0] = xq1 
    outputintz[1] = xq3 + xgap/2
    outputintz[2] = zq1 
    outputintz[3] = zq3 + zgap/2


outputintervalz = [0.0,0.0,0.0,0.0]
spider_segment = removespider(spider_segment, outputintervalz, 7)
# end of remvoing the spider body


web_segment = np.array(web_segment)
AreaSeg = np.array(AreaSeg)
spider_segment = np.array(spider_segment)


#consolidate all the parts into their highest verticality point
#   this is done because the legs are much more dense than the web vertically but not horizontally, so this way we can consolidate them and allow RANSAC to treat them as outliers, finding us a fitted plane
def skimtop(points: np.ndarray):
    points = points[np.argsort(points[:, 2], kind='quicksort')]  # Sort by z, then x, then -y
    
    # Initialize previous values to track duplicates
    prevpointx = prevpointy = prevpointz = None
    skimmedtop = []

    for point in points:
        x, y, z = point

        # Check if the point is a new unique point based on z and x, or if it's a None value
        if (z != prevpointz or x != prevpointx) or (z is None and x is None):
            skimmedtop.append(point)

        prevpointx = x
        prevpointz = z
    
    return np.array(skimmedtop)


ransac = RANSACRegressor()

def RANSACCleanse(spider_segment: np.ndarray, tolerance):
    spider_segment_top = skimtop(spider_segment)
    #RANSAC fitting
    ransac.fit(spider_segment_top[:, [0, 2]], spider_segment_top[:, 1])

    #remove unwanted parts of the spider below RANSAC Plane
    clean_spider_segment = []
    print("ransac cleansing")
    for point in spider_segment: 
        x, y, z = point
        if y + tolerance >= ransac.predict([[x, z]]) and y - tolerance <= ransac.predict([[x, z]]):
            clean_spider_segment.append(point)
    print("ransac cleansed")
    return np.array(clean_spider_segment)
    

spider_segment_top = skimtop(spider_segment)
    #RANSAC fitting
ransac.fit(spider_segment_top[:, [0, 2]], spider_segment_top[:, 1])
ransac_plane_points = []
x_range = np.linspace(np.min(AreaSeg[:, 0]), np.max(AreaSeg[:, 0]), num=50)  # 100 points between min and max x
z_range = np.linspace(np.min(AreaSeg[:, 2]), np.max(AreaSeg[:, 2]), num=50)

#   Iterate over the grid of x and z values
for x in x_range:
    for z in z_range:
        # Predict the corresponding y value using RANSAC
        y_pred = ransac.predict([[x, z]])  # Predict requires a 2D array
        # Append the (x, y_pred, z) tuple to the list
        ransac_plane_points.append((x, y_pred[0], z))

ransac_plane_points_array = np.array(ransac_plane_points)

    

# spider_segment_combined = np.vstack((spider_segment, web_segment))

spider_segment1 = RANSACCleanse(spider_segment, tolerance)





def skimbottom(points: np.ndarray):
    # Sort points by z, x, y values
    points_sorted = points[points[:, 2].argsort(kind='mergesort')]  # Sort by z first, then x, then y
    
    
    # Initialize previous point variables
    prevpointx = prevpointy = prevpointz = None
    skimmedtop = []

    for point in points:
        x, y, z = point

        # Check if the point is a new unique point based on z and x, or if it's a None value
        if (z != prevpointz or x != prevpointx) or (z is None and x is None):
            skimmedtop.append(point)

        prevpointx = x
        prevpointz = z
    
    return np.array(skimmedtop)


def linRegCleanse(spider_segment: np.ndarray, spider_segment_Original: np.ndarray, tolerance):
    spider_bottom = skimbottom(spider_segment)
    x = spider_bottom[:, 0]
    y = spider_bottom[:, 1]
    z = spider_bottom[:, 2]
    X = np.vstack((x, z)).T
    print("Polynomial Fitting")
    # Fit a polynomial regression model to predict y
    degree = 2
    poly_features = PolynomialFeatures(degree)
    X_poly = poly_features.fit_transform(X)
    # Fit the linear regression model
    model = LinearRegression()
    model.fit(X_poly, y)

    clean_spider_segment = []
    for point in spider_segment_Original: 
        x, y, z = point
        X_arbitrary = np.array([[x, z]])
        X_arbitrary_poly = poly_features.transform(X_arbitrary)
        y_pred_arbitrary = model.predict(X_arbitrary_poly)
        if y + tolerance >= y_pred_arbitrary[0]:
            clean_spider_segment.append(point)
    print("polynomial fitted")
    return np.array(clean_spider_segment)
    
    

spider_bottom = skimbottom(spider_segment1)
x = spider_bottom[:, 0]
y = spider_bottom[:, 1]
z = spider_bottom[:, 2]
X = np.vstack((x, z)).T
    # Fit a polynomial regression model to predict y
degree = 2
poly_features = PolynomialFeatures(degree)
X_poly = poly_features.fit_transform(X)
    # Fit the linear regression model
model = LinearRegression()
model.fit(X_poly, y)


polynomialPred = []



for x in x_range:
    for z in z_range:
        # Predict the corresponding y value using RANSAC
        X_arbitrary = np.array([[x, z]])  # Predict requires a 2D array
        # Append the (x, y_pred, z) tuple to the list
        X_arbitrary_poly = poly_features.transform(X_arbitrary)
        y_pred_arbitrary = model.predict(X_arbitrary_poly)
        polynomialPred.append((x, y_pred_arbitrary[0], z))


spider_segment = linRegCleanse(spider_segment1, spider_segmentOriginal, tolerance2)


print(len(spider_segment1))

print(len(spider_bottom))


spider_segment_combined = np.vstack((spider_segment, web_segment, AreaSeg))





pcd = o3d.geometry.PointCloud()

try:
    print(pcd.points)
    pcd.points = o3d.utility.Vector3dVector(spider_segment_combined)


    o3d.io.write_point_cloud("C:/Users/samue/Downloads/Research/SpiderRemoved.pcd", pcd)
    print("done")
except Exception as e:
    print(e)