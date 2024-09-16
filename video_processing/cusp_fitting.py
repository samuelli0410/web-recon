import open3d as o3d
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.optimize import curve_fit

pcd = o3d.io.read_point_cloud('video_processing/point_clouds/@006r 255 2024-09-12 22-59-36.pcd')

def model_function(X, a, b, c, d, e):
    x, y = X
    return a / (b + np.sqrt((x - c)**2 + (y - d)**2)) + e

def extract_cylinder_subset(pcd, target_x, target_z, radius):
    # Convert point cloud to numpy array
    points = np.asarray(pcd.points)
    
    # Extract x, y, z from points
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]

    # Find the closest point in the (x, z) plane
    distances = np.sqrt((x - target_x)**2 + (z - target_z)**2)
    closest_index = np.argmin(distances)
    closest_point = points[closest_index]

    # Define the cylinder around the closest point
    # Since it's a cylinder, we consider points within the radius in the (x, z) plane
    in_cylinder_mask = (np.sqrt((x - closest_point[0])**2 + (z - closest_point[2])**2) <= radius)

    # Extract points within the cylinder
    subset_points = points[in_cylinder_mask]

    return subset_points

def extract_sphere_subset(pcd, target_x, target_y, target_z, radius):
    # Convert point cloud to numpy array
    points = np.asarray(pcd.points)
    
    # Extract x, y, z from points
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]

    # Find the closest point in the (x, z) plane
    distances = np.sqrt((x - target_x)**2 + (y - target_y)**2 + (z - target_z)**2)
    closest_index = np.argmin(distances)
    closest_point = points[closest_index]

    # Define the cylinder around the closest point
    # Since it's a cylinder, we consider points within the radius in the (x, z) plane
    in_sphere_mask = (np.sqrt((x - closest_point[0])**2 + (y - closest_point[1])**2 + (z - closest_point[2])**2) <= radius)

    # Extract points within the cylinder
    subset_points = points[in_sphere_mask]

    return subset_points


fig = plt.figure()

cusp_points = extract_sphere_subset(pcd, 512, 82, -748, 17)
sample_indices = np.random.choice(cusp_points.shape[0], size=int(cusp_points.shape[0]), replace=False)
sample_x = cusp_points[:, 0][sample_indices]
sample_y = cusp_points[:, 1][sample_indices]
sample_z = cusp_points[:, 2][sample_indices]

ax2 = fig.add_subplot(111, projection='3d')
ax2.scatter(sample_x, sample_z, sample_y, color='green')
ax2.scatter([512], [-748], [90], color='red')
plt.show() 

x_data = cusp_points[:, 0]
y_data = cusp_points[:, 2]
z_data = cusp_points[:, 1]
# Fit the model to the data
initial_guess = (1, 1, 1, 1, 1)  # Initial guesses for a, b, c, d, e
params_opt, params_cov = curve_fit(model_function, (x_data, y_data), z_data, p0=initial_guess)

# Extract fitted parameters
a_opt, b_opt, c_opt, d_opt, e_opt = params_opt
print(f"Fitted parameters: a = {a_opt}, b = {b_opt}, c = {c_opt}, d = {d_opt}, e = {e_opt}")

# Predict z values using the fitted model
z_pred = model_function((x_data, y_data), a_opt, b_opt, c_opt, d_opt, e_opt)

# Plot the original data and the fitted model
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x_data, y_data, z_data, color='blue', label='Data')
ax.scatter(x_data, y_data, z_pred, color='red', label='Fitted model', alpha=0.5)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.legend()
plt.show()

