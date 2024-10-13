import open3d as o3d
import numpy as np
from scipy.interpolate import Rbf
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


# Load the point cloud
pcd = o3d.io.read_point_cloud('video_processing/point_clouds/@006r 255 2024-09-12 22-59-36.pcd')

# Convert point cloud to numpy array and extract points
points = np.asarray(pcd.points)

# Simplify mesh to improve quality and make it more watertight
mesh.remove_duplicated_vertices()
mesh.remove_duplicated_triangles()
mesh.remove_degenerate_triangles()
mesh.remove_non_manifold_edges()

# Apply Laplacian smoothing to reduce noise while preserving sharp features
mesh = mesh.filter_smooth_laplacian(number_of_iterations=5)

# Recompute normals after smoothing
mesh.compute_vertex_normals()

# Set the mesh color to blue
mesh.paint_uniform_color([0, 0, 1])  # RGB color for blue

# Predict y using the fitted model
y_pred = model.predict(X_poly)
residuals = np.abs(y - y_pred)

# Threshold for error
error_threshold = 50  # Adjust this threshold as needed

# Identify points with high error
high_error_indices = residuals > error_threshold
# Display coefficients
print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)
# Randomly sample 10% of the points for visualization
sample_indices = np.random.choice(points.shape[0], size=int(points.shape[0] * 0.003), replace=False)
sampled_x = x[sample_indices]
sampled_y = y[sample_indices]
sampled_z = z[sample_indices]
sampled_y_pred = y_pred[sample_indices]

# Visualize the original y and predicted y for the sampled points
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(sampled_x, sampled_z, sampled_y, color='red', label='Original y')
#ax.scatter(sampled_x, sampled_z, sampled_y_pred, color='blue', label='Predicted y', alpha=0.5)
# Plot high-error points
# high_error_x = x[high_error_indices]
# high_error_y = y[high_error_indices]
# high_error_z = z[high_error_indices]
# ax.scatter(high_error_x, high_error_z, high_error_y, color='green', label='High Error Points', marker='^')
ax.set_xlabel('X')
ax.set_ylabel('Z')
ax.set_zlabel('Y')
plt.title('Polynomial Surface Fit (Sampled Points)')
plt.legend()
plt.show()



