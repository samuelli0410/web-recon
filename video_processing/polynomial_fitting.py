import open3d as o3d
import numpy as np
from scipy.interpolate import Rbf
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Load the point cloud
pcd = o3d.io.read_point_cloud('video_processing/point_clouds/@005 255 2024-09-06 18-17-24.pcd')

# # Convert point cloud to numpy array and extract points
# points = np.asarray(pcd.points)

# # Assuming the points are organized as (x, y, z)
# x = points[:, 0]
# y = points[:, 1]
# z = points[:, 2]

# # Combine x and z to form the input features for polynomial fitting
# X = np.vstack((x, z)).T

# # Fit a polynomial regression model to predict y
# degree = 4
# poly_features = PolynomialFeatures(degree)
# X_poly = poly_features.fit_transform(X)

# # Fit the linear regression model
# model = LinearRegression()
# model.fit(X_poly, y)

# # Predict y using the fitted model
# y_pred = model.predict(X_poly)

# # Display coefficients
# print("Coefficients:", model.coef_)
# print("Intercept:", model.intercept_)
# # Randomly sample 10% of the points for visualization
# sample_indices = np.random.choice(points.shape[0], size=int(points.shape[0] * 0.1), replace=False)
# sampled_x = x[sample_indices]
# sampled_y = y[sample_indices]
# sampled_z = z[sample_indices]
# sampled_y_pred = y_pred[sample_indices]

# # Visualize the original y and predicted y for the sampled points
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# #ax.scatter(sampled_x, sampled_z, sampled_y, color='red', label='Original y')
# ax.scatter(sampled_x, sampled_z, sampled_y_pred, color='blue', label='Predicted y', alpha=0.5)
# ax.set_xlabel('X')
# ax.set_ylabel('Z')
# ax.set_zlabel('Y')
# plt.title('Polynomial Surface Fit (Sampled Points)')
# plt.legend()
# plt.show()

# Preprocessing: Remove statistical outliers to clean the point cloud
cl, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
pcd = pcd.select_by_index(ind)

# Optional: Downsample the point cloud to reduce complexity
pcd = pcd.voxel_down_sample(voxel_size=0.001)

# Estimate normals of the cleaned point cloud
pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=30))

# Enhance the normal orientation to align with sharp features
pcd.orient_normals_consistent_tangent_plane(k=30)

# Apply Screened Poisson Surface Reconstruction with a higher depth for better detail
mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
    pcd, depth=12, width=0, scale=1.1, linear_fit=False
)

# Convert densities to a numpy array for filtering
densities = np.asarray(densities)

# Filter out low-density vertices to clean up the mesh
vertices_to_remove = densities < np.quantile(densities, 0.01)
mesh.remove_vertices_by_mask(vertices_to_remove)

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

# Save the mesh to a file
o3d.io.write_triangle_mesh("reconstructed_spider_web.ply", mesh)

# Create a visualizer object for headless rendering
vis = o3d.visualization.Visualizer()
vis.create_window(visible=False)  # Run in headless mode
vis.add_geometry(mesh)
vis.poll_events()
vis.update_renderer()
vis.capture_screen_image("reconstructed_spider_web.png")
vis.destroy_window()

print("Surface reconstruction completed and saved to 'reconstructed_spider_web.ply' and 'reconstructed_spider_web.png'.")