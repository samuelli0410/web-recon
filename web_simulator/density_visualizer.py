# import open3d as o3d
# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.colors import Normalize
# from mpl_toolkits.mplot3d import Axes3D

# # Step 1: Load the Point Cloud Data (PCD)
# def load_pcd(file_path):
#     print("Loading the point cloud...")
#     pcd = o3d.io.read_point_cloud(file_path)
#     points = np.asarray(pcd.points)
#     print(f"Loaded point cloud with {len(points)} points.")
#     return points

# # Step 2: Subdivide into 8000 regions (20x20x20) and count points in each region
# def subdivide_and_count(points):
#     print("Subdividing space into 8000 rectangular regions (20x20x20)...")
    
#     # Get the bounds of the point cloud
#     min_bound = points.min(axis=0)
#     max_bound = points.max(axis=0)
    
#     # Compute the width of each bin (subdivision) along each dimension
#     bin_size = (max_bound - min_bound) / num_subdivisions  # 20 segments per dimension
    
#     # Initialize a 3D array to count points in each of the 8000 subregions
#     count_grid = np.zeros((num_subdivisions, num_subdivisions, num_subdivisions))
    
#     # For each point, determine which subregion (bin) it belongs to
#     for point in points:
#         indices = np.floor((point - min_bound) / bin_size).astype(int)
#         indices = np.clip(indices, 0, num_subdivisions - 1)  # Ensure indices stay within bounds
#         count_grid[tuple(indices)] += 1
    
#     total_points = len(points)
#     print("Finished counting points in each subregion.")
#     return count_grid, total_points

# # Step 3: Calculate and normalize density levels
# def calculate_density(count_grid, total_points):
#     print("Calculating density levels...")
    
#     # Normalize the counts by the total number of points to get the density in each block
#     density_grid = count_grid / total_points  # This gives density as a fraction of total points
    
#     print("Density calculation completed.")
#     return density_grid

# # # Step 4: Visualize Density Distribution in 3D
# # def visualize_density(density_grid):
# #     print("Visualizing density distribution...")
    
# #     # Generate x, y, z coordinates for each block and corresponding density
# #     x, y, z = np.indices(density_grid.shape).reshape(3, -1)
# #     densities = density_grid.flatten()
    
# #     # Filter out blocks with zero density for visualization
# #     mask = densities > 0
# #     x, y, z, densities = x[mask], y[mask], z[mask], densities[mask]
    
# #     # Normalize densities for color mapping
# #     norm = Normalize(vmin=densities.min(), vmax=densities.max())
    
# #     # Plotting the density distribution in 3D with darker colors for higher densities
# #     fig = plt.figure(figsize=(10, 8))
# #     ax = fig.add_subplot(111, projection='3d')
    
# #     # Use a scatter plot to represent density with color
# #     scatter = ax.scatter(x, y, z, c=densities, cmap='Greys', norm=norm, s=40)
    
# #     # Add color bar to show density scale
# #     cbar = fig.colorbar(scatter, ax=ax, shrink=1.0, aspect=5)
# #     cbar.set_label('Silk Density')
    
# #     # Label axes and show plot
# #     ax.set_xlabel("X axis")
# #     ax.set_ylabel("Y axis")
# #     ax.set_zlabel("Z axis")
# #     plt.title("Silk Density Distribution in 3D Space")
# #     plt.show()
# def visualize_density(density_grid):
#     print("Visualizing density distribution...")
    
#     # Generate x, y, z coordinates for each block and corresponding density
#     x, y, z = np.indices(density_grid.shape).reshape(3, -1)
#     densities = density_grid.flatten()
    
#     # Filter out blocks with zero density for visualization
#     mask = densities > 0
#     x, y, z, densities = x[mask], y[mask], z[mask], densities[mask]
    
#     # Normalize densities for color mapping
#     norm = Normalize(vmin=densities.min(), vmax=densities.max())
    
#     # Plotting the density distribution in 3D with a reversed high-contrast colormap
#     fig = plt.figure(figsize=(10, 8))
#     ax = fig.add_subplot(111, projection='3d')
    
#     # Use a scatter plot with the reversed colormap 'plasma_r'
#     scatter = ax.scatter(x, y, z, c=densities, cmap='plasma_r', norm=norm, s=40)
    
#     # Add color bar to show density scale
#     cbar = fig.colorbar(scatter, ax=ax, shrink=1.0, aspect=5)
#     cbar.set_label('Silk Density')
    
#     # Label axes and show plot
#     ax.set_xlabel("X axis")
#     ax.set_ylabel("Y axis")
#     ax.set_zlabel("Z axis")
#     plt.title("Silk Density Distribution")
#     plt.show()


# # Main function to execute the steps and visualize density distribution
# if __name__ == "__main__":
#     # Path to your point cloud file (PCD)
#     file_path = "video_processing/point_clouds/@044 255 2024-11-19 03-16-53.pcd"
#     num_subdivisions = 100
#     # Step 1: Load the PCD file
#     points = load_pcd(file_path)
    
#     # Step 2: Subdivide space and count points in each subregion
#     count_grid, total_points = subdivide_and_count(points)
    
#     # Step 3: Calculate density levels based on point counts
#     density_grid = calculate_density(count_grid, total_points)
    
#     # Step 4: Visualize density distribution
#     visualize_density(density_grid)


# import open3d as o3d
# import numpy as np
# import matplotlib.pyplot as plt

# # Parameters
# pcd_file = "video_processing/point_clouds/@044 255 2024-11-19 03-16-53.pcd"  # Path to your PCD file
# bins = 100  # Number of bins in each dimension for the histogram

# # Load the point cloud
# pcd = o3d.io.read_point_cloud(pcd_file)
# points = np.asarray(pcd.points)

# # Extract x, y, z coordinates
# x = points[:, 0]
# y = points[:, 1]
# z = points[:, 2]

# # For a top-down view, we consider the X-Y plane.
# # That is, we ignore Z and just look at how points distribute in the plane.
# x_min, x_max = x.min(), x.max()
# y_min, y_max = y.min(), y.max()

# # Compute 2D histogram of points on X-Y plane
# H, xedges, yedges = np.histogram2d(y, x, bins=bins, range=[[y_min, y_max], [x_min, x_max]])

# plt.figure(figsize=(8, 6))
# plt.imshow(H, origin='lower', extent=[x_min, x_max, y_min, y_max],
#            cmap='hot', interpolation='nearest', aspect='auto')
# plt.colorbar(label='Point Count')
# plt.title("Top-Down View of Point Density (X-Y Plane)")
# plt.xlabel("X")
# plt.ylabel("Y")
# plt.tight_layout()
# plt.show()

# import open3d as o3d
# import numpy as np
# import matplotlib.pyplot as plt

# # Parameters
# pcd_file = "video_processing/point_clouds/@044 255 2024-11-19 03-16-53.pcd"  # Path to your PCD file
# bins = 100  # Number of bins in each dimension for the histogram

# # Load the point cloud
# pcd = o3d.io.read_point_cloud(pcd_file)
# points = np.asarray(pcd.points)

# # Extract x, y, z coordinates
# x = points[:, 0]
# y = points[:, 1]
# z = points[:, 2]

# # For a top-down view, we consider the X-Y plane.
# # That is, we ignore Z and just look at how points distribute in the plane.
# x_min, x_max = x.min(), x.max()
# z_min, z_max = z.min(), z.max()

# # Compute 2D histogram of points on X-Y plane
# H, xedges, yedges = np.histogram2d(z, x, bins=bins, range=[[z_min, z_max], [x_min, x_max]])



# plt.figure(figsize=(8, 6))
# plt.imshow(H, origin='lower', extent=[x_min, x_max, z_min, z_max],
#            cmap='hot', interpolation='nearest', aspect='auto')
# plt.colorbar(label='Silk Density')
# plt.title("Top-Down View of Silk Density")
# # plt.xlabel("X")
# # plt.ylabel("Y")
# plt.tight_layout()
# plt.show()

import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt

# Parameters
pcd_file = "video_processing/point_clouds/@064 255 2024-12-10 17-49-41.pcd"  # Path to your PCD file
bins = 100  # Number of bins in each dimension for the histogram

# Load the point cloud
pcd = o3d.io.read_point_cloud(pcd_file)
points = np.asarray(pcd.points)

# Extract x, y, z coordinates
x = points[:, 0]
y = points[:, 1]
z = points[:, 2]

# For a top-down view, we consider (X-Z) plane as shown in your code
x_min, x_max = x.min(), x.max()
z_min, z_max = z.min(), z.max()

# Compute 2D histogram (using Z, X for consistency with your given code)
H, xedges, yedges = np.histogram2d(z, x, bins=bins, range=[[z_min, z_max], [x_min, x_max]])

plt.figure(figsize=(8, 6))
img = plt.imshow(H, origin='lower', extent=[x_min, x_max, z_min, z_max],
                 cmap='hot', interpolation='nearest', aspect='auto')

# Remove all axis ticks and labels
plt.xticks([])
plt.yticks([])
plt.gca().spines["top"].set_visible(False)
plt.gca().spines["right"].set_visible(False)
plt.gca().spines["left"].set_visible(False)
plt.gca().spines["bottom"].set_visible(False)


# Add the colorbar, but remove numeric tick labels
cbar = plt.colorbar(img)
cbar.set_label("Density", fontsize=12)   # A label to indicate the meaning of color
cbar.ax.set_yticks([])                   # Remove numeric ticks
cbar.ax.set_yticklabels([])              # Ensure no numeric labels are shown

plt.title("Top-Down View of Silk Density", fontsize=12)
plt.tight_layout()
plt.savefig("High Entropy Heat Map.png", dpi=300, bbox_inches='tight')
plt.show()