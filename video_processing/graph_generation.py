import numpy as np
import open3d as o3d
import networkx as nx
import matplotlib.pyplot as plt


pcd = o3d.io.read_point_cloud('test_web.pcd')

# Assuming pcd is your Open3D point cloud object loaded from earlier
pcd.estimate_normals()

# Poisson surface reconstruction
mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9)

# Optionally remove low density vertices
vertices_to_remove = densities < np.quantile(densities, 0.01)
mesh.remove_vertices_by_mask(vertices_to_remove)

# Visualize the mesh
o3d.visualization.draw_geometries([mesh])