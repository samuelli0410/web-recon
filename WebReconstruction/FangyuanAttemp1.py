#!/usr/bin/env python3
"""
Convert a noisy spider-web PCD into a 3D graph and visualize it.

– Edit INPUT_PCD to point at your .pcd file.
– Tweak the parameters below as needed.
"""

import numpy as np
import open3d as o3d
from skimage.morphology import skeletonize, binary_dilation, binary_erosion, disk
from skan import csr
import networkx as nx

# ─────────── USER SETTINGS ───────────
INPUT_PCD       = "video_processing/point_clouds/sparse3 255 2024-11-30 11-29-33.pcd"
# OUTPUT_GRAPH    = "web_graph.gpickle"   # uncomment to save
VOXEL_SIZE      = 0.5    # down-sample voxel size
NB_NEIGHBORS    = 20     # for outlier removal
STD_RATIO       = 2.0
RES             = 0.5    # world units per pixel
DISK_RADIUS     = 1      # closing radius
MIN_EDGE_LENGTH = 5.0    # prune edges shorter than this
# ──────────────────────────────────────

def load_and_denoise(path):
    pcd = o3d.io.read_point_cloud(path)
    pcd = pcd.voxel_down_sample(voxel_size=VOXEL_SIZE)
    pcd, _ = pcd.remove_statistical_outlier(
        nb_neighbors=NB_NEIGHBORS, std_ratio=STD_RATIO)
    return pcd

def project_to_plane(pcd):
    pts = np.asarray(pcd.points)
    centroid = pts.mean(axis=0)
    _, _, vh = np.linalg.svd(pts - centroid, full_matrices=False)
    axes = vh[:2]
    proj2d = (pts - centroid) @ axes.T
    return proj2d, centroid, axes

def rasterize(points2d):
    min_xy = points2d.min(axis=0)
    grid = ((points2d - min_xy) / RES).astype(int)
    x_max, y_max = grid[:,0].max(), grid[:,1].max()
    H, W = y_max + 3, x_max + 3
    img = np.zeros((H, W), dtype=bool)
    img[grid[:,1], grid[:,0]] = True
    return img, min_xy

def morph_close(img):
    return binary_erosion(binary_dilation(img, disk(DISK_RADIUS)), disk(DISK_RADIUS))

def skeleton_image(img):
    return skeletonize(img)

def build_graph(skel_img, min_xy, centroid, axes):
    sk = csr.Skeleton(skel_img)
    adj_full = sk.graph
    adj = adj_full[1:,1:]
    try:
        pixel_graph = nx.from_scipy_sparse_array(adj)
    except AttributeError:
        pixel_graph = nx.from_scipy_sparse_matrix(adj)

    G = nx.Graph()
    coords = sk.coordinates[1:]  # skip dummy zero-th
    for u, v in pixel_graph.edges():
        y_u, x_u = coords[u]
        y_v, x_v = coords[v]
        pt2d_u = np.array([x_u, y_u]) * RES + min_xy
        pt2d_v = np.array([x_v, y_v]) * RES + min_xy
        pt3d_u = centroid + pt2d_u[0]*axes[0] + pt2d_u[1]*axes[1]
        pt3d_v = centroid + pt2d_v[0]*axes[0] + pt2d_v[1]*axes[1]
        G.add_node(u, point3d=pt3d_u)
        G.add_node(v, point3d=pt3d_v)
        G.add_edge(u, v)
    return G

def prune_graph(G):
    to_remove = [
        (u, v) for u, v in G.edges()
        if np.linalg.norm(G.nodes[u]['point3d'] - G.nodes[v]['point3d']) < MIN_EDGE_LENGTH
    ]
    G.remove_edges_from(to_remove)
    G.remove_nodes_from(list(nx.isolates(G)))
    return G

def graph_to_lineset(G):
    idx = {n:i for i,n in enumerate(G.nodes())}
    pts  = [G.nodes[n]['point3d'] for n in G.nodes()]
    lines = [[idx[u], idx[v]] for u,v in G.edges()]
    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(np.array(pts))
    ls.lines  = o3d.utility.Vector2iVector(np.array(lines))
    ls.colors = o3d.utility.Vector3dVector([[1,0,0] for _ in lines])
    return ls

def main():
    pcd = load_and_denoise(INPUT_PCD)
    proj2d, centroid, axes = project_to_plane(pcd)
    img, min_xy = rasterize(proj2d)
    img = morph_close(img)
    skel = skeleton_image(img)
    G = build_graph(skel, min_xy, centroid, axes)
    G = prune_graph(G)
    # nx.write_gpickle(G, OUTPUT_GRAPH)
    line_set = graph_to_lineset(G)
    o3d.visualization.draw_geometries(
        [pcd, line_set],
        window_name="Web + Extracted Graph",
        width=1024, height=768
    )
    print(f"Graph has {G.number_of_nodes()} nodes, {G.number_of_edges()} edges.")

if __name__ == "__main__":
    main()
