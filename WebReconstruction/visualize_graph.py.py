#!/usr/bin/env python3
"""
visualize_graph.py

Visualizer for a saved NetworkX graph with 3D node positions.

Loads a .pkl file containing a NetworkX Graph where each node has a 'pos'
attribute: an (x, y, z) tuple. Renders nodes as spheres and edges as lines
using Open3D.

Provide the path to your graph.pkl by editing the GRAPH_FILE constant below.
"""
import pickle
import sys

import networkx as nx
import open3d as o3d

# === USER CONFIGURATION ===
# Hard-code your .pkl file path here (no CLI required).
GRAPH_FILE = "cube_1_1_1_graph.pkl"
# Appearance settings:
NODE_RADIUS = 0.02             # radius of each node-sphere
NODE_COLOR = [1.0, 0.2, 0.2]  # RGB in [0,1]
EDGE_COLOR = [0.2, 1.0, 0.2]  # RGB in [0,1]
# ==========================

def load_graph(path: str) -> nx.Graph:
    try:
        with open(path, "rb") as f:
            G = pickle.load(f)
    except Exception as e:
        print(f"Error loading graph from {path}: {e}")
        sys.exit(1)
    return G


def build_geometry(G: nx.Graph, node_radius: float, node_color, edge_color):
    # Ensure consistent ordering
    nodes = sorted(G.nodes())
    idx_map = {nid: i for i, nid in enumerate(nodes)}
    pts = [G.nodes[n]['pos'] for n in nodes]

    # Create a sphere at each node
    spheres = []
    for pos in pts:
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=node_radius)
        sphere.compute_vertex_normals()
        sphere.paint_uniform_color(node_color)
        sphere.translate(pos)
        spheres.append(sphere)

    # Create lines for edges
    lines = [(idx_map[u], idx_map[v]) for u, v in G.edges()]
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(pts),
        lines=o3d.utility.Vector2iVector(lines)
    )
    line_set.paint_uniform_color(edge_color)

    return spheres, line_set


def main():
    # Load graph from hard-coded path
    G = load_graph(GRAPH_FILE)

    # Build visualization geometry
    spheres, line_set = build_geometry(
        G,
        node_radius=NODE_RADIUS,
        node_color=NODE_COLOR,
        edge_color=EDGE_COLOR,
    )

    # Render
    o3d.visualization.draw_geometries(
        spheres + [line_set],
        window_name="Graph Visualization",
        width=800,
        height=600
    )


if __name__ == "__main__":
    main()
