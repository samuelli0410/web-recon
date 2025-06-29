#!/usr/bin/env python3
import open3d as o3d
import networkx as nx
import numpy as np
import pickle
import os
import sys

def main():
    # ---- 1. Load the PCD ----
    pcd_path = "video_processing/point_clouds/sparse3 255 2024-11-30 11-29-33.pcd"
    if not os.path.isfile(pcd_path):
        print(f"Error: file not found:\n  {pcd_path}")
        sys.exit(1)
    pcd = o3d.io.read_point_cloud(pcd_path)
    pts = np.asarray(pcd.points)
    print(f"[INFO] Loaded PCD with {pts.shape[0]} points.")

    # ---- 2. Pick points to become nodes ----
    print("""
INSTRUCTIONS:
  • In the Open3D window, hold ⇧ Shift and Left-click to select each node.
  • When you’re done picking all nodes, press Q (or Esc) to close the window.
""")
    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window(window_name="Pick graph nodes")
    vis.add_geometry(pcd)
    vis.run()          # user picks points
    vis.destroy_window()
    picked = vis.get_picked_points()
    if not picked:
        print("[WARN] No points picked; exiting.")
        sys.exit(0)

    # ---- 3. Build NetworkX graph with those nodes ----
    G = nx.Graph()
    for i, idx in enumerate(picked):
        coord = pts[idx]
        G.add_node(i, pos=tuple(coord))
        print(f"[NODE] {i}: point index {idx} at {tuple(coord)}")

    # ---- 4. Manually add edges ----
    print("""
EDGE ENTRY MODE:
  Enter two node indices (u v) to connect with an edge.
  Hit Enter on a blank line when you’re done.
""")
    while True:
        line = input("Edge (u v): ").strip()
        if line == "":
            break
        try:
            u, v = map(int, line.split())
            if u in G.nodes and v in G.nodes:
                G.add_edge(u, v)
                print(f"[EDGE] Added: {u} -- {v}")
            else:
                print("  → Invalid node index; valid range is 0 to", len(G.nodes)-1)
        except ValueError:
            print("  → Please enter two integers separated by space.")

    # ---- 5. Save the graph ----
    out_file = "graph.pkl"
    with open(out_file, "wb") as f:
        pickle.dump(G, f)
    print(f"[DONE] Graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges saved to {out_file}")

if __name__ == "__main__":
    main()
