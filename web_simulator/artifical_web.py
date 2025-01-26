import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from scipy.spatial import Delaunay

# Parameters
num_points = 8000
boundary_size = 1.0
edge_removal_prob = 0.025
num_rings = 10

half_size = boundary_size / 2.0
total_area = boundary_size**2

def distance_to_boundary(x, y, half_size):
    dist_x = half_size - abs(x)
    dist_y = half_size - abs(y)
    return min(dist_x, dist_y)

# Equal-area rings based on distance to boundary
area_per_ring = total_area / num_rings
ring_boundaries = [0.0]
for i in range(1, num_rings):
    inner_area = boundary_size**2 - i * area_per_ring
    boundary_dist = (boundary_size - np.sqrt(inner_area)) / 2.0
    ring_boundaries.append(boundary_dist)
ring_boundaries.append(half_size)

# Define densities that decrease from outer to inner.
outer_density = 0.1
inner_density = 0.001
rings = np.arange(num_rings)
densities = outer_density * (inner_density/outer_density)**(rings/(num_rings-1))

# Rejection sampling
accepted_points = []
while len(accepted_points) < num_points:
    cx = np.random.uniform(-half_size, half_size)
    cy = np.random.uniform(-half_size, half_size)
    d = distance_to_boundary(cx, cy, half_size)
    
    ring_index = np.searchsorted(ring_boundaries, d, side='right') - 1
    if ring_index < 0:
        ring_index = 0
    if ring_index >= num_rings:
        ring_index = num_rings - 1
    
    p = densities[ring_index]
    if np.random.rand() < p:
        accepted_points.append((cx, cy))

points = np.array(accepted_points)
x, y = points[:,0], points[:,1]

# Delaunay triangulation
triangulation = Delaunay(points)
G = nx.Graph()
for i in range(len(points)):
    G.add_node(i, pos=(x[i], y[i]))

edges = []
for simplex in triangulation.simplices:
    for a in range(3):
        for b in range(a+1, 3):
            edge = tuple(sorted((simplex[a], simplex[b])))
            edges.append(edge)

# Random edge removal
num_edges_to_remove = int(len(edges)*edge_removal_prob)
edges_to_remove_indices = np.random.choice(len(edges), size=num_edges_to_remove, replace=False)
edges_to_remove = {edges[i] for i in edges_to_remove_indices}

for edge in edges:
    if edge not in edges_to_remove:
        G.add_edge(edge[0], edge[1])

pos = {i: (x[i], y[i]) for i in range(len(points))}

# Plot density heat map
bins = 50
H, xedges, yedges = np.histogram2d(y, x, bins=bins, range=[[-half_size, half_size], [-half_size, half_size]])
plt.figure(figsize=(8,8))
img = plt.imshow(H, origin='lower', extent=(-half_size, half_size, -half_size, half_size),
           cmap='hot', interpolation='nearest')
plt.title("Density Distribution")

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

plt.axis('equal')
plt.show()

# Plot the network edges
plt.figure(figsize=(8,8))
nx.draw_networkx_edges(G, pos, edge_color="gray", alpha=0.7)
plt.title("Computer-Generated Web with Relatively High Entropy for Silk Density")
plt.xlim(-half_size, half_size)
plt.ylim(-half_size, half_size)
plt.axis('equal')
plt.show()





import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from scipy.spatial import Delaunay

# Parameters
num_points = 5000
boundary_size = 1.0
edge_removal_prob = 0.025
num_rings = 10

half_size = boundary_size / 2.0
total_area = boundary_size**2

def distance_to_boundary(x, y, half_size):
    dist_x = half_size - abs(x)
    dist_y = half_size - abs(y)
    return min(dist_x, dist_y)

# Equal-area rings based on distance to boundary
area_per_ring = total_area / num_rings
ring_boundaries = [0.0]
for i in range(1, num_rings):
    inner_area = boundary_size**2 - i * area_per_ring
    boundary_dist = (boundary_size - np.sqrt(inner_area)) / 2.0
    ring_boundaries.append(boundary_dist)
ring_boundaries.append(half_size)

# Define densities that heavily favor low-density regions
# Most rings have very low density, only the outermost ring has higher density
outer_density = 0.05  # Increased density for the outermost ring
low_density = 0.001    # Very low density for all other rings
densities = np.full(num_rings, low_density)
densities[0] = outer_density  # Only the outermost ring has higher density

# Rejection sampling
accepted_points = []
while len(accepted_points) < num_points:
    cx = np.random.uniform(-half_size, half_size)
    cy = np.random.uniform(-half_size, half_size)
    d = distance_to_boundary(cx, cy, half_size)
    
    ring_index = np.searchsorted(ring_boundaries, d, side='right') - 1
    if ring_index < 0:
        ring_index = 0
    if ring_index >= num_rings:
        ring_index = num_rings - 1
    
    p = densities[ring_index]
    if np.random.rand() < p:
        accepted_points.append((cx, cy))

points = np.array(accepted_points)
x, y = points[:,0], points[:,1]

# Delaunay triangulation
triangulation = Delaunay(points)
G = nx.Graph()
for i in range(len(points)):
    G.add_node(i, pos=(x[i], y[i]))

edges = []
for simplex in triangulation.simplices:
    for a in range(3):
        for b in range(a+1, 3):
            edge = tuple(sorted((simplex[a], simplex[b])))
            edges.append(edge)

# Random edge removal
num_edges_to_remove = int(len(edges)*edge_removal_prob)
edges_to_remove_indices = np.random.choice(len(edges), size=num_edges_to_remove, replace=False)
edges_to_remove = {edges[i] for i in edges_to_remove_indices}

for edge in edges:
    if edge not in edges_to_remove:
        G.add_edge(edge[0], edge[1])

pos = {i: (x[i], y[i]) for i in range(len(points))}

# Plot density heat map
bins = 50
H, xedges, yedges = np.histogram2d(y, x, bins=bins, range=[[-half_size, half_size], [-half_size, half_size]])
plt.figure(figsize=(8,8))
img = plt.imshow(H, origin='lower', extent=(-half_size, half_size, -half_size, half_size),
           cmap='hot', interpolation='nearest')
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

plt.title("Computer-Generated Web with Relatively Low Entropy for Silk Density")
plt.axis('equal')
plt.show()

# Plot the network edges
plt.figure(figsize=(8,8))
nx.draw_networkx_edges(G, pos, edge_color="gray", alpha=0.7)
plt.title("")
plt.xlim(-half_size, half_size)
plt.ylim(-half_size, half_size)
plt.axis('equal')
plt.show()


