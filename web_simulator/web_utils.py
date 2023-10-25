from __future__ import annotations
import random
from typing import Callable, Union
import numpy as np

from .web import Point3D, VERTICES_T, EDGES_T, Web


def sample_segment(p1: Point3D, p2: Point3D, threshold_min: float = 1.0, threshold_max: Union[float, np.infty] = np.infty):
    d = p1.dist(p2)
    if d < threshold_min:
        return True
    elif d > threshold_max:
        return False
    else:
        # probability is inversely proportional to the distance
        # may change this function if needed
        prob = threshold_min / (d ** 2)
        return random.random() < prob


def sample_vertices(func: Callable, R: float = 1.0, N: int = 1000, epsilon: float = 0.01) -> VERTICES_T:
    # func: 2-variable function
    xy_coords = np.random.uniform(-R, R, size=(N, 2))
    z_coords = np.array([func(xy[0], xy[1]) + np.random.uniform(-epsilon, epsilon) for xy in xy_coords])
    vertices = [Point3D(xy[0], xy[1], z) for xy, z in zip(xy_coords, z_coords)]
    return vertices


def sample_edges(vertices: VERTICES_T, threshold_min: float = 1.0, threshold_max: Union[float, np.infty] = np.infty) -> EDGES_T:
    # return pair of indices corresponding to the sampled edges
    edges = []
    for i in range(len(vertices)):
        for j in range(i + 1, len(vertices)):
            if sample_segment(vertices[i], vertices[j], threshold_min=threshold_min, threshold_max=threshold_max):
                edges.append((i, j))
    return edges


def sample_web(
    func: Callable,
    R: float = 1.0,
    N: int = 1000,
    epsilon: float = 0.01,
    threshold_min: float = 1.0,
    threshold_max: Union[float, np.infty] = np.infty,
) -> Web:
    vertices = sample_vertices(func, R, N, epsilon)
    edges = sample_edges(vertices, threshold_min, threshold_max)
    return Web(vertices, edges)
