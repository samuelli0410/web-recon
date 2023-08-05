from __future__ import annotations
import random
import math
from typing import Callable, List, Tuple, Union
import pandas as pd
import numpy as np

from .point import Point3D


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


def sample_points(func: Callable, R: float = 1.0, N: int = 1000, epsilon: float = 0.01) -> List[Point3D]:
    # func: 2-variable function
    xy_coords = np.random.uniform(-R, R, size=(N, 2))
    z_coords = np.array([func(xy[0], xy[1]) + np.random.uniform(-epsilon, epsilon) for xy in xy_coords])
    points = [Point3D(xy[0], xy[1], z) for xy, z in zip(xy_coords, z_coords)]
    return points


def sample_edges(points: List[Point3D], threshold_min: float = 1.0, threshold_max: Union[float, np.infty] = np.infty) -> List[Tuple[int, int]]:
    # return pair of indices corresponding to the sampled edges
    edges = []
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            if sample_segment(points[i], points[j], threshold_min=threshold_min, threshold_max=threshold_max):
                edges.append((i, j))
    return edges


def find_intersection(p1: Point3D, p2: Point3D, theta: float):
    tat = math.tan(theta)
    # math.tan(theta) * (p1.x + t * (p2.x - p1.x)) = (p1.y + t * (p2.y - p1.y))
    # tat * p1.x - p1.y = t * (p2.y - p1.y + tat * (p2.x - p1.x))
    if p2.y - p1.y + tat * (p2.x - p1.x) == 0.0:
        return None
    t = (tat * p1.x - p1.y) / (p2.y - p1.y + tat * (p2.x - p1.x))
    if 0 <= t <= 1:
        x = p1.x + t * (p2.x - p1.x)
        y = p1.y + t * (p2.y - p1.y)
        z = p1.z + t * (p2.z - p1.z)
        return Point3D(x, y, z)
    else:
        return None


def plane_section(points: List[Point3D], edges: List[Tuple[int, int]], theta: float) -> List[Point3D]:
    # find the intersection points with edges of web and plane represented by theta (polar coordinate)
    intersection_points = []
    for i, j in edges:
        p1, p2 = points[i], points[j]
        ip = find_intersection(p1, p2, theta)
        if ip is not None:
            intersection_points.append(ip)
    return intersection_points


def graph_to_df(points: List[Point3D], edges: List[Tuple[int, int]]) -> pd.DataFrame:
    rows = []
    for s, e in edges:
        rows.append([points[s].x, points[s].y, points[s].z])
        rows.append([points[e].x, points[e].y, points[e].z])
        rows.append([np.nan, np.nan, np.nan])
    df = pd.DataFrame(rows, columns=["x", "y", "z"])
    return df

