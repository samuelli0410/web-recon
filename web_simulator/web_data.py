import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Tuple

from web_simulator.web_graph import plane_section
from web_simulator.point import Point3D


def save_web(path: str, points: List[Point3D], edges: List[Tuple[int, int]]) -> None:
    """Save web data as `path/points.csv` and `path/edges.csv`.

    Args:
        path (str): Directory to save the web.
        points (List[Point3D]): Vertices of the web.
        edges (List[Tuple[int, int]]): Pairs of indices that represent edges of the web.
    """
    # points
    columns = ["x", "y", "z"]
    coords = [p.to_list() for p in points]
    df_points = pd.DataFrame(data=coords, columns=columns, dtype=float)
    df_points.to_csv(Path(path) / "points.csv")

    # edges
    columns = ["i", "j"]
    arr = [[e[0], e[1]] for e in edges]  # (num_edges, 2)
    df_edges = pd.DataFrame(data=arr, columns=columns, dtype=int)
    df_edges.to_csv(Path(path) / "edges.csv")


def save_sections(path: str, points: List[Point3D], edges: List[Tuple[int, int]], num_section: int):
    """Save vertical plan sections of a web as `path/sections.csv`.

    Args:
        path (str): _description_
        points (List[Point3D]): _description_
        edges (List[Tuple[int, int]]): _description_
        num_section (int): _description_
    """
    columns = ["theta", "x", "y", "z"]
    df = pd.DataFrame({c: pd.Series(dtype=float) for c in columns}, dtype=float)
    for i in range(num_section):
        theta = (i / num_section) * (2 * np.pi)
        section = plane_section(points, edges, theta)
        num_points = len(section)
        arr = np.array([theta] * num_points)
        arr = np.expand_dims(arr, axis=1)  # (num_points, 1)
        coords = np.array([p.to_list() for p in section])  # (num_points, 3)
        arr = np.concatenate([arr, coords], axis=1)  # (num_points, 4)
        df_section = pd.DataFrame(data=arr, columns=columns, dtype=float)
        df = pd.concat([df, df_section], ignore_index=True)

    print(f"Total {len(df)} points from {num_section} sections.")
    # save
    df.to_csv(Path(path) / "sections.csv")


def load_web(path: str):
    path_points = Path(path) / "points.csv"
    path_edges = Path(path) / "edges.csv"

    df_points = pd.read_csv(path_points)
    points = df_points.values.tolist()
    points = [Point3D(p[0], p[1], p[2]) for p in points]

    df_edges = pd.read_csv(path_edges)
    edges = df_edges.values.tolist()
    edges = [(e[0], e[1]) for e in edges]

    return points, edges
