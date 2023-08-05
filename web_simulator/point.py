from __future__ import annotations
import math
from typing import Tuple, List


class Point3D:
    def __init__(self, x: float, y: float, z: float) -> None:
        self.x = x
        self.y = y
        self.z = z

    def dist(self, other: Point3D) -> float:
        return math.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2 + (self.z - other.z) ** 2)

    def __repr__(self) -> str:
        return f"Point3D({self.x},{self.y},{self.z})"

    def to_list(self) -> List[float, float, float]:
        return [self.x, self.y, self.z]

    @staticmethod
    def from_coords(x: float, y: float, z: float):
        return Point3D(x, y, z)
