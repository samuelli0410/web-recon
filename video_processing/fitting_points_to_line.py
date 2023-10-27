import numpy as np
import igraph as ig
from web_simulator.web import *


class CollinearPointsFitting:
    """Class which takes in a list of 3D points and creates a graph of their connections."""
    def __init__(self, list_of_points: list) -> None:
        self.index_graph = ig.Graph(n=len(list_of_points))
        self.index_to_point = {}
        for index, point in enumerate(list_of_points):
            self.index_to_point[index] = point


    def connect_within_n(self, max_distance: float):
        """Points within max_distance of each other are connected in this object's index_graph.
        
        max_distance (float): the max pythagorean distacne that two points can be apart to be connected. Recommended to be very small.

        Returns None.
        """
        for i_key, i_val in self.index_to_point.values():
            for j_key, j_val in self.index_to_point.values():
                if i_val.dist(j_val) <= max_distance:
                    self.index_graph.add_edge(i_key, j_key)

    
    