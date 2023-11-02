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

    
    def merge_close_vertices(self, max_distance: float):
        """Vertices that have two edges of length max_distance or less are eliminated, and the two edges are merged into one.
        
        max_distance (float): the max pythagorean distacne that two edges can be to be merged. Recommended to be very small.

        Returns None.
        """
        for index in self.index_to_point.keys():
            neighbors = self.index_graph.neighbors(index)
            if len(neighbors) == 2:
                index_1 = neighbors[0]
                index_2 = neighbors[1]

                point = self.index_to_point[index]
                neighbor_1 = self.index_to_point[index_1]
                neighbor_2 = self.index_to_point[index_2]
                if point.dist(neighbor_1) <= max_distance and point.dist(neighbor_2) <= max_distance and self.collinear_within_n(0.0174533, point, neighbor_1, neighbor_2):
                    self.index_graph.add_edge(index_1, index_2) # add new edge between neighbors
                    self.index_to_point.pop(index) # remove original point from map
                    self.index_graph.delete_vertices(index) # remove original vertex from graph
                
    
    def collinear_within_n(n_radians: float, point_1: Point3D, point_2: Point3D, point_3: Point3D):
        """Determines whether the given three points are collinear to each other, within a error margin of n_radians.
        
        n_degrees (float): the max possible deviation from the line formed by the three points, in radians.
        point_1 (Point3D): first point to be considered.
        point_2 (Point3D): second point to be considered.
        point_3 (Point3D): third point to be considered.

        Returns (bool) whether the three points are collinear within n_radians.
        """
        # two vectors formed by the three points
        vector_1 = [point_1.x - point_2.x, point_1.y - point_2.y, point_1.z - point_2.z]
        vector_2 = [point_1.x - point_3.x, point_1.y - point_3.y, point_1.z - point_3.z]

        # angle calculated using: <v1, v2> / (||v1||*||v2||) with np.clip to remove extreme values
        angle_between_vectors = np.arccos(np.clip(np.dot(vector_1, vector_2) / (np.linalg.norm(vector_1) * np.linalg.norm(vector_2)), -1, 1))

        return abs(angle_between_vectors - np.pi) <= n_radians # return whether the deviation from 0/pi is within n_radians
        
