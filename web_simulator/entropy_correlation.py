import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from cubes_density_entropy_calculator import load_pcd, calculate_density_levels, record_distribution, compute_entropy, calculate_entropies_for_subdivisions
import os
import csv
from natsort import natsorted

# BEFORE RUNNING THIS FILE, SCROLL TO SPECIES_LIST AND COMMENT OUT THE PCDS YOU DON'T HAVE!!!

# Loop through all files in the folder
if __name__ == "__main__":
    web_names = []
    web_points = []
    entropies_without_zero = []
    entropies_with_zero = []
    
    print("Starting correlation analysis... ")

    # Sort the list of spider webs
    folder_path = "../video_processing/point_clouds" # TODO: Change this to where pcds are stored
    sorted_file_paths = []
    for file_name in os.listdir(folder_path):
        if file_name[-4:] == '.pcd':               
            sorted_file_paths.append(file_name)
    sorted_file_paths = natsorted(sorted_file_paths)
    print(sorted_file_paths)

    # Compute entropies
    for file_name in sorted_file_paths:
        print(f"Starting analysis on {file_name[:8]}... ")

        # Load the pcd
        file_path = os.path.join(folder_path, file_name)
        points = load_pcd(file_path)

        # Name and number of points
        web_names.append(file_path[33:41]) # Ignore the ../video_processing/point_cloduds/, then [@/!] + [sample num] + [space] + [brightness]
        web_points.append(len(points))

        # Entropy calculations
        entropies_including_zero, entropies_excluding_zero = calculate_entropies_for_subdivisions(points)
        entropies_with_zero.append(float(f"{entropies_including_zero[-1][1]:.5g}"))         # 5 represents number of sig figs
        entropies_without_zero.append(float(f"{entropies_excluding_zero[-1][1]:.5g}"))
        
    species_list = [
    'N. digna Male',                # !010
    'Needs ID',                     # @010
    'Needs ID',                     # @011
    'N. digna Female',              # @012
    # 'Needs ID',                     # @013 small
    'N. digna Imm. male', 
    'N. digna Female', 
    'Needs ID', 
    'N. digna Female', 
    'N. digna Female', 
    'N. digna Female', 
    'N. digna Female', 
    'N. digna Female',              # @021
    'Needs ID',                     # @022
    'N. digna Female',              # @023
    'N. litigiosa Female',          # @024
    # 'Needs ID',                     # @025 small
    'N. digna Female', 
    'N. digna Imm. male', 
    # 'N. digna Female',              # @028 small
    'N. digna Female', 
    'M. dana Female', 
    'N. digna Female', 
    # 'Therrid spp.',                   # @032 small
    'M. dana Female', 
    'M. dana Female', 
    'M. dana Female', 
    'N. digna Female', 
    'N. litigiosa Female', 
    'M. dana Female', 
    'M. dana Female', 
    'N. digna Penultimate Male', 
    'N. digna Female', 
    'M. dana Female', 
    'M. dana Female', 
    'M. dana Imm. male',                # @044
    'M. dana Juv'                       # @045
    ]
    
    # Create the output csv
    rows = zip(web_names, species_list, entropies_with_zero, entropies_without_zero)
    with open("output_cubes.csv", "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Sample ID", "Species", "H in bits", "H (Excl. Level 0 in bits)"])  # Header
        writer.writerows(rows)

    # Calculate the correlation
    correlation_matrix = np.corrcoef(web_points, entropies_without_zero)
    correlation = correlation_matrix[0, 1]
    print("Correlation without zero level:", correlation)

    correlation_matrix = np.corrcoef(web_points, entropies_with_zero)
    correlation = correlation_matrix[0, 1]
    print("Correlation with zero level: ", correlation)
