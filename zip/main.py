# set imports
import sys
import os
import cv2

current_dir = os.path.dirname(__file__)  # Get the directory of the current script
parent_dir = os.path.abspath(
    os.path.join(current_dir, "..")
)  # Get the absolute path of the parent directory
sys.path.append(parent_dir)

from utils import detect_grid_lines, cut_cells_from_image
from zip_utils import build_grid_from_cells, find_hamiltonian_path

# Main script to process the image and find the Hamiltonian path
img = cv2.imread("./zip/images/zip.jpg")
intersections, clustered_xs, clustered_ys = detect_grid_lines(img)
cells = cut_cells_from_image(img, clustered_xs, clustered_ys)
grid = build_grid_from_cells(cells, len(clustered_ys) - 1, len(clustered_xs) - 1)

# print the extracted grid
print("Extracted Grid:")
for row in grid:
    print(" | ".join(row))
path = find_hamiltonian_path(grid)

# Create a visualization grid with path steps
path_grid = [["  " for _ in row] for row in grid]
for i, (r, c) in enumerate(path):
    path_grid[r][c] = f"{i:02}"

print("\nHamiltonian Path Over Grid:")
for row in path_grid:
    print(" | ".join(row))
