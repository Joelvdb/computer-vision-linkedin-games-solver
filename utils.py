import cv2
import numpy as np
from itertools import combinations
import pytesseract


def detect_corners_of_grid(intersections):
    """
    Given a list of (x,y) intersections, find the corners of the grid.
    """
    if len(intersections) < 4:
        raise ValueError("Not enough intersections to form a grid.")
    xs, ys = zip(*intersections)
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    # Find corners
    tl = (min_x, min_y)  # Top-left
    rt = (max_x, min_y)  # Top-right
    bl = (min_x, max_y)  # Bottom-left
    br = (max_x, max_y)  # Bottom-right
    return tl, rt, bl, br


def cluster_positions(positions, eps=10):
    """Cluster 1D positions (x or y) using simple distance threshold."""
    positions = sorted(positions)
    clusters = []
    current_cluster = [positions[0]]

    for pos in positions[1:]:
        if abs(pos - current_cluster[-1]) < eps:
            current_cluster.append(pos)
        else:
            clusters.append(int(np.mean(current_cluster)))
            current_cluster = [pos]
    clusters.append(int(np.mean(current_cluster)))
    return clusters


def detect_grid_lines(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Could not load image.")

    # find edges using Canny
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)

    # Detect lines using HoughLinesP
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=100,
        minLineLength=50,
        maxLineGap=10,
    )

    if lines is None:
        raise ValueError("No lines detected.")

    vertical_lines = []
    horizontal_lines = []

    # find vertical and horizontal lines
    for line in lines:
        x1, y1, x2, y2 = line[0]
        dx = x2 - x1
        dy = y2 - y1

        if dx == 0:  # perfectly vertical
            vertical_lines.append((x1, y1, x2, y2))
        elif dy == 0:  # perfectly horizontal
            horizontal_lines.append((x1, y1, x2, y2))
        else:
            angle = abs(np.arctan2(dy, dx))
            if angle < np.pi / 6:
                horizontal_lines.append((x1, y1, x2, y2))
            elif angle > np.pi / 3:
                vertical_lines.append((x1, y1, x2, y2))

    print(f"Raw vertical lines: {len(vertical_lines)}")
    print(f"Raw horizontal lines: {len(horizontal_lines)}")

    # Cluster x-positions of vertical lines and y-positions of horizontal lines
    x_positions = [x1 for (x1, y1, x2, y2) in vertical_lines]
    y_positions = [y1 for (x1, y1, x2, y2) in horizontal_lines]

    clustered_xs = cluster_positions(x_positions, eps=10)
    clustered_ys = cluster_positions(y_positions, eps=10)

    # get vertical and horizontal lines and from clustered positions
    print(f"Clustered vertical lines: {len(clustered_xs)}")
    print(f"Clustered horizontal lines: {len(clustered_ys)}")

    height, width = img.shape[:2]

    # Generate clean vertical and horizontal lines from clusters
    vertical_lines_clean = [(x, 0, x, height) for x in clustered_xs]
    horizontal_lines_clean = [(0, y, width, y) for y in clustered_ys]

    # Compute intersections
    intersections = [(x, y) for x in clustered_xs for y in clustered_ys]
    print(f"Intersections found: {len(intersections)}")
    (tl, tr, bl, br) = detect_corners_of_grid(intersections)

    # Return all intersctions inside (tl, tr, bl, br)
    intersections = [
        (x, y)
        for (x, y) in intersections
        if tl[0] <= x <= tr[0] and tl[1] <= y <= bl[1]
    ]
    clustered_xs = [x for x in clustered_xs if tl[0] <= x <= tr[0]]
    clustered_ys = [y for y in clustered_ys if tl[1] <= y <= bl[1]]

    return intersections, clustered_xs, clustered_ys


def cut_cells_from_image(image_path, xs, ys):
    """Cut cells from an image based on clustered x and y lines positions."""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Could not load image.")

    cells = []
    for i in range(len(xs) - 1):
        for j in range(len(ys) - 1):
            x1, x2 = xs[i], xs[i + 1]
            y1, y2 = ys[j], ys[j + 1]
            cell_img = img[y1:y2, x1:x2]
            cells.append(cell_img)

    return cells
