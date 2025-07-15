import pytesseract
import cv2
import numpy as np


def ocr_cell(cell_img):
    """Extract number from a single cell using Tesseract OCR."""
    # Convert to grayscale and apply thresholding
    gray = cv2.cvtColor(cell_img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)
    h, w = thresh.shape[:2]
    # Crop to center
    x1, x2 = int(w * 0.35), int(w * 0.65)
    y1, y2 = int(h * 0.35), int(h * 0.70)
    center_crop = thresh[y1:y2, x1:x2]

    # Invert and dilate to enhance digits
    inverted = cv2.bitwise_not(center_crop)
    kernel = np.ones((2, 2), np.uint8)
    dilated = cv2.dilate(inverted, kernel, iterations=1)
    result = cv2.bitwise_not(dilated)

    # Resize for better OCR accuracy
    resized = cv2.resize(result, None, fx=3, fy=3, interpolation=cv2.INTER_LINEAR)

    # Use Tesseract to extract digits
    text = pytesseract.image_to_string(resized, config="--psm 10 digits")
    text = "".join(filter(str.isdigit, text))
    return text if text else "*"


def build_grid_from_cells(cells, num_rows, num_cols):
    """
    Convert list of cell images to grid of OCR values.
    """
    grid = []
    idx = 0
    for r in range(num_rows):
        row = []
        for c in range(num_cols):
            if idx < len(cells):
                value = ocr_cell(cells[idx])
                row.append(value)
                idx += 1
            else:
                row.append("*")  # if something went wrong
        grid.append(row)
    return grid


def find_hamiltonian_path(grid):
    rows, cols = len(grid), len(grid[0])

    # Parse all positions and constraints
    targets = {}
    total_to_visit = 0
    start = None
    max_number = 0
    max_number_position = None

    for r in range(rows):
        for c in range(cols):
            val = grid[r][c]
            if val == "*":
                total_to_visit += 1
            elif val.isdigit():
                num = int(val)
                targets[num] = (r, c)
                total_to_visit += 1
                if num == 1:
                    start = (r, c)
                if num > max_number:
                    max_number = num
                    max_number_position = (r, c)

    visited = set()
    path = []

    def dfs(r, c, current_number, visited_count):
        nonlocal path

        if (r, c) in visited:
            return False

        visited.add((r, c))
        path.append((r, c))

        cell = grid[r][c]
        if cell.isdigit():
            cell_num = int(cell)
            if cell_num != current_number:
                visited.remove((r, c))
                path.pop()
                return False
            # If this is a numbered cell and matched, prepare for the next expected number
            current_number += 1

        if visited_count == total_to_visit:
            # Only valid if we're at the highest-numbered cell
            if (r, c) == max_number_position:
                return True
            else:
                visited.remove((r, c))
                path.pop()
                return False

        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                next_cell = grid[nr][nc]
                if next_cell == "*" or (
                    next_cell.isdigit() and int(next_cell) == current_number
                ):
                    if dfs(nr, nc, current_number, visited_count + 1):
                        return True

        visited.remove((r, c))
        path.pop()
        return False

    if start is None:
        return None

    if dfs(start[0], start[1], 1, 1):
        return path
    else:
        return None
