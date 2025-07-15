# LinkedIn Puzzle Game Solver

This repository provides a lightweight computer vision-based solver for the [LinkedIn "Zip" puzzle game](https://www.linkedin.com/games/zip/?isGameStart=false). Given an image of the game board, it extracts the grid layout and cell values, then computes a valid Hamiltonian path that respects the number constraints and fills the entire grid.

## Pipeline Overview
1. **Preprocessing**:

   * Convert image to grayscale
   * Apply Canny edge detection
   * Use Hough Transform to detect grid lines

2. **Grid Detection**:

   * Cluster detected lines to determine rows and columns
   * Find intersections to locate grid cell corners
   * Crop individual cells based on intersections

3. **Digit Recognition**:

   * Use Tesseract OCR or template matching to identify numbers inside the cells
   * Represent empty cells with `*`

4. **Pathfinding**:

   * Build a graph based on adjacency
   * Perform a DFS-based search to find a Hamiltonian path.
   * Ensure numbered cells are visited in order 1->2->3->...->X and final cell matches the highest number

5. **Visualization**:

   * Overlay the discovered path on a visual grid using step indices

## Technologies Used

* Python
* OpenCV
* NumPy
* pytesseract (for OCR)

## Setup

```bash
pip install -r requirements.txt
```
Make sure Tesseract OCR is installed and available in your PATH if using pytesseract.

## Example
<img src="https://github.com/joelvdb/computer-vision-linkedin-games-solver/blob/main/zip/images/zip.jpg?raw=true" width="200" />
<img src="https://github.com/joelvdb/computer-vision-linkedin-games-solver/blob/main/zip/images/zip_processed.png?raw=true" width="200" />


```
Extracted Grid:
1 | * | * | * | * | *
* | * | * | * | 7 | *
* | 4 | * | 8 | * | *
* | * | 6 | * | 5 | *
* | 3 | * | * | * | *
* | * | * | * | * | 2
```
```
Path Over Grid:
00 | 19 | 20 | 21 | 22 | 23
01 | 18 | 31 | 32 | 33 | 24
02 | 17 | 30 | 35 | 34 | 25
03 | 16 | 29 | 28 | 27 | 26
04 | 15 | 14 | 13 | 12 | 11
05 | 06 | 07 | 08 | 09 | 10
```


