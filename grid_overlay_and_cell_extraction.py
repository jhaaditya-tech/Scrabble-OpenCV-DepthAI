import cv2
import numpy as np
from perspective_transformation import align_board

import cv2
import numpy as np
from perspective_transformation import align_board  # Import your perspective transformation function

# Define the board size and grid layout
BOARD_SIZE = 800  # Size should match the output size from perspective_transformation.py
GRID_SIZE = 15    # Number of rows and columns on the board for a Scrabble grid
CELL_SIZE = BOARD_SIZE // GRID_SIZE  # Size of each cell in pixels

# Capture and align the board
# Use your existing code to capture and align the board with `align_board`
frame = ...  # Capture or load your input frame
detected_corners = ...  # Detect corners using the perspective transformation method

aligned_board = align_board(frame, detected_corners)

if aligned_board is not None:
    # Draw the grid overlay for visualization
    for i in range(0, BOARD_SIZE, CELL_SIZE):
        cv2.line(aligned_board, (0, i), (BOARD_SIZE, i), (255, 255, 255), 1)
        cv2.line(aligned_board, (i, 0), (i, BOARD_SIZE), (255, 255, 255), 1)

    cv2.imshow("Aligned Board with Grid", aligned_board)

    # Extract each cell as a sub-image for further processing
    cells = []
    for row in range(GRID_SIZE):
        row_cells = []
        for col in range(GRID_SIZE):
            x_start, y_start = col * CELL_SIZE, row * CELL_SIZE
            cell = aligned_board[y_start:y_start + CELL_SIZE, x_start:x_start + CELL_SIZE]
            row_cells.append(cell)
            # Further processing like OCR can be done here
        cells.append(row_cells)

    # Show a sample cell if desired
    cv2.imshow("Sample Cell", cells[0][0])

cv2.waitKey(0)
cv2.destroyAllWindows()
