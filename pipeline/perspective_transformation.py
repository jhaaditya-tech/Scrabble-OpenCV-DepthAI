# perspective_transformation.py

import numpy as np


def order_points(points):
    """
    Orders contour points in a consistent order: top-left, top-right, bottom-right, bottom-left.

    Args:
        points (np.ndarray): Array of contour points.

    Returns:
        np.ndarray: Ordered points.
    """
    rect = np.zeros((4, 2), dtype="float32")

    # Calculate the sum and difference of points to identify corners
    s = points.sum(axis=1)
    diff = np.diff(points, axis=1)

    rect[0] = points[np.argmin(s)]  # Top-left point has the smallest sum
    rect[2] = points[np.argmax(s)]  # Bottom-right point has the largest sum
    rect[1] = points[np.argmin(diff)]  # Top-right point has the smallest difference
    rect[3] = points[np.argmax(diff)]  # Bottom-left point has the largest difference

    return rect
