import cv2
import numpy as np

def process_image(image_path):
    """
    Processes the captured image to detect and warp the Scrabble board.

    Args:
        image_path (str): Path to the captured image.
    
    Returns:
        np.ndarray: Warped image of the detected board.
    """
    # Load the captured image
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"❌ Failed to load image at {image_path}")

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian Blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Detect edges using Canny
    edges = cv2.Canny(blurred, 50, 150)

    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv2.contourArea, default=None)

    if largest_contour is None:
        raise ValueError("❌ No contours detected in the image.")

    # Approximate the contour to a polygon
    epsilon = 0.02 * cv2.arcLength(largest_contour, True)
    approx = cv2.approxPolyDP(largest_contour, epsilon, True)

    if len(approx) != 4:
        raise ValueError("❌ Could not detect a rectangular board.")

    # Sort the points for warping
    approx = sorted(approx, key=lambda x: x[0][0] + x[0][1])
    top_left, bottom_left = approx[:2]
    bottom_right, top_right = approx[2:]

    # Define the destination points
    width = 500
    height = 500
    dest_points = np.array([
        [0, 0],
        [width - 1, 0],
        [width - 1, height - 1],
        [0, height - 1]
    ], dtype="float32")

    # Perform perspective transformation
    src_points = np.array([top_left[0], top_right[0], bottom_right[0], bottom_left[0]], dtype="float32")
    matrix = cv2.getPerspectiveTransform(src_points, dest_points)
    warped = cv2.warpPerspective(image, matrix, (width, height))

    # Display warped im
