import cv2
import numpy as np

def apply_perspective_transform(image, corners):
    width, height = 500, 500  # Output resolution

    pts1 = np.float32(corners)
    pts2 = np.float32([[0, 0], [width, 0], [width, height], [0, height]])

    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    warped = cv2.warpPerspective(image, matrix, (width, height))
    
    return warped
