#Testing the Scarabble Board detection logic on a static image instead of live camera feed

import cv2
import numpy as np

# Load the test image
image = cv2.imread('test_images/board_test.jpg')

# Function to detect Scrabble board
def detect_board(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    lower_red1 = np.array([0, 70, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 70, 50])
    upper_red2 = np.array([180, 255, 255])
    
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = mask1 | mask2
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest_contour) > 1000:
            cv2.drawContours(frame, [largest_contour], -1, (0, 255, 0), 2)
            x, y, w, h = cv2.boundingRect(largest_contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
    return frame

# Apply detection
detected_frame = detect_board(image)
cv2.imshow('Board Detection Test', detected_frame)
cv2.waitKey(0)
cv2.destroyAllWindows()
