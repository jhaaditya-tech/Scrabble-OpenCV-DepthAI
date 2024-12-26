#Detect and outline the board using the camera feed

import depthai as dai
import cv2
import numpy as np

# Create DepthAI pipeline
pipeline = dai.Pipeline()

# Define camera node
cam_rgb = pipeline.create(dai.node.ColorCamera)
xout_rgb = pipeline.create(dai.node.XLinkOut)
xout_rgb.setStreamName("rgb")

# Camera configuration
cam_rgb.setPreviewSize(1280, 720)
cam_rgb.setInterleaved(False)
cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
cam_rgb.setFps(24)

# Link nodes
cam_rgb.preview.link(xout_rgb.input)

# Function to detect Scrabble board
def detect_board(frame):
    # Convert to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Define lower and upper bounds for red color
    lower_red1 = np.array([0, 70, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 70, 50])
    upper_red2 = np.array([180, 255, 255])
    
    # Create masks
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = mask1 | mask2
    
    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        # Find the largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest_contour) > 1000:  # Threshold to avoid noise
            # Draw contour
            cv2.drawContours(frame, [largest_contour], -1, (0, 255, 0), 2)
            # Get bounding box
            x, y, w, h = cv2.boundingRect(largest_contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
    return frame

# Start the pipeline
with dai.Device(pipeline) as device:
    rgb_queue = device.getOutputQueue(name="rgb", maxSize=1, blocking=False)
    while True:
        frame = rgb_queue.get().getCvFrame()
        detected_frame = detect_board(frame)
        cv2.imshow("Scrabble Board Detection", detected_frame)
        if cv2.waitKey(1) == ord('q'):
            break

cv2.destroyAllWindows()


"""
Converted to HSV as it is better for filtering than RGB
Red Color masking - two ranges are defined to account for variations in red
Contour detecton - cv2.findContours identifies the largest red area.
Bounding Box = A rectnage is drwan around the detected area
The detected board is displayted in rela time

"""