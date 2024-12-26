#Divide the detected Scrabble board into a 15*15 grid and exact individual tiles

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


def correct_perspective(frame, contour):
    """Apply perspective transform to the detected board."""
    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    
    width = 600
    height = 600
    src_pts = box.astype(np.float32)
    dst_pts = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype=np.float32)
    
    # Get perspective transformation matrix
    matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped = cv2.warpPerspective(frame, matrix, (width, height))
    
    return warped


def extract_tiles(frame):
    """Detect Scrabble board and divide into a 15x15 grid."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Red color filtering
    lower_red1 = np.array([0, 70, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 70, 50])
    upper_red2 = np.array([180, 255, 255])
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = mask1 | mask2
    
    # Find the largest contour (Scrabble board)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        warped = correct_perspective(frame, largest_contour)
        
        # Grid division (15x15)
        height, width, _ = warped.shape
        tile_height = height // 15
        tile_width = width // 15
        
        for i in range(15):
            for j in range(15):
                tile_x = j * tile_width
                tile_y = i * tile_height
                cv2.rectangle(warped, 
                              (tile_x, tile_y), 
                              (tile_x + tile_width, tile_y + tile_height), 
                              (255, 0, 0), 1)
        
        return warped
    return frame


# Start the pipeline
with dai.Device(pipeline) as device:
    rgb_queue = device.getOutputQueue(name="rgb", maxSize=1, blocking=False)
    while True:
        frame = rgb_queue.get().getCvFrame()
        grid_frame = extract_tiles(frame)
        cv2.imshow("Tile Detection with Perspective Correction", grid_frame)
        if cv2.waitKey(1) == ord('q'):
            break

cv2.destroyAllWindows()


