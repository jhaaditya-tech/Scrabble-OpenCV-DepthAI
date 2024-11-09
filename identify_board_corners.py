import cv2
import cv2.aruco as aruco
import depthai as dai

# ArUco setup
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
parameters = aruco.DetectorParameters()

# Corner marker IDs (adjust these based on your actual setup)
corner_ids = {
    "top_left": 0,      # ID for top-left corner
    "top_right": 1,     # ID for top-right corner
    "bottom_right": 2,  # ID for bottom-right corner
    "bottom_left": 3    # ID for bottom-left corner
}

def detect_aruco_markers(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
    
    detected_corners = {}
    
    if ids is not None:
        aruco.drawDetectedMarkers(frame, corners, ids)
        
        # Map detected corners to specific IDs
        for i, marker_id in enumerate(ids.flatten()):
            if marker_id in corner_ids.values():
                position = list(corner_ids.keys())[list(corner_ids.values()).index(marker_id)]
                detected_corners[position] = corners[i][0]
        
        print("Detected corner markers:", detected_corners)
        
    else:
        print("No markers detected.")
    
    cv2.imshow("Aruco Detection", frame)
    return detected_corners

# Pipeline setup for DepthAI
pipeline = dai.Pipeline()
cam_rgb = pipeline.create(dai.node.ColorCamera)
cam_rgb.setPreviewSize(1280, 720)
cam_rgb.setInterleaved(False)
xout_rgb = pipeline.create(dai.node.XLinkOut)
xout_rgb.setStreamName("rgb")
cam_rgb.preview.link(xout_rgb.input)

with dai.Device(pipeline) as device:
    q_rgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)

    while True:
        in_rgb = q_rgb.tryGet()
        if in_rgb:
            frame_rgb = in_rgb.getCvFrame()
            corners = detect_aruco_markers(frame_rgb)
            
            # Check if all four corners are detected
            if corners and len(corners) == 4:
                print("All four corners detected:", corners)
                break  # Exit the loop if all corners are detected

        if cv2.waitKey(1) == ord('q'):
            print("Exited by user.")
            break

cv2.destroyAllWindows()
