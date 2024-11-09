import cv2
import numpy as np
import depthai as dai
import cv2.aruco as aruco

"""
Captures an images of a bord using Depth AI, detects the specfied AruCO markers for each corner, and perfoems a perspective transformation
to align the boardto a standard orientation. It displays three views: a live preview, a labeled corner detetcrion view, and a final aligned 
board view

"""

# Define ArUco marker dictionary and parameters
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
parameters = aruco.DetectorParameters()

# Define expected marker IDs for each corner
corner_ids = {
    "top_left": 10,
    "top_right": 20,
    "bottom_left": 30,
    "bottom_right": 40
}

def detect_aruco_markers(frame, show_labels=False):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
    
    detected_corners = { "top_left": None, "top_right": None, "bottom_right": None, "bottom_left": None }
    if ids is not None:
        if show_labels:
            aruco.drawDetectedMarkers(frame, corners, ids)
        
        for i, marker_id in enumerate(ids.flatten()):
            for position, expected_id in corner_ids.items():
                if marker_id == expected_id:
                    detected_corners[position] = np.mean(corners[i][0], axis=0)
                    if show_labels:
                        # Draw circle and label only if show_labels is True
                        center = tuple(detected_corners[position].astype(int))
                        cv2.circle(frame, center, 10, (0, 255, 0), -1)
                        cv2.putText(frame, f"{position} ({marker_id})", (center[0] + 10, center[1] - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    
    ordered_corners = [detected_corners[key] for key in ["top_left", "top_right", "bottom_right", "bottom_left"]]
    
    if all(corner is not None for corner in ordered_corners):
        return np.array(ordered_corners, dtype="float32")
    else:
        print("Error: Not all required corners detected.")
        return None

def align_board(frame, src_points):
    if src_points is None:
        print("Error: Source points are missing.")
        return None

    board_size = 800  # Define board size
    dst_points = np.array([
        [0, 0],                          # Top-left corner in output
        [board_size - 1, 0],             # Top-right corner in output
        [board_size - 1, board_size - 1],# Bottom-right corner in output
        [0, board_size - 1]              # Bottom-left corner in output
    ], dtype="float32")

    matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    aligned_board = cv2.warpPerspective(frame, matrix, (board_size, board_size))
    
    return aligned_board

# Setup DepthAI pipeline to capture a single frame
pipeline = dai.Pipeline()
cam_rgb = pipeline.create(dai.node.ColorCamera)
cam_rgb.setPreviewSize(1920, 1080)
cam_rgb.setInterleaved(False)
xout_rgb = pipeline.create(dai.node.XLinkOut)
xout_rgb.setStreamName("rgb")
cam_rgb.preview.link(xout_rgb.input)

# Run the DepthAI pipeline
with dai.Device(pipeline) as device:
    q_rgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
    frame_rgb = None

    while True:
        in_rgb = q_rgb.tryGet()
        if in_rgb:
            frame_rgb = in_rgb.getCvFrame()
            cv2.imshow("Camera Preview - Press Space to Capture", frame_rgb)

        key = cv2.waitKey(1)
        
        if key == ord(' '):  # Spacebar to capture frame
            if frame_rgb is not None:
                print("Frame captured. Detecting markers and aligning board...")
                
                # Create a copy for corner detection display only
                frame_with_labels = frame_rgb.copy()
                
                # Detect corners with labels only on this copy
                detected_corners = detect_aruco_markers(frame_with_labels, show_labels=True)
                cv2.imshow("Detected Corners with Labels", frame_with_labels)

                # Perform perspective transformation if all four corners are detected
                if detected_corners is not None and detected_corners.shape[0] == 4:
                    aligned_board = align_board(frame_rgb, detected_corners)
                    
                    if aligned_board is not None:
                        # Show the aligned board without any marker labels
                        cv2.imshow("Aligned Board without Marker Labels", aligned_board)
                    else:
                        print("Failed to align board - check corners.")
                else:
                    print("Not all corners detected. Try again.")
            else:
                print("No frame captured.")
        
        elif key == ord('q'):  # Press 'q' to quit
            print("Exited by user.")
            break

    cv2.destroyAllWindows()