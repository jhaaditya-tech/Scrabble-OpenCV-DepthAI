# Load ArUco dictionary and parameters
# aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_50)
# parameters = aruco.DetectorParameters_create()

# def detect_aruco_markers(frame):
#     # Convert frame to grayscale
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#     # Detect ArUco markers
#     corners, ids, _ = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
    
#     # Draw markers and show the frame with detected markers
#     if ids is not None:
#         aruco.drawDetectedMarkers(frame, corners, ids)
#         print("Detected ArUco marker IDs:", ids)
#     else:
#         print("No markers detected.")
    
#     cv2.imshow("Aruco Detection", frame)
#     return corners, ids

# # Set up the DepthAI pipeline
# pipeline = dai.Pipeline()

# # Set up RGB camera node
# cam_rgb = pipeline.create(dai.node.ColorCamera)
# cam_rgb.setPreviewSize(640, 480)
# cam_rgb.setInterleaved(False)

# # Set up output node for RGB data
# xout_rgb = pipeline.create(dai.node.XLinkOut)
# xout_rgb.setStreamName("rgb")

# # Link RGB camera output to the XLinkOut
# cam_rgb.preview.link(xout_rgb.input)

# # Run pipeline
# with dai.Device(pipeline) as device:
#     q_rgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)

#     while True:
#         in_rgb = q_rgb.tryGet()
#         if in_rgb:
#             frame_rgb = in_rgb.getCvFrame()
#             corners, ids = detect_aruco_markers(frame_rgb)

#         if cv2.waitKey(1) == ord('q'):
#             break

# cv2.destroyAllWindows()
