import cv2
import torch
import depthai as dai
import numpy as np

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'custom', 
                       path='/Users/adityajha/Documents/CV Thesis/Project Codes/Scrabble-OpenCV-DepthAI/yolov5_project/yolov5/runs/train/exp2/weights/best.pt')

# Initialize DepthAI Pipeline
pipeline = dai.Pipeline()

# Define the camera node
cam_rgb = pipeline.createColorCamera()
cam_rgb.setPreviewSize(640, 480)
cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
cam_rgb.setInterleaved(False)
cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)

# Define the XLink output to send frames to the host
xout_rgb = pipeline.createXLinkOut()
xout_rgb.setStreamName("rgb")
cam_rgb.preview.link(xout_rgb.input)

# Start the pipeline
with dai.Device(pipeline) as device:
    rgb_queue = device.getOutputQueue(name="rgb", maxSize=1, blocking=False)

    while True:
        in_rgb = rgb_queue.get()
        if in_rgb is None:
            continue
        
        frame = in_rgb.getCvFrame()
        
        # Display instructions
        cv2.putText(frame, "Press 'c' to capture image, 'q' to quit", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        # Display the frame
        cv2.imshow('OAK-D Lite Camera Feed', frame)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('c'):
            print("[INFO] Capturing image...")
            captured_image = frame.copy()
            cv2.imwrite('captured_board.jpg', captured_image)
            print("[INFO] Image saved as 'captured_board.jpg'")
            
            # YOLO Detection on the captured image
            results = model(captured_image)
            boxes = results.xyxy[0].cpu().numpy()
            
            if len(boxes) > 0:
                x1, y1, x2, y2, conf, cls = boxes[0]
                corners = np.array([
                    [x1, y1],
                    [x2, y1],
                    [x2, y2],
                    [x1, y2]
                ], dtype=np.float32)
                
                rect = np.array([
                    [0, 0],
                    [640, 0],
                    [640, 640],
                    [0, 640]
                ], dtype=np.float32)
                
                # Apply perspective correction
                M = cv2.getPerspectiveTransform(corners, rect)
                warped = cv2.warpPerspective(captured_image, M, (640, 640))
                cv2.imwrite('warped_board.jpg', warped)
                print("[INFO] Perspective correction applied and saved as 'warped_board.jpg'")
                cv2.imshow('Warped Board', warped)
            else:
                print("[ERROR] Failed to detect board for perspective correction.")
        
        if key == ord('q'):
            break

    cv2.destroyAllWindows()
