import torch
import cv2
import depthai as dai

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'custom',
                       path='/Users/adityajha/Documents/CV Thesis/Project Codes/Scrabble-OpenCV-DepthAI/yolov5_project/yolov5/runs/train/exp2/weights/best.pt',
                       force_reload=True)

# Initialize DepthAI pipeline
pipeline = dai.Pipeline()

# Define the OAK-D Lite camera
cam_rgb = pipeline.create(dai.node.ColorCamera)
cam_rgb.setPreviewSize(640, 480)
cam_rgb.setInterleaved(False)
cam_rgb.setFps(30)

# Output queue for the camera
xout_rgb = pipeline.create(dai.node.XLinkOut)
xout_rgb.setStreamName("rgb")
cam_rgb.preview.link(xout_rgb.input)

# Start pipeline
with dai.Device(pipeline) as device:
    rgb_queue = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
    while True:
        frame = rgb_queue.get().getCvFrame()
        if frame is None:
            continue
        
        # Perform inference
        results = model(frame)
        
        # Render detections
        annotated_frame = results.render()[0]
        
        # Display the frame
        cv2.imshow("OAK-D Lite Detection", annotated_frame)
        
        if cv2.waitKey(1) == ord('q'):
            break

cv2.destroyAllWindows()
