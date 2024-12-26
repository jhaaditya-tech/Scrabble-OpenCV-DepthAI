#Purpose: Displaying a live RGB Video feed from the Oak-D Lite Camera using DepthAI

import depthai as dai
import cv2
import sys

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

# Start the pipeline
with dai.Device(pipeline) as device:
    rgb_queue = device.getOutputQueue(name="rgb", maxSize=1, blocking=False)
    print("Camera Resolution:", cam_rgb.getResolution())

    try:
        while True:
            # Clear queue to avoid latency
            while rgb_queue.has():
                frame = rgb_queue.get().getCvFrame()
            
            frame = rgb_queue.get().getCvFrame()
            cv2.imshow("Low-Latency Camera Feed", frame)
            if cv2.waitKey(1) == ord('q'):
                break
    except KeyboardInterrupt:
        print("Stream stopped by user")
    finally:
        cv2.destroyAllWindows()
        sys.exit()



"""
File Description

> Sets Up Pipeline: Initialized a communication channel between the OAK-D Lite Camera and your program
> Configures Camera : Setting Resoultion to 1280*720 disables interleaving and ensuring BGR color order
> Links Outut Stream: Connects the camera output to a stream named rgb
> Displays Live Feed: Captures Frames in real time and displays using OpenCV
> Exit mechanism: Press 'q' to close the video feed window.

"""