q
import depthai as dai
import cv2

# Create pipeline
pipeline = dai.Pipeline()

# Define camera node
cam_rgb = pipeline.create(dai.node.ColorCamera)
xout_rgb = pipeline.create(dai.node.XLinkOut)
xout_rgb.setStreamName("rgb")
cam_rgb.isp.link(xout_rgb.input)

# Start camera feed
with dai.Device(pipeline) as device:
    rgb_queue = device.getOutputQueue(name="rgb", maxSize=1, blocking=False)
    while True:
        frame = rgb_queue.get().getCvFrame()
        cv2.imshow("Camera Test", frame)
        if cv2.waitKey(1) == ord('q'):
            break

cv2.destroyAllWindows()
