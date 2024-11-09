import cv2
import depthai as dai

# Set up the DepthAI pipeline
pipeline = dai.Pipeline()

# Set up RGB camera node
cam_rgb = pipeline.create(dai.node.ColorCamera)
cam_rgb.setPreviewSize(640, 480)
cam_rgb.setInterleaved(False)

# Set up output node for RGB data
xout_rgb = pipeline.create(dai.node.XLinkOut)
xout_rgb.setStreamName("rgb")

# Link RGB camera output to the XLinkOut
cam_rgb.preview.link(xout_rgb.input)

# Run pipeline
with dai.Device(pipeline) as device:
    q_rgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)

    while True:
        in_rgb = q_rgb.tryGet()

        if in_rgb:
            frame_rgb = in_rgb.getCvFrame()
            cv2.imshow("RGB", frame_rgb)

        if cv2.waitKey(1) == ord('q'):
            break

cv2.destroyAllWindows()
