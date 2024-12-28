import depthai as dai
import cv2
import os
from pipeline.config import CAPTURED_IMAGE_PATH

def initialize_camera():
    """
    Initializes the OAK-D Lite camera and sets up the RGB stream.
    Returns:
        tuple: (device, rgb_queue)
    """
    pipeline = dai.Pipeline()
    cam_rgb = pipeline.createColorCamera()
    cam_rgb.setPreviewSize(640, 480)
    cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    cam_rgb.setInterleaved(False)
    cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)

    xout_rgb = pipeline.createXLinkOut()
    xout_rgb.setStreamName("rgb")
    cam_rgb.preview.link(xout_rgb.input)

    device = dai.Device(pipeline)
    rgb_queue = device.getOutputQueue(name="rgb", maxSize=1, blocking=False)
    print("✅ Camera initialized successfully.")
    return device, rgb_queue


def capture_image(frame):
    """
    Saves the captured image.
    Args:
        frame (np.ndarray): Captured camera frame.
    Returns:
        str: Path to the saved image.
    """
    if not os.path.exists('images'):
        os.makedirs('images')
    
    cv2.imwrite(CAPTURED_IMAGE_PATH, frame)
    return CAPTURED_IMAGE_PATH
