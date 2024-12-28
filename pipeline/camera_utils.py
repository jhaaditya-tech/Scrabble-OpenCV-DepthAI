import cv2
import depthai as dai
import os

def capture_image(save_dir='images'):
    os.makedirs(save_dir, exist_ok=True)
    image_path = os.path.join(save_dir, 'captured_image.jpg')

    pipeline = dai.Pipeline()
    cam_rgb = pipeline.create(dai.node.ColorCamera)
    xout_rgb = pipeline.create(dai.node.XLinkOut)
    xout_rgb.setStreamName("rgb")
    cam_rgb.preview.link(xout_rgb.input)

    with dai.Device(pipeline) as device:
        q_rgb = device.getOutputQueue("rgb")
        frame = q_rgb.get().getCvFrame()
        cv2.imwrite(image_path, frame)
        print(f"✅ Image Captured and Saved to {image_path}")
    
    return image_path
