import cv2
import depthai as dai
import os

# Directory to save images
SAVE_DIR = "captured_images"
os.makedirs(SAVE_DIR, exist_ok=True)

# Function to get the next available filename
def get_next_filename(directory, prefix="image", extension=".jpg"):
    index = 1
    while True:
        filename = f"{prefix}{index:03d}{extension}"
        if not os.path.exists(os.path.join(directory, filename)):
            return os.path.join(directory, filename)
        index += 1

# Start pipeline
pipeline = dai.Pipeline()

# Create RGB camera node
cam_rgb = pipeline.create(dai.node.ColorCamera)
cam_rgb.setPreviewSize(1920, 1080)  # Set resolution
cam_rgb.setInterleaved(False)
cam_rgb.setFps(30)

# Output link to host
xout_rgb = pipeline.create(dai.node.XLinkOut)
xout_rgb.setStreamName("video")
cam_rgb.preview.link(xout_rgb.input)

# Connect to device
with dai.Device(pipeline) as device:
    print("Oak-D Lite Camera connected. Press 'c' to capture an image, 'q' to quit.")
    
    video_queue = device.getOutputQueue(name="video", maxSize=1, blocking=False)
    
    while True:
        in_frame = video_queue.get()
        frame = in_frame.getCvFrame()
        
        if frame is not None:
            cv2.imshow("Camera Preview", frame)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('c'):  # Press 'c' to capture
            filename = get_next_filename(SAVE_DIR)
            cv2.imwrite(filename, frame)
            print(f"Image saved: {filename}")
        
        if key == ord('q'):  # Press 'q' to quit
            break

cv2.destroyAllWindows()
