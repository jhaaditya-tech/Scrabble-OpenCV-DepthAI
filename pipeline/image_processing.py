import cv2
import numpy as np
import torch
from pipeline.config import WARPED_SAVE_PATH, MODEL_PATH

# Load YOLOv5 Model
model = torch.hub.load('ultralytics/yolov5', 'custom', path=MODEL_PATH)


def process_image(image_path):
    """
    Processes the captured image to detect and warp the Scrabble board using YOLOv5.

    Args:
        image_path (str): Path to the captured image.
    
    Returns:
        str: Path to the warped image of the detected board.
    """
    # Load the captured image
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"❌ Failed to load image at {image_path}")

    # YOLOv5 Inference
    results = model(image)
    boxes = results.xyxy[0].cpu().numpy()
    
    if len(boxes) == 0:
        raise ValueError("❌ YOLO failed to detect any board in the image.")
    
    # Assuming the first detected box is the Scrabble board
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
    
    # Apply Perspective Transformation
    M = cv2.getPerspectiveTransform(corners, rect)
    warped = cv2.warpPerspective(image, M, (640, 640))
    
    # Save the warped image
    cv2.imwrite(WARPED_SAVE_PATH, warped)
    print(f"✅ Warped image saved at: {WARPED_SAVE_PATH}")
    
    # Display the warped image for verification
    cv2.imshow("Warped Board", warped)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return WARPED_SAVE_PATH
