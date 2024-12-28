import os
import cv2

def save_image(image, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, image)
    print(f"✅ Image saved at: {path}")
