import sys
import os

# Add the project root directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pipeline.camera_utils import capture_image
from pipeline.image_processing import process_image


def main():
    print("🚀 Step 1: Capturing Image...")
    image_path = capture_image()
    print(f"✅ Image saved at: {image_path}")

    print("🚀 Step 2: Processing Image...")
    try:
        processed_image = process_image(image_path)
        print("✅ Image processed successfully.")
    except Exception as e:
        print(f"❌ Error processing image: {e}")


if __name__ == "__main__":
    main()
