# capture_and_process.py
# Main script to capture and process the image

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
