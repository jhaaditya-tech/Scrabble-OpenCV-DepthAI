import cv2  # For displaying and handling image frames
import torch  # For YOLO model inference
import depthai as dai  # For working with the OAK-D Lite camera
from pipeline.camera_utils import initialize_camera, capture_image
from pipeline.image_processing import process_image


def main():
    """
    Main execution function to capture, process, and warp the Scrabble board.
    """
    print("🚀 Initializing Camera...")
    device, rgb_queue = initialize_camera()

    while True:
        in_rgb = rgb_queue.get()
        if in_rgb is None:
            continue

        frame = in_rgb.getCvFrame()
        cv2.putText(frame, "Press 'c' to capture image, 'q' to quit", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow('OAK-D Lite Camera Feed', frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('c'):
            print("📸 Capturing image...")
            image_path = capture_image(frame)
            print(f"✅ Image captured and saved at {image_path}")

            print("🛠️ Processing image for playable area...")
            try:
                warped_path = process_image(image_path)
                print(f"✅ Warped image saved at {warped_path}")
            except ValueError as e:
                print(f"❌ Error: {e}")
        
        if key == ord('q'):
            print("👋 Exiting...")
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
