import cv2
import cv2.aruco as aruco
import os

# Define the dictionary and marker size
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
marker_size = 200  # Size of each marker in pixels

# Define marker IDs for each corner
marker_ids = {
    "top_left": 10,
    "top_right": 20,
    "bottom_left": 30,
    "bottom_right": 40
}

# Directory to save the marker images
output_dir = "markers"
os.makedirs(output_dir, exist_ok=True)

# Generate and save each marker
for position, marker_id in marker_ids.items():
    # Create an image for the marker
    marker_image = aruco.generateImageMarker(aruco_dict, marker_id, marker_size)
    
    # Save the marker image to a file
    file_path = os.path.join(output_dir, f"{position}_marker_{marker_id}.png")
    cv2.imwrite(file_path, marker_image)
    
    print(f"Saved {position} marker with ID {marker_id} to {file_path}")

print("All corner markers generated and saved.")
