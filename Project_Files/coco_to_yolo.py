import json
import os

# Paths
coco_json_path = 'dataset/annotations/annotations.json'
output_dir = 'dataset/labels'
images_dir = 'dataset/images'

os.makedirs(output_dir, exist_ok=True)

# Load COCO JSON
with open(coco_json_path, 'r') as f:
    coco_data = json.load(f)

# Map image IDs to filenames
image_id_to_filename = {image['id']: image['file_name'] for image in coco_data['images']}

# Map categories to ensure proper indexing (YOLO starts from 0)
category_mapping = {category['id']: i for i, category in enumerate(coco_data['categories'])}

# Process annotations
for annotation in coco_data['annotations']:
    image_id = annotation['image_id']
    bbox = annotation.get('bbox', None)
    category_id = annotation.get('category_id', None)

    if bbox is None or category_id is None:
        print(f"Skipping annotation without bbox or category_id: {annotation}")
        continue

    # Adjust category index (YOLO expects 0-based indexing)
    category_index = category_mapping.get(category_id, -1)
    if category_index == -1:
        print(f"Skipping unknown category_id: {category_id}")
        continue

    x, y, width, height = bbox
    image_filename = image_id_to_filename.get(image_id, None)

    if not image_filename:
        print(f"Skipping annotation with missing image_id: {image_id}")
        continue

    # Get image dimensions
    img_data = next((img for img in coco_data['images'] if img['id'] == image_id), None)
    if img_data is None:
        print(f"Skipping annotation with missing image data: {image_id}")
        continue

    img_width = img_data['width']
    img_height = img_data['height']

    # Normalize YOLO coordinates
    x_center = (x + width / 2) / img_width
    y_center = (y + height / 2) / img_height
    width /= img_width
    height /= img_height

    # Write to YOLO label file
    label_filename = os.path.splitext(image_filename)[0] + '.txt'
    label_path = os.path.join(output_dir, label_filename)

    with open(label_path, 'a') as label_file:
        label_file.write(f"{category_index} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

print("✅ Conversion completed! YOLO labels are saved in 'dataset/labels'.")
