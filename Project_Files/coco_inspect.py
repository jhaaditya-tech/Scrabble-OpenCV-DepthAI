
import json

# Path to your COCO annotations
coco_json_path = 'dataset/annotations/annotations.json'

with open(coco_json_path, 'r') as f:
    coco_data = json.load(f)

# Check the structure of one annotation
print(coco_data['annotations'][0])
