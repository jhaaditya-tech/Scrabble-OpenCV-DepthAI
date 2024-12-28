import torch
import cv2

def detect_board(image):
    model = torch.hub.load('ultralytics/yolov5', 'custom', path='runs/train/exp2/weights/best.pt', force_reload=True)
    results = model(image)
    detections = results.pandas().xyxy[0]

    if len(detections) > 0:
        x_min, y_min = int(detections.iloc[0]['xmin']), int(detections.iloc[0]['ymin'])
        x_max, y_max = int(detections.iloc[0]['xmax']), int(detections.iloc[0]['ymax'])
        return [(x_min, y_min), (x_max, y_min), (x_max, y_max), (x_min, y_max)]
    else:
        return None
