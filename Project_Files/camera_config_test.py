#Purpose: to verify and print camera configuration details (e.g. resolution, supposrted frame rates, etc)

import depthai as dai

# Create pipeline
pipeline = dai.Pipeline()

# Create camera node
cam_rgb = pipeline.create(dai.node.ColorCamera)

# Print available configuration details
print("Supported Video Sizes:", cam_rgb.getResolution())
print("Supported Frame Rates:", cam_rgb.getFps())
print("Default Color Order:", cam_rgb.getColorOrder())

"""
Camera Result:
Supported Video Sizes: SensorResolution.THE_1080_P
Supported Frame Rates: 30.0
Default Color Order: ColorOrder.BGR

Notes:
We will use a resolution of 1920*1080 for maximum image quality
Frame rates will be set to 30FPS for smooth live streaming and analysis
The BGR color order is compatible with OpenCV, simplifying our image processing pipelines.

"""