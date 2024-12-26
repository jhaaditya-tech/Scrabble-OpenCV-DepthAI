import cv2
import numpy as np
import depthai as dai

# Initialize the Oak-D Lite pipeline
pipeline = dai.Pipeline()

# Set up RGB camera on Oak-D Lite
cam_rgb = pipeline.create(dai.node.ColorCamera)
cam_rgb.setPreviewSize(640, 480)
cam_rgb.setInterleaved(False)
cam_rgb.setBoardSocket(dai.CameraBoardSocket.RGB)

# Set up XLink output for RGB camera
xout_rgb = pipeline.create(dai.node.XLinkOut)
xout_rgb.setStreamName("rgb")
cam_rgb.preview.link(xout_rgb.input)

# Start the pipeline
with dai.Device(pipeline) as device:
    rgb_queue = device.getOutputQueue(name="rgb", maxSize=1, blocking=False)

    print("Press the spacebar to capture an image.")

    while True:
        # Get RGB frame from the Oak-D Lite
        in_rgb = rgb_queue.get()
        frame = in_rgb.getCvFrame()

        # Display the live RGB feed
        cv2.imshow("RGB Feed", frame)

        # Capture an image on spacebar press
        key = cv2.waitKey(1)
        if key == ord(" "):  # Spacebar to capture
            print("Image captured.")

            # Convert to HSV and apply a mask to isolate red color (Scrabble board border)
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            
            # Define the red color ranges for masking
            lower_red1 = np.array([0, 100, 50])
            upper_red1 = np.array([10, 255, 255])
            lower_red2 = np.array([160, 100, 50])
            upper_red2 = np.array([180, 255, 255])
            
            # Combine both red masks
            mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
            mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
            mask = mask1 | mask2

            # Find contours on the mask
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Filter contours by area to focus on larger contours
            contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 5000]  # Adjust the area threshold if needed

            found_board = False
            for contour in contours:
                epsilon = 0.02 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)

                # Ensure the contour is a quadrilateral (4 points)
                if len(approx) == 4:
                    found_board = True
                    print("Found a quadrilateral contour for the board.")

                    # Draw detected points on the original image for verification
                    for point in approx:
                        cv2.circle(frame, tuple(point[0]), 5, (0, 255, 0), -1)

                    # Display the image with detected points
                    cv2.imshow("Detected Points", frame)
                    cv2.waitKey(0)  # Pause to review detected points

                    # Custom inward adjustments based on the thickness of each border side
                    # Adjust these values according to your specific board
                    top_left_adjust = (20, 20)       # Adjust top-left corner inward
                    top_right_adjust = (-10, 20)     # Adjust top-right corner inward
                    bottom_right_adjust = (-10, -10) # Adjust bottom-right corner inward
                    bottom_left_adjust = (20, -10)   # Adjust bottom-left corner inward

                    # Apply the custom adjustments to each corner point
                    src_pts = np.array([
                        [approx[0][0][0] + top_left_adjust[0], approx[0][0][1] + top_left_adjust[1]],     # Top-left
                        [approx[1][0][0] + top_right_adjust[0], approx[1][0][1] + top_right_adjust[1]],   # Top-right
                        [approx[2][0][0] + bottom_right_adjust[0], approx[2][0][1] + bottom_right_adjust[1]], # Bottom-right
                        [approx[3][0][0] + bottom_left_adjust[0], approx[3][0][1] + bottom_left_adjust[1]]  # Bottom-left
                    ], dtype="float32")

                    # Define the output dimensions (adjust according to the board's actual aspect ratio)
                    width = 500  # Example width for the interior of the board
                    height = 500  # Example height for the interior of the board
                    dst_pts = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype="float32")

                    # Compute the perspective transform matrix and apply warp to the original frame
                    matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
                    warped = cv2.warpPerspective(frame, matrix, (width, height))  # Output size

                    # Display the perspective-transformed original image
                    cv2.imshow("Warped Board Interior", warped)
                    cv2.waitKey(0)  # Pause until a key is pressed
                    break

            if not found_board:
                print("No quadrilateral contour detected for the board interior.")

        # Exit on 'q' key
        elif key == ord("q"):
            print("Exiting.")
            break

    cv2.destroyAllWindows()
