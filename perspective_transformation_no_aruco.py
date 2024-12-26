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

            # Debugging: Display the red mask to ensure it captures the red border
            cv2.imshow("Red Mask", mask)
            cv2.waitKey(0)

            # Apply Canny edge detection to the mask to help contour detection
            edges = cv2.Canny(mask, 50, 150)

            # Debugging: Display the edges to see if they outline the board
            cv2.imshow("Edges", edges)
            cv2.waitKey(0)

            # Find contours on the edges
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Debugging: Print the number of contours detected
            print(f"Number of contours found: {len(contours)}")

            # Filter contours by area to focus on larger contours
            contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 5000]  # Adjust the area threshold if needed

            # Debugging: Print the area of each contour to check sizes
            for i, cnt in enumerate(contours):
                area = cv2.contourArea(cnt)
                print(f"Contour {i} area: {area}")

            for contour in contours:
                epsilon = 0.02 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)

                # Ensure the contour is a quadrilateral (4 points)
                if len(approx) == 4:
                    print("Found a quadrilateral contour for the board.")

                    # Debug: Draw detected points on the original image
                    for point in approx:
                        cv2.circle(frame, tuple(point[0]), 5, (0, 255, 0), -1)

                    # Display the image with detected points for verification
                    cv2.imshow("Detected Points", frame)
                    cv2.waitKey(0)  # Pause to review detected points

                    # Reorder and map the corner points to a square output
                    src_pts = np.array([point[0] for point in approx], dtype="float32")
                    dst_pts = np.array([[0, 0], [300, 0], [300, 300], [0, 300]], dtype="float32")  # Adjust as needed

                    # Compute the perspective transform matrix and apply warp to the original frame
                    matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
                    warped = cv2.warpPerspective(frame, matrix, (300, 300))  # Output size

                    # Display the perspective-transformed original image
                    cv2.imshow("Warped Board (Original Image)", warped)
                    cv2.waitKey(0)  # Pause until a key is pressed
                    break
                else:
                    print("No quadrilateral contour detected for the board.")

        # Exit on 'q' key
        elif key == ord("q"):
            print("Exiting.")
            break

    cv2.destroyAllWindows()
