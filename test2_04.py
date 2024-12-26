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

            # Convert to HSV and apply a mask to isolate white color (white border)
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            # Define the HSV range for white
            lower_white = np.array([0, 0, 200])    # Low saturation, high brightness
            upper_white = np.array([180, 55, 255])  # Adjust for higher brightness if needed

            # Create mask for the white color
            mask = cv2.inRange(hsv, lower_white, upper_white)

            # Find contours on the mask to detect the outer white border
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 5000]  # Filter by area

            found_border = False
            for contour in contours:
                epsilon = 0.02 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)

                # Ensure the contour is a quadrilateral (4 points)
                if len(approx) == 4:
                    found_border = True
                    print("Found the outer white border contour.")

                    # Create a mask with only the outer white border filled
                    border_mask = np.zeros(mask.shape, dtype=np.uint8)
                    cv2.drawContours(border_mask, [approx], -1, (255), thickness=cv2.FILLED)

                    # Erode the mask inward to create a smaller interior mask for the inner edge of the white border
                    erosion_kernel = np.ones((15, 15), np.uint8)  # Adjust kernel size based on border thickness
                    inner_mask = cv2.erode(border_mask, erosion_kernel, iterations=1)

                    # Find contours again on this eroded mask to get the inner contour
                    inner_contours, _ = cv2.findContours(inner_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    inner_contours = [cnt for cnt in inner_contours if cv2.contourArea(cnt) > 5000]  # Filter by area

                    for inner_contour in inner_contours:
                        epsilon_inner = 0.02 * cv2.arcLength(inner_contour, True)
                        inner_approx = cv2.approxPolyDP(inner_contour, epsilon_inner, True)

                        # Ensure the inner contour is also a quadrilateral
                        if len(inner_approx) == 4:
                            print("Found the inner contour along the edge of the white border.")

                            # Draw detected inner points on the original image for verification
                            for point in inner_approx:
                                cv2.circle(frame, tuple(point[0]), 5, (0, 255, 0), -1)

                            # Display the image with detected inner points
                            cv2.imshow("Detected Inner White Border Points", frame)
                            cv2.waitKey(0)  # Pause to review detected points

                            # Use these inner points for perspective transformation
                            src_pts = np.array([point[0] for point in inner_approx], dtype="float32")

                            # Define the output dimensions (adjust according to the board's actual aspect ratio)
                            width = 500  # Example width for the interior of the board
                            height = 500  # Example height for the interior of the board
                            dst_pts = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype="float32")

                            # Compute the perspective transform matrix and apply warp to the original frame
                            matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
                            warped = cv2.warpPerspective(frame, matrix, (width, height))  # Output size

                            # Display the perspective-transformed interior of the board
                            cv2.imshow("Warped Board Interior", warped)
                            cv2.waitKey(0)  # Pause until a key is pressed
                            break
                    break

            if not found_border:
                print("No quadrilateral contour detected for the white border.")

        elif key == ord("q"):  # Quit program
            print("Exiting.")
            break

    cv2.destroyAllWindows()
