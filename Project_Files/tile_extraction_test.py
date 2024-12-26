import cv2
import numpy as np

# Load the test image
image = cv2.imread('test_images/board_test.jpg')



def correct_perspective(frame, contour):
    """Apply perspective transform to focus on the white grid area."""
    peri = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
    
    if len(approx) == 4:
        src_pts = np.array([point[0] for point in approx], dtype=np.float32)
        width, height = 600, 600  # Standardized grid size
        dst_pts = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype=np.float32)
        matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
        warped = cv2.warpPerspective(frame, matrix, (width, height))
        return warped
    else:
        print("Contour approximation failed. Points detected:", len(approx))
    return None


def extract_tiles(frame):
    """Detect Scrabble board and divide it into a 15x15 grid."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Refined red mask to isolate the board
    lower_red1 = np.array([0, 70, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 70, 50])
    upper_red2 = np.array([180, 255, 255])
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = mask1 | mask2
    
    # Show intermediate mask for debugging
    cv2.imshow('Red Mask', mask)
    cv2.waitKey(0)
    
    # Find contours and focus on the largest one
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        cv2.drawContours(frame, [largest_contour], -1, (0, 255, 0), 2)
        cv2.imshow('Contour Detection', frame)
        cv2.waitKey(0)
        
        warped = correct_perspective(frame, largest_contour)
        
        if warped is not None:
            height, width = 600, 600
            tile_height = height // 15
            tile_width = width // 15
            
            for i in range(15):
                for j in range(15):
                    cv2.rectangle(warped, 
                                  (j * tile_width, i * tile_height),
                                  ((j + 1) * tile_width, (i + 1) * tile_height),
                                  (255, 0, 0), 1)
            
            cv2.imshow('Tile Grid', warped)
            cv2.waitKey(0)
            return warped
        else:
            print("Perspective transformation failed.")
    else:
        print("No valid contours detected.")
    return frame


# Run the extraction
grid_frame = extract_tiles(image)
cv2.imshow('Final Output', grid_frame)
cv2.waitKey(0)
cv2.destroyAllWindows()
