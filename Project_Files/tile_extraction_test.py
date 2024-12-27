import cv2
import numpy as np


def apply_red_mask(frame):
    """Apply red mask to detect both outer and inner red areas."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Outer Red Mask
    lower_red1 = np.array([0, 70, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 70, 50])
    upper_red2 = np.array([180, 255, 255])

    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    combined_mask = cv2.bitwise_or(mask1, mask2)

    combined_mask = cv2.GaussianBlur(combined_mask, (5, 5), 0)

    # Debug: Show Mask
    cv2.imshow('Refined Red Mask', combined_mask)
    cv2.waitKey(0)

    return combined_mask


def find_largest_rectangle_contour(mask, frame):
    """Find the largest valid rectangular contour."""
    edges = cv2.Canny(mask, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    debug_contours = frame.copy()
    cv2.drawContours(debug_contours, contours, -1, (0, 255, 0), 2)
    cv2.imshow('All Contours Debug', debug_contours)
    cv2.waitKey(0)

    for contour in sorted(contours, key=cv2.contourArea, reverse=True):
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)

        if len(approx) == 4:
            debug_frame = frame.copy()
            cv2.drawContours(debug_frame, [approx], -1, (0, 0, 255), 3)
            cv2.imshow('Selected Contour', debug_frame)
            cv2.waitKey(0)
            return approx

    print("❌ No valid rectangular contour found.")
    return None


def correct_perspective(frame, contour):
    """Correct the perspective using the detected contour."""
    src_pts = np.array([point[0] for point in contour], dtype=np.float32)
    width, height = 600, 600  # Standard grid size
    dst_pts = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped = cv2.warpPerspective(frame, matrix, (width, height))

    cv2.imshow('Warped Perspective', warped)
    cv2.waitKey(0)

    return warped


def draw_grid_on_playable_area(warped):
    """Draw a 15x15 grid over the playable area."""
    height, width = warped.shape[:2]
    tile_height = height // 15
    tile_width = width // 15

    for i in range(15):
        for j in range(15):
            cv2.rectangle(
                warped,
                (j * tile_width, i * tile_height),
                ((j + 1) * tile_width, (i + 1) * tile_height),
                (255, 0, 0), 1
            )
    cv2.imshow('Playable Area with Grid', warped)
    cv2.waitKey(0)


# Main Execution
image = cv2.imread('test_images/board_test_1.jpg')

if image is not None:
    red_mask = apply_red_mask(image)
    largest_contour = find_largest_rectangle_contour(red_mask, image)

    if largest_contour is not None:
        warped = correct_perspective(image, largest_contour)
        if warped is not None:
            draw_grid_on_playable_area(warped)
    else:
        print("❌ Failed to detect a valid rectangular contour.")

cv2.destroyAllWindows()
