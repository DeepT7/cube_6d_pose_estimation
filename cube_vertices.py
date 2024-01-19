import cv2
import numpy as np

def detect_vertex(img_path):
    # Load image
    img = cv2.imread(img_path)

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply corner detection
    corners = cv2.goodFeaturesToTrack(gray, 140, 0.03, 10)
    corners = np.int0(corners)

    # Filter corners based on distance (at least 50 pixels apart)
    filtered_corners = []
    min_distance = 20

    for i in range(corners.shape[0]):
        x1, y1 = corners[i].ravel()

        # Check the distance with previously selected corners
        if all(np.sqrt((x1 - x2)**2 + (y1 - y2)**2) > min_distance for x2, y2 in filtered_corners):
            filtered_corners.append((x1, y1))

    # Draw circles on detected corners
    for x, y in filtered_corners:
        cv2.circle(img, (x, y), 8, 255, -1)

    # Display image with detected corners
    cv2.imwrite("frame10.png", img)
    cv2.imshow('Detected Corners', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
# detect_vertex('result/BIPED2CLASSIC/fused/frame10.png')