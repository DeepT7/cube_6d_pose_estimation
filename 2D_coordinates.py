import cv2 

coordinates = []

image = cv2.imread("edges/frame0.png")

def click_event(event, x, y, flags, data):
    if event == cv2.EVENT_LBUTTONDOWN:
        coordinates.append((x, y))
        cv2.circle(image, (x, y), 5, 255, -1)
        print(f"Clicked at : ({x}, {y})")


cv2.namedWindow('image')
cv2.setMouseCallback("image", click_event)
cv2.imshow("image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite("frame10.png", image)
print(coordinates)
