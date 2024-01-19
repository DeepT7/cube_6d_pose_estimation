import cv2

# Create a VideoCapture object
cap = cv2.VideoCapture(1)

# Frame number
frame_number = 7

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Display the resulting frame
    cv2.imshow('frame', frame)

    # Save the frame on pressing 's'
    if cv2.waitKey(1) & 0xFF == ord('s'):
        cv2.imwrite('frame{}.png'.format(frame_number), frame)
        frame_number += 1

    # Break the loop on pressing 'q'
    elif cv2.waitKey(1) & 0xFF == ord('q'):
        break

# After the loop, release the VideoCapture object
cap.release()

# Close all OpenCV windows
cv2.destroyAllWindows()
