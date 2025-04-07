import cv2
import numpy as np

# Load the video
cap = cv2.VideoCapture('path_to_tunnel_video.mp4')

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Resize to lower resolution
    frame = cv2.resize(frame, (320, 240))

    # Add Gaussian noise
    noise = np.random.normal(0, 25, frame.shape).astype(np.uint8)
    frame = cv2.add(frame, noise)

    # Adjust brightness and contrast
    frame = cv2.convertScaleAbs(frame, alpha=0.5, beta=10)

    # Display the frame
    cv2.imshow('Simulated Poor CCTV', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
