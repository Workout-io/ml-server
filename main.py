# main.py
import cv2
from exercises.Pushup2 import process_frame
from exercises.Squat2 import process_frame_squatjump

# Path to the video file
video_path = "tmp/videos/two.mp4"

# Initialize video capture with the video file
cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Process frame using pushup.py
    processed_frame = process_frame(frame)

    # Display the frame
    cv2.imshow("Push-up Counter", processed_frame)

    # # Display the frame
    # cv2.imshow("Squat Counter", image)

    if cv2.waitKey(10) & 0xFF == ord("q"):
        break

# Release the capture
cap.release()
cv2.destroyAllWindows()
