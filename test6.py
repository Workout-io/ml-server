# main.py
import cv2
import time
from exercises.Lunges import process_frame_lunge

# Path to the video file
video_path = "tmp/videos/test_1.mp4"

# Initialize video capture with the video file
cap = cv2.VideoCapture(video_path)

# Get video properties
fps = cap.get(cv2.CAP_PROP_FPS)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
duration = frame_count / fps

print(f"Duration: {duration:.2f} seconds")

# Initialize variables for FPS calculation
prev_frame_time = 0
new_frame_time = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Calculate the time taken to process each frame
    new_frame_time = time.time()

    # Process the frame for squat detection
    image = process_frame_lunge(frame)

    # Calculate FPS
    fps = 1 / (new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time

    # Display FPS on the frame
    cv2.putText(
        image,
        f"FPS: {int(fps)}",
        (10, 150),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )

    # Display the frame
    cv2.imshow("Lunges Counter", image)

    if cv2.waitKey(10) & 0xFF == ord("q"):
        break

# Release the capture
cap.release()
cv2.destroyAllWindows()
