# This situp function
import cv2
import mediapipe as mp
import numpy as np
from utils.utils import calculate_angle

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Initialize sit-up counter and correctness tracker
correct_situps = 0
incorrect_situps = 0
last_position = None
screenshot_counter = 0


def process_frame_situp(frame):
    global correct_situps, incorrect_situps, last_position, screenshot_counter

    feedback = ""  # Initialize feedback variable
    color = (0, 0, 0)  # Initialize color variable

    # Convert the frame to RGB
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False

    # Process the frame to get pose landmarks
    results = pose.process(image)

    # Draw landmarks on the frame
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if results.pose_landmarks:
        # Get coordinates
        landmarks = results.pose_landmarks.landmark

        # Get relevant landmarks for sit-up detection
        shoulder = [
            landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
            landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y,
        ]
        hip = [
            landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
            landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y,
        ]
        knee = [
            landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
            landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y,
        ]

        # Calculate angles
        hip_angle = calculate_angle(shoulder, hip, knee)

        # Sit-up counter logic
        if hip_angle > 160:  # Person lying down
            if last_position != "down":
                last_position = "down"
                feedback = "Start Position"
                color = (0, 255, 0)  # Green for correct
        elif hip_angle < 90:  # Person sitting up
            if last_position == "down":
                last_position = "up"
                if 60 <= hip_angle <= 90:
                    correct_situps += 1
                    feedback = "Good Sit-up"
                    color = (0, 255, 0)  # Green for correct
                else:
                    incorrect_situps += 1
                    feedback = "Try Again"
                    color = (0, 0, 255)  # Red for incorrect

        # Draw pose landmarks on the image
        mp.solutions.drawing_utils.draw_landmarks(
            image,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp.solutions.drawing_styles.get_default_pose_landmarks_style(),
        )

        # Display feedback on the frame
        cv2.putText(
            image,
            feedback,
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            color,
            2,
            cv2.LINE_AA,
        )

    # Display sit-up count and correctness count
    total_situps = correct_situps + incorrect_situps
    cv2.putText(
        image,
        f"Total Sit-ups: {total_situps}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        image,
        f"Correct: {correct_situps}",
        (10, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        image,
        f"Incorrect: {incorrect_situps}",
        (10, 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 255),
        2,
        cv2.LINE_AA,
    )

    return image
