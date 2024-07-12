import cv2
import mediapipe as mp
import numpy as np
from utils.utils import (
    calculate_angle,
)  # Assuming you have a utility function to calculate angles

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Initialize push-up counter and correctness tracker
counter = 0
correct_pushups = 0
incorrect_pushups = 0
screenshot_counter = 0
stage = None


def process_frame_pushup(frame):
    global counter, correct_pushups, incorrect_pushups, screenshot_counter, stage

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

        # Get relevant landmarks for push-up detection
        shoulder = [
            landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
            landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y,
        ]
        elbow = [
            landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
            landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y,
        ]
        wrist = [
            landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
            landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y,
        ]

        # Calculate angles
        elbow_angle = calculate_angle(shoulder, elbow, wrist)

        # Visualize angles
        cv2.putText(
            image,
            f"Elbow: {int(elbow_angle)}",
            tuple(np.multiply(elbow, [640, 480]).astype(int)),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        # Draw lines for visualization
        cv2.line(
            image,
            tuple(np.multiply(shoulder, [640, 480]).astype(int)),
            tuple(np.multiply(elbow, [640, 480]).astype(int)),
            (0, 255, 0),
            2,
        )
        cv2.line(
            image,
            tuple(np.multiply(elbow, [640, 480]).astype(int)),
            tuple(np.multiply(wrist, [640, 480]).astype(int)),
            (0, 255, 0),
            2,
        )

        # Push-up counter logic
        if elbow_angle > 160:
            stage = "up"
        if elbow_angle < 90 and stage == "up":
            stage = "down"
            counter += 1
            # Determine if push-up is correct or incorrect
            if 80 <= elbow_angle <= 100:
                correct_pushups += 1
                feedback = "Good Push-up"
                color = (0, 255, 0)  # Green for correct
            else:
                incorrect_pushups += 1
                feedback = "Try Again"
                color = (0, 255, 255)  # Yellow for less strict feedback
                # Draw lines indicating error
                cv2.line(
                    image,
                    tuple(np.multiply(shoulder, [640, 480]).astype(int)),
                    tuple(np.multiply(elbow, [640, 480]).astype(int)),
                    (0, 0, 255),
                    2,
                )
                cv2.line(
                    image,
                    tuple(np.multiply(elbow, [640, 480]).astype(int)),
                    tuple(np.multiply(wrist, [640, 480]).astype(int)),
                    (0, 0, 255),
                    2,
                )
                # Take a screenshot
                screenshot_name = f"screenshot_{screenshot_counter}.png"
                cv2.imwrite(screenshot_name, image)
                screenshot_counter += 1
            print(counter, feedback)

        # Draw pose landmarks on the image
        mp.solutions.drawing_utils.draw_landmarks(
            image,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp.solutions.drawing_styles.get_default_pose_landmarks_style(),
        )

        # Display feedback on the frame
        if stage == "down":
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

    # Display push-up count and correctness count
    cv2.putText(
        image,
        "Push-up Count: {}".format(counter),
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        image,
        "Correct: {}".format(correct_pushups),
        (10, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        image,
        "Incorrect: {}".format(incorrect_pushups),
        (10, 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 255),
        2,
        cv2.LINE_AA,
    )

    # Prepare JSON output
    output = {
        "pushup_count": counter,
        "correct_pushups": correct_pushups,
        "incorrect_pushups": incorrect_pushups,
        "feedback": feedback,
        "screenshot_counter": screenshot_counter,
    }

    return image, output
