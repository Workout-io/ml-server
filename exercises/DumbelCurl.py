import cv2
import mediapipe as mp
import numpy as np
from utils.utils import calculate_angle

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Initialize dumbbell curl counter and correctness tracker
correct_curls = 0
incorrect_curls = 0
stage = None
screenshot_counter = 0


def process_frame_curl(frame):
    global correct_curls, incorrect_curls, stage, screenshot_counter

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

        # Get relevant landmarks for dumbbell curl detection
        right_shoulder = [
            landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
            landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y,
        ]
        right_elbow = [
            landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
            landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y,
        ]
        right_wrist = [
            landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
            landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y,
        ]

        left_shoulder = [
            landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
            landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y,
        ]
        left_elbow = [
            landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
            landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y,
        ]
        left_wrist = [
            landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
            landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y,
        ]

        # Calculate angles
        right_elbow_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)
        left_elbow_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)

        # Visualize angles
        cv2.putText(
            image,
            f"Right Elbow: {int(right_elbow_angle)}",
            tuple(np.multiply(right_elbow, [640, 480]).astype(int)),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            image,
            f"Left Elbow: {int(left_elbow_angle)}",
            tuple(np.multiply(left_elbow, [640, 480]).astype(int)),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        # Draw lines for visualization
        cv2.line(
            image,
            tuple(np.multiply(right_shoulder, [640, 480]).astype(int)),
            tuple(np.multiply(right_elbow, [640, 480]).astype(int)),
            (0, 255, 0),
            2,
        )
        cv2.line(
            image,
            tuple(np.multiply(right_elbow, [640, 480]).astype(int)),
            tuple(np.multiply(right_wrist, [640, 480]).astype(int)),
            (0, 255, 0),
            2,
        )
        cv2.line(
            image,
            tuple(np.multiply(left_shoulder, [640, 480]).astype(int)),
            tuple(np.multiply(left_elbow, [640, 480]).astype(int)),
            (0, 255, 0),
            2,
        )
        cv2.line(
            image,
            tuple(np.multiply(left_elbow, [640, 480]).astype(int)),
            tuple(np.multiply(left_wrist, [640, 480]).astype(int)),
            (0, 255, 0),
            2,
        )

        # Curl counter logic
        if 120 <= right_elbow_angle <= 160 and 120 <= left_elbow_angle <= 160:
            stage = "up"
        if right_elbow_angle < 90 and left_elbow_angle < 90 and stage == "up":
            stage = "down"
            # Determine if curl is correct or incorrect
            if 160 <= right_elbow_angle <= 180 and 160 <= left_elbow_angle <= 180:
                correct_curls += 1
                feedback = "Correct Curl"
                color = (0, 255, 0)  # Green for correct
            else:
                incorrect_curls += 1
                feedback = "Incorrect Curl"
                color = (0, 0, 255)  # Red for incorrect
                # Draw lines indicating error
                cv2.line(
                    image,
                    tuple(np.multiply(right_shoulder, [640, 480]).astype(int)),
                    tuple(np.multiply(right_elbow, [640, 480]).astype(int)),
                    (0, 0, 255),
                    2,
                )
                cv2.line(
                    image,
                    tuple(np.multiply(right_elbow, [640, 480]).astype(int)),
                    tuple(np.multiply(right_wrist, [640, 480]).astype(int)),
                    (0, 0, 255),
                    2,
                )
                cv2.line(
                    image,
                    tuple(np.multiply(left_shoulder, [640, 480]).astype(int)),
                    tuple(np.multiply(left_elbow, [640, 480]).astype(int)),
                    (0, 0, 255),
                    2,
                )
                cv2.line(
                    image,
                    tuple(np.multiply(left_elbow, [640, 480]).astype(int)),
                    tuple(np.multiply(left_wrist, [640, 480]).astype(int)),
                    (0, 0, 255),
                    2,
                )
                # Take a screenshot
                screenshot_name = f"screenshot_{screenshot_counter}.png"
                cv2.imwrite(screenshot_name, image)
                screenshot_counter += 1
            print(feedback)

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

    # Display curl count and correctness count
    total_curls = correct_curls + incorrect_curls
    cv2.putText(
        image,
        f"Total Curls: {total_curls}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        image,
        f"Correct: {correct_curls}",
        (10, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        image,
        f"Incorrect: {incorrect_curls}",
        (10, 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 255),
        2,
        cv2.LINE_AA,
    )

    return image
