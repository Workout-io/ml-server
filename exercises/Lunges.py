import cv2
import mediapipe as mp
import numpy as np
from utils.utils import calculate_angle

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Initialize lunge counter and correctness tracker
correct_lunges = 0
incorrect_lunges = 0
stage = None
screenshot_counter = 0


def process_frame_lunge(frame):
    global correct_lunges, incorrect_lunges, stage, screenshot_counter

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

        # Get relevant landmarks for lunge detection
        right_hip = [
            landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
            landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y,
        ]
        right_knee = [
            landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
            landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y,
        ]
        right_ankle = [
            landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
            landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y,
        ]

        left_hip = [
            landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
            landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y,
        ]
        left_knee = [
            landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
            landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y,
        ]
        left_ankle = [
            landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
            landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y,
        ]

        # Calculate angles
        right_knee_angle = calculate_angle(right_hip, right_knee, right_ankle)
        left_knee_angle = calculate_angle(left_hip, left_knee, left_ankle)

        # Visualize angles
        cv2.putText(
            image,
            f"Right Knee: {int(right_knee_angle)}",
            tuple(np.multiply(right_knee, [640, 480]).astype(int)),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            image,
            f"Left Knee: {int(left_knee_angle)}",
            tuple(np.multiply(left_knee, [640, 480]).astype(int)),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        # Draw lines for visualization
        cv2.line(
            image,
            tuple(np.multiply(right_hip, [640, 480]).astype(int)),
            tuple(np.multiply(right_knee, [640, 480]).astype(int)),
            (0, 255, 0),
            2,
        )
        cv2.line(
            image,
            tuple(np.multiply(right_knee, [640, 480]).astype(int)),
            tuple(np.multiply(right_ankle, [640, 480]).astype(int)),
            (0, 255, 0),
            2,
        )
        cv2.line(
            image,
            tuple(np.multiply(left_hip, [640, 480]).astype(int)),
            tuple(np.multiply(left_knee, [640, 480]).astype(int)),
            (0, 255, 0),
            2,
        )
        cv2.line(
            image,
            tuple(np.multiply(left_knee, [640, 480]).astype(int)),
            tuple(np.multiply(left_ankle, [640, 480]).astype(int)),
            (0, 255, 0),
            2,
        )

        # Correct lines for feedback
        correct_right_knee = [
            right_knee[0],
            right_knee[1] - 0.1,
        ]  # Adjust the y value as needed
        correct_left_knee = [
            left_knee[0],
            left_knee[1] - 0.1,
        ]  # Adjust the y value as needed

        # Draw correct position lines
        cv2.line(
            image,
            tuple(np.multiply(right_knee, [640, 480]).astype(int)),
            tuple(np.multiply(correct_right_knee, [640, 480]).astype(int)),
            (255, 255, 0),  # Cyan for correct line
            2,
        )
        cv2.line(
            image,
            tuple(np.multiply(left_knee, [640, 480]).astype(int)),
            tuple(np.multiply(correct_left_knee, [640, 480]).astype(int)),
            (255, 255, 0),  # Cyan for correct line
            2,
        )

        # Lunge counter logic
        if right_knee_angle > 160 and left_knee_angle > 160:
            stage = "up"
        if right_knee_angle < 90 and left_knee_angle > 90 and stage == "up":
            stage = "down"
            # Determine if lunge is correct or incorrect
            if 80 <= right_knee_angle <= 100 and 160 <= left_knee_angle <= 180:
                correct_lunges += 1
                feedback = "Correct Lunge"
                color = (0, 255, 0)  # Green for correct
            else:
                incorrect_lunges += 1
                feedback = "Incorrect Lunge"
                color = (0, 0, 255)  # Red for incorrect
                # Draw lines indicating error
                cv2.line(
                    image,
                    tuple(np.multiply(right_hip, [640, 480]).astype(int)),
                    tuple(np.multiply(right_knee, [640, 480]).astype(int)),
                    (0, 0, 255),
                    2,
                )
                cv2.line(
                    image,
                    tuple(np.multiply(right_knee, [640, 480]).astype(int)),
                    tuple(np.multiply(right_ankle, [640, 480]).astype(int)),
                    (0, 0, 255),
                    2,
                )
                cv2.line(
                    image,
                    tuple(np.multiply(left_hip, [640, 480]).astype(int)),
                    tuple(np.multiply(left_knee, [640, 480]).astype(int)),
                    (0, 0, 255),
                    2,
                )
                cv2.line(
                    image,
                    tuple(np.multiply(left_knee, [640, 480]).astype(int)),
                    tuple(np.multiply(left_ankle, [640, 480]).astype(int)),
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

    # Display lunge count and correctness count
    total_lunges = correct_lunges + incorrect_lunges
    cv2.putText(
        image,
        f"Total Lunges: {total_lunges}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        image,
        f"Correct: {correct_lunges}",
        (10, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        image,
        f"Incorrect: {incorrect_lunges}",
        (10, 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 255),
        2,
        cv2.LINE_AA,
    )

    return image
