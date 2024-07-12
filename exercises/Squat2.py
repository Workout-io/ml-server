import cv2
import mediapipe as mp
import numpy as np
from utils.utils import calculate_angle

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Initialize squat counter and correctness tracker
counter = 0
stage = None
correct_squats = 0
incorrect_squats = 0
screenshot_counter = 0


def process_frame_squatjump(frame):
    global counter, stage, correct_squats, incorrect_squats, screenshot_counter

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

        # Get relevant landmarks for squat detection
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
        ankle = [
            landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
            landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y,
        ]

        # Calculate angles
        knee_angle = calculate_angle(hip, knee, ankle)
        hip_angle = calculate_angle(shoulder, hip, knee)

        # Visualize angles
        cv2.putText(
            image,
            f"Knee: {int(knee_angle)}",
            tuple(np.multiply(knee, [640, 480]).astype(int)),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            image,
            f"Hip: {int(hip_angle)}",
            tuple(np.multiply(hip, [640, 480]).astype(int)),
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
            tuple(np.multiply(hip, [640, 480]).astype(int)),
            (0, 255, 0),
            2,
        )
        cv2.line(
            image,
            tuple(np.multiply(hip, [640, 480]).astype(int)),
            tuple(np.multiply(knee, [640, 480]).astype(int)),
            (0, 255, 0),
            2,
        )
        cv2.line(
            image,
            tuple(np.multiply(knee, [640, 480]).astype(int)),
            tuple(np.multiply(ankle, [640, 480]).astype(int)),
            (0, 255, 0),
            2,
        )

        # Correct lines for feedback
        correct_hip = [hip[0], hip[1] - 0.1]  # Adjust the y value as needed
        correct_knee = [knee[0], knee[1] + 0.1]  # Adjust the y value as needed

        # Draw correct position lines
        cv2.line(
            image,
            tuple(np.multiply(hip, [640, 480]).astype(int)),
            tuple(np.multiply(correct_hip, [640, 480]).astype(int)),
            (255, 255, 0),
            2,
        )
        cv2.line(
            image,
            tuple(np.multiply(knee, [640, 480]).astype(int)),
            tuple(np.multiply(correct_knee, [640, 480]).astype(int)),
            (255, 255, 0),
            2,
        )

        # Squat counter logic
        if knee_angle > 160 and hip_angle > 150:
            stage = "up"
        if knee_angle < 110 and stage == "up":
            stage = "down"
            counter += 1
            # Determine if squat is correct or incorrect
            if 100 <= knee_angle <= 130 and 70 <= hip_angle <= 110:
                correct_squats += 1
                feedback = "Good Squat"
                color = (0, 255, 0)  # Green for correct
            else:
                incorrect_squats += 1
                feedback = "Try Again"
                color = (0, 255, 255)  # Yellow for less strict feedback
                # Draw lines indicating error
                cv2.line(
                    image,
                    tuple(np.multiply(hip, [640, 480]).astype(int)),
                    tuple(np.multiply(knee, [640, 480]).astype(int)),
                    (0, 0, 255),
                    2,
                )
                cv2.line(
                    image,
                    tuple(np.multiply(knee, [640, 480]).astype(int)),
                    tuple(np.multiply(ankle, [640, 480]).astype(int)),
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

    # Display squat count and correctness count
    cv2.putText(
        image,
        "Squat Count: {}".format(counter),
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        image,
        "Correct: {}".format(correct_squats),
        (10, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        image,
        "Incorrect: {}".format(incorrect_squats),
        (10, 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 255),
        2,
        cv2.LINE_AA,
    )

    return image
