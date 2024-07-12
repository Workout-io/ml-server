import cv2
import mediapipe as mp
import numpy as np
from utils.utils import calculate_angle

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Initialize plank tracking variables
plank_start_time = None
plank_duration = 0
correct_plank_duration = 0
incorrect_plank_duration = 0
feedback = ""
color = (0, 0, 0)


def process_frame_plank(frame):
    global plank_start_time, plank_duration, correct_plank_duration, incorrect_plank_duration, feedback, color

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

        # Get relevant landmarks for plank detection
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
        shoulder_hip_knee_angle = calculate_angle(shoulder, hip, knee)
        hip_knee_ankle_angle = calculate_angle(hip, knee, ankle)

        # Visualize angles
        cv2.putText(
            image,
            f"Hip Angle: {int(shoulder_hip_knee_angle)}",
            tuple(np.multiply(hip, [640, 480]).astype(int)),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            image,
            f"Knee Angle: {int(hip_knee_ankle_angle)}",
            tuple(np.multiply(knee, [640, 480]).astype(int)),
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

        # Plank detection logic
        if 160 <= shoulder_hip_knee_angle <= 180 and 160 <= hip_knee_ankle_angle <= 180:
            feedback = "Correct Plank"
            color = (0, 255, 0)  # Green for correct
            if plank_start_time is None:
                plank_start_time = cv2.getTickCount()
            else:
                plank_duration = (
                    cv2.getTickCount() - plank_start_time
                ) / cv2.getTickFrequency()
                correct_plank_duration = plank_duration
        else:
            feedback = "Incorrect Plank"
            color = (0, 0, 255)  # Red for incorrect
            if plank_start_time is not None:
                plank_duration = (
                    cv2.getTickCount() - plank_start_time
                ) / cv2.getTickFrequency()
                incorrect_plank_duration = plank_duration
                plank_start_time = None

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

    # Display plank duration
    cv2.putText(
        image,
        f"Correct Duration: {correct_plank_duration:.2f}s",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        image,
        f"Incorrect Duration: {incorrect_plank_duration:.2f}s",
        (10, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 255),
        2,
        cv2.LINE_AA,
    )

    return image
