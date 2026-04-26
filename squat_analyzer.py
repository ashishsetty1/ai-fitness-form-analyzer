import cv2
import mediapipe as mp
import numpy as np
import sys
import csv
from datetime import datetime

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose


def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(
        a[1] - b[1], a[0] - b[0]
    )

    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180:
        angle = 360 - angle

    return angle


def get_point(landmarks, landmark):
    return [
        landmarks[landmark.value].x,
        landmarks[landmark.value].y,
    ]


def normalized_to_pixel(point, width, height):
    return int(point[0] * width), int(point[1] * height)


def detect_weight_near_wrist(frame, wrist_point, label):
    height, width, _ = frame.shape
    wrist_x, wrist_y = normalized_to_pixel(wrist_point, width, height)

    box_size = 90

    x1 = max(0, wrist_x - box_size)
    y1 = max(0, wrist_y - box_size)
    x2 = min(width, wrist_x + box_size)
    y2 = min(height, wrist_y + box_size)

    roi = frame[y1:y2, x1:x2]

    if roi.size == 0:
        return False

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, threshold = cv2.threshold(gray, 70, 255, cv2.THRESH_BINARY_INV)

    contours, _ = cv2.findContours(
        threshold,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    for contour in contours:
        area = cv2.contourArea(contour)
        if 250 < area < 6000:
            x, y, w, h = cv2.boundingRect(contour)

            if w > 15 and h > 15:
                cv2.rectangle(
                    frame,
                    (x1 + x, y1 + y),
                    (x1 + x + w, y1 + y + h),
                    (255, 255, 255),
                    2,
                )

                cv2.putText(
                    frame,
                    label,
                    (x1 + x, max(20, y1 + y - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 255),
                    2,
                )
                return True

    return False


def save_workout(squat_reps, pushup_reps, left_curls, right_curls):
    filename = "workout_log.csv"
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    try:
        with open(filename, "r"):
            file_exists = True
    except:
        file_exists = False

    with open(filename, "a", newline="") as f:
        writer = csv.writer(f)

        if not file_exists:
            writer.writerow(["timestamp", "squats", "pushups", "left_curls", "right_curls"])

        writer.writerow([timestamp, squat_reps, pushup_reps, left_curls, right_curls])


cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)

if not cap.isOpened():
    print("Camera error")
    sys.exit(1)

# 🎥 VIDEO RECORDING SETUP
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter("demo.mp4", fourcc, 20.0, (frame_width, frame_height))


mode = "squat"

squat_stage = None
pushup_stage = None
left_curl_stage = None
right_curl_stage = None

squat_reps = 0
pushup_reps = 0
left_curls = 0
right_curls = 0

left_weight_detected = False
right_weight_detected = False


with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False
        results = pose.process(image_rgb)

        image_rgb.flags.writeable = True
        image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

        feedback = "No person detected"
        angle_text = ""
        weight_text = ""

        try:
            landmarks = results.pose_landmarks.landmark

            left_shoulder = get_point(landmarks, mp_pose.PoseLandmark.LEFT_SHOULDER)
            left_elbow = get_point(landmarks, mp_pose.PoseLandmark.LEFT_ELBOW)
            left_wrist = get_point(landmarks, mp_pose.PoseLandmark.LEFT_WRIST)

            right_shoulder = get_point(landmarks, mp_pose.PoseLandmark.RIGHT_SHOULDER)
            right_elbow = get_point(landmarks, mp_pose.PoseLandmark.RIGHT_ELBOW)
            right_wrist = get_point(landmarks, mp_pose.PoseLandmark.RIGHT_WRIST)

            left_hip = get_point(landmarks, mp_pose.PoseLandmark.LEFT_HIP)
            left_knee = get_point(landmarks, mp_pose.PoseLandmark.LEFT_KNEE)
            left_ankle = get_point(landmarks, mp_pose.PoseLandmark.LEFT_ANKLE)

            if mode == "curl":
                left_weight_detected = detect_weight_near_wrist(image, left_wrist, "Left Weight")
                right_weight_detected = detect_weight_near_wrist(image, right_wrist, "Right Weight")

                weight_text = f"L: {'YES' if left_weight_detected else 'NO'} | R: {'YES' if right_weight_detected else 'NO'}"

            if mode == "squat":
                knee_angle = calculate_angle(left_hip, left_knee, left_ankle)

                if knee_angle > 160:
                    if squat_stage == "down":
                        squat_reps += 1
                    squat_stage = "up"

                elif knee_angle < 100:
                    squat_stage = "down"

            elif mode == "pushup":
                angle = calculate_angle(left_shoulder, left_elbow, left_wrist)

                if angle > 160:
                    if pushup_stage == "down":
                        pushup_reps += 1
                    pushup_stage = "up"

                elif angle < 90:
                    pushup_stage = "down"

            elif mode == "curl":
                left_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
                right_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)

                if left_angle > 150:
                    left_curl_stage = "down"
                if left_angle < 55 and left_curl_stage == "down":
                    left_curls += 1
                    left_curl_stage = "up"

                if right_angle > 150:
                    right_curl_stage = "down"
                if right_angle < 55 and right_curl_stage == "down":
                    right_curls += 1
                    right_curl_stage = "up"

        except:
            pass

        cv2.putText(image, f"S:{squat_reps} P:{pushup_reps} R:{left_curls} L:{right_curls}",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

        cv2.imshow("AI Fitness Analyzer", image)

        # 🎥 SAVE FRAME TO VIDEO
        out.write(image)

        key = cv2.waitKey(10) & 0xFF

        if key == ord("q"):
            save_workout(squat_reps, pushup_reps, left_curls, right_curls)
            break
        elif key == ord("s"):
            mode = "squat"
        elif key == ord("p"):
            mode = "pushup"
        elif key == ord("c"):
            mode = "curl"

cap.release()
out.release()
cv2.destroyAllWindows()