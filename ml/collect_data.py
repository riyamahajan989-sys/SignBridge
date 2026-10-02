import cv2
import csv
import os
import time
import mediapipe as mp

# -----------------------------
# Settings
# -----------------------------
SIGNS = [
    # "Hello",
    # "Thank_You",
    # "Yes",
    # "No",
    # "Help",
    # "Stop",
    # "Water",
    # "Food",
    # "Goodbye",
    # "Emergency"
    "Sorry",
    "Please",
    "School",
    "Home",
    "Doctor"
]

SAMPLES_PER_SIGN = 100

# -----------------------------
# MediaPipe setup
# -----------------------------
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

model_path = os.path.join(
    os.path.dirname(__file__),
    "hand_landmarker.task"
)

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5
)

# -----------------------------
# Dataset file
# -----------------------------
dataset_folder = os.path.join(
    os.path.dirname(__file__),
    "dataset"
)

os.makedirs(dataset_folder, exist_ok=True)

csv_file = os.path.join(dataset_folder, "sign_data.csv")

# Create CSV if it doesn't exist
if not os.path.exists(csv_file):
    with open(csv_file, "w", newline="") as file:
        writer = csv.writer(file)

        header = ["label"]

        for i in range(21):
            header += [f"x{i}", f"y{i}", f"z{i}"]

        writer.writerow(header)

# -----------------------------
# Camera
# -----------------------------
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Camera could not be opened.")
    exit()

print("\nSignBridge Data Collection")
print("-------------------------")

with HandLandmarker.create_from_options(options) as landmarker:

    for sign in SIGNS:

        print(f"\nGet ready for: {sign}")
        print("Press SPACE to start collecting.")
        print("Press Q to quit.")

        while True:
            ret, frame = cap.read()

            if not ret:
                break

            cv2.putText(
                frame,
                f"Sign: {sign}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                "Press SPACE to start",
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

            cv2.imshow("SignBridge Data Collection", frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord(" "):
                break

            if key == ord("q"):
                cap.release()
                cv2.destroyAllWindows()
                exit()

        print(f"Collecting {SAMPLES_PER_SIGN} samples for {sign}...")

        count = 0

        while count < SAMPLES_PER_SIGN:

            ret, frame = cap.read()

            if not ret:
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb_frame
            )

            timestamp_ms = int(time.time() * 1000)

            result = landmarker.detect_for_video(
                mp_image,
                timestamp_ms
            )

            if result.hand_landmarks:

                landmarks = result.hand_landmarks[0]

                row = [sign]

                for landmark in landmarks:
                    row.extend([
                        landmark.x,
                        landmark.y,
                        landmark.z
                    ])

                with open(csv_file, "a", newline="") as file:
                    writer = csv.writer(file)
                    writer.writerow(row)

                count += 1

            cv2.putText(
                frame,
                f"{sign}: {count}/{SAMPLES_PER_SIGN}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

            cv2.imshow("SignBridge Data Collection", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                cap.release()
                cv2.destroyAllWindows()
                exit()

        print(f"Completed: {sign}")

cap.release()
cv2.destroyAllWindows()

print("\nAll sign data collected successfully!")
print(f"Dataset saved at: {csv_file}")