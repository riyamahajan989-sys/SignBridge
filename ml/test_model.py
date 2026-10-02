import pandas as pd
import cv2
import mediapipe as mp
import pickle
import os


# -----------------------------------
# File paths
# -----------------------------------
BASE_DIR = os.path.dirname(__file__)

model_path = os.path.join(
    BASE_DIR,
    "model",
    "sign_model.pkl"
)

landmarker_path = os.path.join(
    BASE_DIR,
    "hand_landmarker.task"
)


# -----------------------------------
# Load trained KNN model
# -----------------------------------
with open(model_path, "rb") as file:
    model = pickle.load(file)

print("KNN model loaded successfully!")


# -----------------------------------
# MediaPipe setup
# -----------------------------------
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=landmarker_path
    ),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5
)


# -----------------------------------
# Start camera
# -----------------------------------
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

print("Camera started.")
print("Press Q to quit.")


frame_count = 0

with HandLandmarker.create_from_options(options) as landmarker:

    while True:

        ret, frame = cap.read()

        if not ret:
            print("Could not read camera frame.")
            break

        # Convert BGR → RGB
        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        # Create MediaPipe image
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        # Increasing timestamp for VIDEO mode
        frame_count += 1
        timestamp_ms = frame_count * 33

        # Detect hand
        result = landmarker.detect_for_video(
            mp_image,
            timestamp_ms
        )

        prediction = "No hand detected"

        # -----------------------------------
        # If hand is detected
        # -----------------------------------
        if result.hand_landmarks:

            landmarks = result.hand_landmarks[0]

            features = []

            for landmark in landmarks:
                features.extend([
                    landmark.x,
                    landmark.y,
                    landmark.z
                ])

            # Make prediction
            columns = []

            for i in range(21):
                columns.extend([
                    f"x{i}",
                    f"y{i}",
                    f"z{i}"
                ])

            input_data = pd.DataFrame(
                [features],
                columns=columns
            )

            prediction = model.predict(input_data)[0]

            # Draw hand landmarks
            h, w, _ = frame.shape

            for landmark in landmarks:

                x = int(landmark.x * w)
                y = int(landmark.y * h)

                cv2.circle(
                    frame,
                    (x, y),
                    5,
                    (0, 255, 0),
                    -1
                )

        # -----------------------------------
        # Display prediction
        # -----------------------------------

        cv2.putText(
            frame,
            f"Prediction: {prediction}",
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            "Press Q to quit",
            (20, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        cv2.imshow(
            "SignBridge - Sign Recognition",
            frame
        )

        # Quit
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break


# -----------------------------------
# Close everything
# -----------------------------------
cap.release()
cv2.destroyAllWindows()

print("Test completed.")