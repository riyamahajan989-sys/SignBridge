from flask import Flask, request, jsonify
from flask_cors import CORS

import cv2
import mediapipe as mp
import pickle
import numpy as np
import pandas as pd
import os
from pathlib import Path
app = Flask(__name__)
CORS(app)

# ---------------------------------------
# Paths
# ---------------------------------------
BASE_DIR = os.path.dirname(__file__)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "model",
    "sign_model.pkl"
)

LANDMARKER_PATH = os.path.join(
    BASE_DIR,
    "model",
    "hand_landmarker.task"
)

# ---------------------------------------
# Load KNN model
# ---------------------------------------
with open(MODEL_PATH, "rb") as file:
    model = pickle.load(file)

print("KNN model loaded successfully!")

# ---------------------------------------
# MediaPipe setup
# ---------------------------------------
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

model_bytes = Path(LANDMARKER_PATH).read_bytes()
options = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=LANDMARKER_PATH
    ),
    running_mode=VisionRunningMode.IMAGE,
    num_hands=1,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5
)

landmarker = HandLandmarker.create_from_options(options)

# ---------------------------------------
# English → Marathi
# ---------------------------------------
MARATHI_LABELS = {
    "Hello": "नमस्कार",
    "Thank_You": "धन्यवाद",
    "Yes": "हो",
    "No": "नाही",
    "Help": "मदत",
    "Stop": "थांबा",
    "Water": "पाणी",
    "Food": "अन्न",
    "Goodbye": "निरोप",
    "Emergency": "आपत्कालीन मदत",
    "Sorry": "माफ करा",
    "Please": "कृपया",
    "School": "शाळा",
    "Home": "घर",
    "Doctor": "डॉक्टर"
}

# ---------------------------------------
# Home
# ---------------------------------------
@app.route("/")
def home():
    return "SignBridge Backend is running!"


# ---------------------------------------
# Prediction
# ---------------------------------------
@app.route("/predict", methods=["POST"])
def predict():

    if "image" not in request.files:
        return jsonify({
            "error": "No image received"
        }), 400

    file = request.files["image"]

    image_bytes = file.read()

    image_array = np.frombuffer(
        image_bytes,
        np.uint8
    )

    frame = cv2.imdecode(
        image_array,
        cv2.IMREAD_COLOR
    )

    if frame is None:
        return jsonify({
            "error": "Invalid image"
        }), 400

    # BGR → RGB
    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )

    # Detect hand
    result = landmarker.detect(mp_image)

    if not result.hand_landmarks:
        return jsonify({
            "prediction": None,
            "meaning": "हात दिसला नाही"
        })

    landmarks = result.hand_landmarks[0]

    # ---------------------------------------
    # Create 63 features
    # ---------------------------------------
    features = []

    for landmark in landmarks:
        features.extend([
            landmark.x,
            landmark.y,
            landmark.z
        ])

    # ---------------------------------------
    # Use same feature names as training
    # ---------------------------------------
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

    # ---------------------------------------
    # KNN prediction
    # ---------------------------------------
    prediction = model.predict(input_data)[0]

    marathi_meaning = MARATHI_LABELS.get(
        prediction,
        prediction
    )

    return jsonify({
        "prediction": prediction,
        "meaning": marathi_meaning
    })


# ---------------------------------------
# Run server
# ---------------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))

    app.run(
        host="0.0.0.0",
        port=port,
        debug=True
    )