from collections import deque
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import pyttsx3
import tensorflow as tf

from gesture_utils import HandCropper


PROJECT_DIR = Path(__file__).resolve().parent
LANDMARK_MODEL_PATH = PROJECT_DIR / "models" / "gesture_landmark_model.keras"
CLASS_NAMES_PATH = PROJECT_DIR / "class_names.txt"
CONFIDENCE_THRESHOLD = 0.80
SMOOTHING_WINDOW = 5
STABLE_FRAMES = 4
HAND_CONNECTIONS = [
    (connection.start, connection.end)
    for connection in mp.tasks.vision.HandLandmarksConnections.HAND_CONNECTIONS
]

# Load trained model
model = tf.keras.models.load_model(LANDMARK_MODEL_PATH)

# Load class names
with CLASS_NAMES_PATH.open("r") as f:
    class_names = [line.strip() for line in f.readlines()]

def speak_prediction(prediction: str) -> None:
    engine = pyttsx3.init()
    engine.say(prediction)
    engine.runAndWait()
    engine.stop()

cap = cv2.VideoCapture(0)
hand_cropper = HandCropper()

prediction_history = deque(maxlen=SMOOTHING_WINDOW)
last_spoken_prediction = ""

print("Starting real-time gesture recognition...")
print("Press Q to quit.")

while True:
    ret, frame = cap.read()

    if not ret:
        print("Could not access camera.")
        break

    detection = hand_cropper.detect(frame)
    if detection.vector is None:
        predicted_class = "No_Gesture"
        confidence = 1.0
    else:
        model_input = np.expand_dims(detection.vector, axis=0)
        predictions = model.predict(model_input, verbose=0)
        predicted_index = int(np.argmax(predictions[0]))
        confidence = float(predictions[0][predicted_index])
        predicted_class = class_names[predicted_index]
        if confidence < CONFIDENCE_THRESHOLD:
            predicted_class = "No_Gesture"

    prediction_history.append(predicted_class)
    stable_class = predicted_class
    if len(prediction_history) == SMOOTHING_WINDOW:
        candidate = prediction_history[-1]
        if prediction_history.count(candidate) >= STABLE_FRAMES:
            stable_class = candidate

    if detection.box is not None:
        left, top, right, bottom = detection.box
        cv2.rectangle(frame, (left, top), (right, bottom), (255, 180, 0), 2)

    if detection.landmarks is not None:
        for start, end in HAND_CONNECTIONS:
            cv2.line(
                frame,
                detection.landmarks[start],
                detection.landmarks[end],
                (0, 255, 0),
                2,
            )
        for point in detection.landmarks:
            cv2.circle(frame, point, 5, (0, 0, 255), -1)

    # Display prediction
    text = f"Prediction: {stable_class}"
    confidence_text = f"Confidence: {confidence * 100:.2f}%"

    cv2.putText(
        frame,
        text,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        confidence_text,
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2
    )

    cv2.imshow(
        "Gesture to Speech",
        frame
    )

    if stable_class == "No_Gesture":
        last_spoken_prediction = ""
    elif confidence >= CONFIDENCE_THRESHOLD and stable_class != last_spoken_prediction:
        print(f"Recognized: {stable_class} ({confidence * 100:.2f}%)")
        speak_prediction(stable_class)
        last_spoken_prediction = stable_class

    # Quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
hand_cropper.close()
cv2.destroyAllWindows()
