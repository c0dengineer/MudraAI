import cv2
import numpy as np
import tensorflow as tf
import pyttsx3

# Load trained model
model = tf.keras.models.load_model(
    "models/gesture_model.keras"
)

# Load class names
with open("class_names.txt", "r") as f:
    class_names = [line.strip() for line in f.readlines()]

# Text-to-speech engine
engine = pyttsx3.init()

cap = cv2.VideoCapture(0)

last_prediction = ""
spoken_prediction = ""

print("Starting real-time gesture recognition...")
print("Press Q to quit.")

while True:
    ret, frame = cap.read()

    if not ret:
        print("Could not access camera.")
        break

    # Resize frame to CNN input size
    image = cv2.resize(frame, (64, 64))

    # Normalize pixel values
    image = image.astype("float32") / 255.0

    # Add batch dimension
    image = np.expand_dims(image, axis=0)

    # Prediction
    predictions = model.predict(image, verbose=0)

    predicted_index = np.argmax(predictions[0])
    confidence = float(predictions[0][predicted_index])

    predicted_class = class_names[predicted_index]

    # Display prediction
    text = f"Prediction: {predicted_class}"
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

    # Do not speak when the model sees an empty or unrelated frame. Reset the
    # last spoken gesture so the same sign can be spoken after no gesture.
    if predicted_class == "No_Gesture":
        spoken_prediction = ""

    # Speak only confident A/B/C predictions when the gesture changes.
    elif confidence > 0.80 and predicted_class != spoken_prediction:

        print(
            f"Recognized: {predicted_class} "
            f"({confidence * 100:.2f}%)"
        )

        engine.say(predicted_class)
        engine.runAndWait()

        spoken_prediction = predicted_class

    # Reset speech when prediction changes
    if predicted_class != last_prediction:
        last_prediction = predicted_class

    # Quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
