from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from gesture_utils import HandCropper


PROJECT_DIR = Path(__file__).resolve().parent
DATASET_DIR = PROJECT_DIR / "dataset"
MODEL_PATH = PROJECT_DIR / "models" / "gesture_landmark_model.keras"
CLASS_NAMES = ["A", "B", "C", "No_Gesture"]
CLASS_SOURCES = {
    "A": ["A"],
    "B": ["B"],
    "C": ["C"],
    "No_Gesture": ["No_Gesture", "nothing"],
}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
FEATURE_COUNT = 21 * 3
VALIDATION_RATIO = 0.20
SEED = 42


def load_class_features(class_name: str, cropper: HandCropper) -> list[np.ndarray]:
    features: list[np.ndarray] = []
    paths = sorted(
        path
        for folder_name in CLASS_SOURCES[class_name]
        for path in (DATASET_DIR / folder_name).iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )

    for index, path in enumerate(paths, start=1):
        frame = cv2.imread(str(path))
        if frame is None:
            continue

        detection = cropper.detect(frame)
        if detection.vector is None:
            if class_name == "No_Gesture":
                features.append(np.zeros(FEATURE_COUNT, dtype=np.float32))
            continue

        features.append(detection.vector)
        if index % 250 == 0:
            print(f"{class_name}: processed {index}/{len(paths)}")

    print(f"{class_name}: using {len(features)} landmark samples")
    return features


def build_model() -> keras.Model:
    inputs = keras.Input(shape=(FEATURE_COUNT,))
    x = layers.Dense(128, activation="relu")(inputs)
    x = layers.Dropout(0.30)(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.20)(x)
    outputs = layers.Dense(len(CLASS_NAMES), activation="softmax")(x)
    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main() -> None:
    np.random.seed(SEED)
    tf.random.set_seed(SEED)
    cropper = HandCropper(static_image_mode=True)
    try:
        class_features = [load_class_features(name, cropper) for name in CLASS_NAMES]
    finally:
        cropper.close()

    if any(not values for values in class_features):
        raise RuntimeError("Every class needs at least one landmark sample.")

    train_x: list[np.ndarray] = []
    validation_x: list[np.ndarray] = []
    train_y: list[int] = []
    validation_y: list[int] = []

    for label, values in enumerate(class_features):
        shuffled = np.asarray(values, dtype=np.float32)
        np.random.shuffle(shuffled)
        split = max(1, int(len(shuffled) * (1 - VALIDATION_RATIO)))
        train_x.extend(shuffled[:split])
        validation_x.extend(shuffled[split:])
        train_y.extend([label] * split)
        validation_y.extend([label] * (len(shuffled) - split))

    train_x = np.asarray(train_x, dtype=np.float32)
    validation_x = np.asarray(validation_x, dtype=np.float32)
    train_y = np.asarray(train_y, dtype=np.int32)
    validation_y = np.asarray(validation_y, dtype=np.int32)
    order = np.random.permutation(len(train_x))
    train_x, train_y = train_x[order], train_y[order]

    counts = np.bincount(train_y, minlength=len(CLASS_NAMES))
    total = len(train_y)
    class_weights = {
        label: total / (len(CLASS_NAMES) * count)
        for label, count in enumerate(counts)
        if count
    }

    model = build_model()
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=10, restore_best_weights=True
        ),
        keras.callbacks.ModelCheckpoint(
            MODEL_PATH, monitor="val_loss", save_best_only=True
        ),
    ]
    model.fit(
        train_x,
        train_y,
        validation_data=(validation_x, validation_y),
        epochs=60,
        batch_size=32,
        class_weight=class_weights,
        callbacks=callbacks,
    )
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    model.save(MODEL_PATH)
    print(f"Saved landmark model to {MODEL_PATH}")


if __name__ == "__main__":
    main()
