# src/cnn_model.py
from pathlib import Path

import numpy as np
from tensorflow.keras import layers, models
from tensorflow.keras.models import load_model as keras_load_model

from .data_preparation import IMG_SIZE, build_dataset

CNN_MODEL_PATH = Path("models/cnn_signature.h5")


def build_cnn(input_shape=(128, 128, 1)):
    model = models.Sequential(
        [
            layers.Input(shape=input_shape),
            layers.Conv2D(32, (3, 3), activation="relu"),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation="relu"),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(128, (3, 3), activation="relu"),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dense(128, activation="relu"),
            layers.Dropout(0.5),
            layers.Dense(1, activation="sigmoid"),  # binary: genuine / forged
        ]
    )
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def prepare_cnn_data():
    X, y = build_dataset()
    # X shape: (N, 128, 128) -> (N, 128, 128, 1), normalize
    X = X.astype("float32") / 255.0
    X = np.expand_dims(X, axis=-1)
    y = y.astype("float32")
    return X, y


def load_cnn_model():
    if not CNN_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"CNN model not found at {CNN_MODEL_PATH}. Train it first using train_cnn.py"
        )
    return keras_load_model(CNN_MODEL_PATH)
