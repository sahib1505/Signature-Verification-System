# src/train_cnn.py
from pathlib import Path

from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint

from .cnn_model import build_cnn, prepare_cnn_data, CNN_MODEL_PATH


def train_cnn_model():
    X, y = prepare_cnn_data()

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = build_cnn(input_shape=X_train.shape[1:])

    CNN_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = ModelCheckpoint(
        filepath=str(CNN_MODEL_PATH),
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1,
    )

    history = model.fit(
        X_train,
        y_train,
        epochs=10,          # increase if you have time
        batch_size=32,
        validation_data=(X_val, y_val),
        callbacks=[checkpoint],
    )

    print("Training finished. Best model saved to", CNN_MODEL_PATH)
    return model, history


if __name__ == "__main__":
    train_cnn_model()
