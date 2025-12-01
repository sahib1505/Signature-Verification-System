from pathlib import Path

import cv2
import joblib
import numpy as np
from skimage.feature import hog

from .data_preparation import IMG_SIZE
from .cnn_model import load_cnn_model

SVM_MODEL_PATH = Path("models/svm_signature.pkl")


# ---------- Common helpers ----------

def preprocess_image(signature_path: str | Path):
    img = cv2.imread(str(signature_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not read image at {signature_path}")
    img = cv2.resize(img, IMG_SIZE)
    return img


def extract_single_hog(img) -> np.ndarray:
    """HOG features for a single image."""
    h = hog(
        img,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
    )
    return h.reshape(1, -1)


# ---------- SVM (HOG) ----------

def load_svm_model():
    if not SVM_MODEL_PATH.exists():
        raise FileNotFoundError(f"SVM model not found at {SVM_MODEL_PATH}")
    return joblib.load(SVM_MODEL_PATH)


def verify_with_svm(signature_path: str | Path, svm_model=None):
    if svm_model is None:
        svm_model = load_svm_model()

    img = preprocess_image(signature_path)
    features = extract_single_hog(img)

    proba = svm_model.predict_proba(features)[0]  # [p_forged, p_genuine]
    pred = svm_model.predict(features)[0]

    label = "Genuine" if pred == 1 else "Forged"
    confidence = float(max(proba))
    return label, confidence


# ---------- CNN ----------

def verify_with_cnn(signature_path: str | Path, cnn_model=None):
    if cnn_model is None:
        cnn_model = load_cnn_model()

    img = preprocess_image(signature_path)  # (128,128)
    img = img.astype("float32") / 255.0
    img = img.reshape(1, IMG_SIZE[0], IMG_SIZE[1], 1)

    proba = float(cnn_model.predict(img, verbose=0)[0][0])  # prob(genuine)
    label = "Genuine" if proba >= 0.5 else "Forged"
    confidence = proba if label == "Genuine" else 1.0 - proba
    return label, confidence


# ---------- Unified interface ----------

def verify_signature(
    signature_path: str | Path,
    model_type: str = "svm",
    svm_model=None,
    cnn_model=None,
):
    """
    model_type: 'svm' or 'cnn'
    returns: (label, confidence)
    """
    if model_type == "cnn":
        return verify_with_cnn(signature_path, cnn_model)
    else:
        return verify_with_svm(signature_path, svm_model)
