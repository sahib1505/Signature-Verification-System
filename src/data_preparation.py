import os
from pathlib import Path

import cv2
import numpy as np

IMG_SIZE = (128, 128)


def load_images(folder: str | Path, label: int):
    """
    Load all images from a folder, convert to grayscale, resize, and return
    arrays of images and labels.
    """
    folder = Path(folder)
    images = []
    labels = []

    for filename in folder.iterdir():
        if not filename.is_file():
            continue

        img = cv2.imread(str(filename), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        img = cv2.resize(img, IMG_SIZE)
        images.append(img)
        labels.append(label)

    return np.array(images), np.array(labels)


def build_dataset(
    genuine_dir: str | Path = "data/genuine",
    forged_dir: str | Path = "data/forged",
):
    """
    Build full dataset: X (images) and y (labels).
    label 1 = genuine, 0 = forged
    """
    genuine_images, genuine_labels = load_images(genuine_dir, 1)
    forged_images, forged_labels = load_images(forged_dir, 0)

    X = np.concatenate([genuine_images, forged_images], axis=0)
    y = np.concatenate([genuine_labels, forged_labels], axis=0)

    return X, y


if __name__ == "__main__":
    X, y = build_dataset()
    print("Images shape:", X.shape)
    print("Labels shape:", y.shape)
