import numpy as np
from skimage.feature import hog

from .data_preparation import IMG_SIZE


def extract_hog_features(images: np.ndarray) -> np.ndarray:
    """
    Compute HOG features for a batch of images.
    """
    features = []

    for img in images:
        img_2d = img.reshape(IMG_SIZE)
        hog_features = hog(
            img_2d,
            orientations=9,
            pixels_per_cell=(8, 8),
            cells_per_block=(2, 2),
        )
        features.append(hog_features)

    return np.array(features)


if __name__ == "__main__":
    # quick test
    from .data_preparation import build_dataset

    X, y = build_dataset()
    X_features = extract_hog_features(X)
    print("Feature shape:", X_features.shape)
