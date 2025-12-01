import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

from .data_preparation import build_dataset
from .feature_extraction import extract_hog_features

MODEL_PATH = "models/svm_signature.pkl"


def train_svm_model():
    # 1. Load dataset
    X, y = build_dataset()

    # 2. Extract features
    X_features = extract_hog_features(X)

    # 3. Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_features, y, test_size=0.2, random_state=42, stratify=y
    )

    # 4. Train model
    svm_model = SVC(kernel="rbf", C=1.0, probability=True)
    svm_model.fit(X_train, y_train)

    # 5. Evaluate
    y_pred = svm_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc:.3f}")

    # 6. Save model
    joblib.dump(svm_model, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

    return svm_model


if __name__ == "__main__":
    train_svm_model()

