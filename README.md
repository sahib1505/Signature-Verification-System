🛡️ SignGuard – AI-Powered Signature Verification System

SignGuard is an intelligent web-based application that verifies whether a handwritten signature is Genuine or Forged using both Machine Learning (HOG + SVM) and Deep Learning (CNN) models.

It provides a clean UI, real-time verification, confidence score, and user signature registration.


📝 Introduction

SignGuard is a hybrid Machine Learning + Deep Learning system designed to automate signature verification.
It helps detect forged signatures and provides accurate results with a confidence percentage.

This system can be used in banking, legal verification, corporate workflows, educational institutions, and forensic applications.

✨ Features

✔ Upload signature image for verification
✔ Choose model: HOG + SVM (fast) or CNN (accurate)
✔ Shows Genuine / Forged with confidence score
✔ Signature preview after upload
✔ User signature registration module
✔ Professional UI with responsive design
✔ Easy to extend with new models and datasets

🏗️ System Architecture
User → Upload Signature
       ↓
Flask Web UI
       ↓
Preprocessing (Grayscale, Resize 128x128)
       ↓
Model Selection (SVM / CNN)
       ↓
SVM → HOG Feature Extraction → Classification
CNN → Image Normalization → Deep Feature Detection
       ↓
Prediction + Confidence Score
       ↓
Results Displayed on UI

💻 Tech Stack
Backend
 Python 3

Flask Web Framework

SVM (Machine Learning)

CNN (Deep Learning – TensorFlow/Keras)

Frontend
  HTML5, CSS3, Bootstrap

Jinja2 Templates

Others
  OpenCV, NumPy, scikit-learn

scikit-image (HOG)

joblib (model saving)

📚 Libraries Used
Library	Purpose
Flask	Web backend, routing, UI rendering
OpenCV	Image loading, resizing, preprocessing
NumPy	Array operations
TensorFlow / Keras	CNN model training
scikit-learn	SVM classifier, train-test split
scikit-image	HOG feature extraction
joblib	Save/load ML models
Werkzeug	Secure file uploads
Pathlib	File path handling
🗂️ Dataset

The dataset consists of two types:

data/
   genuine/        # Real signatures
   forged/         # Fake signatures

Preprocessing Steps

Convert to grayscale

Resize to 128 × 128 px

Normalize (CNN)

Extract HOG features (SVM)

🧠 Model Details
1. HOG + SVM Model

Extracts gradient-based features from signatures

Fast, lightweight, works well with small datasets

Stored as: models/svm_signature.pkl

2. CNN (Convolutional Neural Network)

Learns deep handwriting patterns

Higher accuracy than SVM

Stored as: models/cnn_signature.h5

⚙️ Installation
1. Clone using the web URL
https://github.com/sahib1505/Signature-Verification-System.git
cd to the path of project


3. Create virtual environment
python -m venv venv

4. Activate environment

Windows:

venv\Scripts\activate


Linux/Mac:

source venv/bin/activate

4. Install dependencies
pip install -r requirements.txt

🚀 How to Run
Train SVM Model
python -m src.train_svm

Train CNN Model
python -m src.train_cnn

Start Flask App
python -m ui.app

Open in browser
http://127.0.0.1:5000/

📁 Project Structure
SignGuard/
│── src/
│   ├── data_preparation.py
│   ├── feature_extraction.py
│   ├── train_svm.py
│   ├── train_cnn.py
│   ├── cnn_model.py
│   ├── verify_signature.py
│
│── models/
│── data/
│    ├── genuine/
│    ├── forged/
│
│── ui/
│    ├── templates/
│    │     ├── index.html
│    │     ├── register.html
│    ├── static/
│         ├── style.css
│    ├── app.py
│
│── README.md
│── requirements.txt

🌍 Other Uses of This Project

SignGuard can be used in:

🔹 Banking & Finance

Cheque signature verification

Fraud prevention

🔹 Legal & Government

Contract verification

Document authentication

🔹 Educational Institutions

Certificate validation

Exam attendance verification

🔹 Corporate / HR

Approvals and onboarding documents

🔹 Forensic Analysis

Detect forged handwriting

Court evidence validation

🔹 Logistics

Delivery signature verification

🔮 Future Scope

Implement Siamese Neural Network for signature matching

Mobile app integration

Cloud deployment (AWS, Heroku)

Multi-signature comparison

Real-time digital pad signature verification

👨‍💻 Contributors

Sahib Singh
B.Tech CSE – Final Year
Developer & Researcher
