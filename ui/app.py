from pathlib import Path

from flask import (
    Flask,
    render_template,
    request,
    send_from_directory,
)
from werkzeug.utils import secure_filename

from src.verify_signature import (
    verify_signature,
    load_svm_model,
)
from src.cnn_model import load_cnn_model

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
UPLOAD_FOLDER = BASE_DIR / "data" / "uploads"
USER_FOLDER = BASE_DIR / "data" / "user_signatures"
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
USER_FOLDER.mkdir(parents=True, exist_ok=True)

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = str(UPLOAD_FOLDER)
app.config["USER_FOLDER"] = str(USER_FOLDER)

# Load models once at startup
svm_model = load_svm_model()
try:
    cnn_model = load_cnn_model()
except FileNotFoundError:
    cnn_model = None  # UI will handle this


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


# ---------- Home: verification ----------

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    filename = None
    error = None
    conf = None
    selected_model = "svm"

    if request.method == "POST":
        selected_model = request.form.get("model_type", "svm")

        if "file" not in request.files:
            error = "No file part in the request."
        else:
            file = request.files["file"]

            if file.filename == "":
                error = "Please select an image file."
            elif file and allowed_file(file.filename):
                fname = secure_filename(file.filename)
                save_path = UPLOAD_FOLDER / fname
                file.save(save_path)

                # choose model
                if selected_model == "cnn":
                    if cnn_model is None:
                        error = "CNN model is not trained yet. Train it first."
                    else:
                        label, confidence = verify_signature(
                            save_path,
                            model_type="cnn",
                            cnn_model=cnn_model,
                        )
                else:
                    label, confidence = verify_signature(
                        save_path,
                        model_type="svm",
                        svm_model=svm_model,
                    )

                if error is None:
                    result = label
                    conf = round(confidence * 100, 2)
                    filename = fname
            else:
                error = "Unsupported file type. Please upload PNG / JPG / JPEG."

    return render_template(
        "index.html",
        result=result,
        filename=filename,
        error=error,
        confidence=conf,
        selected_model=selected_model,
        cnn_available=(cnn_model is not None),
    )


# ---------- Register user's own signature ----------

@app.route("/register", methods=["GET", "POST"])
def register():
    message = None
    error = None

    if request.method == "POST":
        name = request.form.get("name", "").strip()
        user_id = request.form.get("user_id", "").strip()
        note = request.form.get("note", "").strip()
        file = request.files.get("file")

        if not name or not user_id:
            error = "Name and ID are required."
        elif not file or file.filename == "":
            error = "Please upload a signature image."
        elif not allowed_file(file.filename):
            error = "Unsupported file type. Please upload PNG / JPG / JPEG."
        else:
            fname = f"{user_id}_{secure_filename(name)}_{secure_filename(file.filename)}"
            save_path = USER_FOLDER / fname
            file.save(save_path)

            # append simple log
            log_path = USER_FOLDER / "registry.csv"
            with log_path.open("a", encoding="utf-8") as f:
                f.write(f'"{user_id}","{name}","{note}","{save_path.as_posix()}"\n')

            message = "Signature registered successfully!"

    return render_template(
        "register.html",
        message=message,
        error=error,
    )


@app.route("/uploads/<path:filename>")
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)


if __name__ == "__main__":
    app.run(debug=True)
