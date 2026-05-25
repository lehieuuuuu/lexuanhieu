import os

# ================= FIX KERAS CŨ =================
os.environ["TF_USE_LEGACY_KERAS"] = "1"

# ================= TẮT LOG =================
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# ================= IMPORT =================
from flask import Flask, render_template, request, jsonify

import tensorflow as tf
from tf_keras.models import load_model

from PIL import Image, ImageOps
import numpy as np
import base64

# ================= FLASK =================
app = Flask(__name__)

# ================= LOAD MODEL =================
model = load_model("keras_Model.h5", compile=False)

# ================= LOAD LABELS =================
class_names = open("labels.txt", "r", encoding="utf-8").readlines()

print("Model loaded!")
print(class_names)

# ================= HÀM DỰ ĐOÁN =================
def predict_image(image_path):

    # Tạo array đúng shape model
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

    # Mở ảnh
    image = Image.open(image_path).convert("RGB")

    # Resize chuẩn Teachable Machine
    size = (224, 224)

    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

    # Convert sang numpy
    image_array = np.asarray(image)

    # Normalize
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

    # Load vào array
    data[0] = normalized_image_array

    # Predict
    prediction = model.predict(data, verbose=0)

    index = np.argmax(prediction)

    class_name = class_names[index].strip()

    confidence_score = float(prediction[0][index])

    # Xóa số phía trước label
    if len(class_name) > 2:
        class_name = class_name[2:]

    # Nếu confidence thấp
    if confidence_score < 0.55:
        class_name = "Không nhận diện rõ"

    return class_name, confidence_score


# ================= HOME =================
@app.route("/")
def index():
    return render_template("index.html")


# ================= WEBCAM =================
@app.route("/predict_webcam", methods=["POST"])
def predict_webcam():

    try:

        data_url = request.form["webcam_image"]

        header, encoded = data_url.split(",", 1)

        binary_data = base64.b64decode(encoded)

        os.makedirs("static", exist_ok=True)

        path = os.path.join("static", "webcam.jpg")

        with open(path, "wb") as f:
            f.write(binary_data)

        label, confidence = predict_image(path)

        result = f"{label} ({confidence:.2%})"

        return jsonify({"result": result})

    except Exception as e:
        return jsonify({"result": str(e)})


# ================= UPLOAD =================
@app.route("/predict_upload", methods=["POST"])
def predict_upload():

    try:

        if "upload_image" not in request.files:
            return jsonify({"result": "Không có file upload"})

        file = request.files["upload_image"]

        os.makedirs("static", exist_ok=True)

        path = os.path.join("static", "upload.jpg")

        file.save(path)

        label, confidence = predict_image(path)

        result = f"{label} ({confidence:.2%})"

        return jsonify({"result": result})

    except Exception as e:
        return jsonify({"result": str(e)})


# ================= MAIN =================
if __name__ == "__main__":
    app.run(debug=True)