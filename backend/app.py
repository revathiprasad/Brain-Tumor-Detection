from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import tensorflow as tf
from PIL import Image
import io

app = Flask(__name__)
CORS(app)

# ---------------------------------
# Class labels
# ---------------------------------
CLASSES = ['glioma', 'meningioma', 'notumor', 'pituitary']

# ---------------------------------
# Load models (ONLY 3)
# ---------------------------------
print("Loading models...")

cnn_model = tf.saved_model.load("cnn_baseline_savedmodel")
infer_cnn = cnn_model.signatures["serving_default"]

cnn_aug_model = tf.saved_model.load("cnn_baseline_aug_savedmodel")
infer_cnn_aug = cnn_aug_model.signatures["serving_default"]

vgg_model = tf.saved_model.load("vgg_model_savedmodel")
infer_vgg = vgg_model.signatures["serving_default"]

print("All 3 models loaded successfully.")

# ---------------------------------
# Health check
# ---------------------------------
@app.route("/", methods=["GET"])
def home():
    return jsonify({"status": "Backend running with 3 models"})

# ---------------------------------
# Prediction API
# ---------------------------------
@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    model_type = request.form.get("model")
    file = request.files["file"]

    # Image preprocessing
    img = Image.open(io.BytesIO(file.read())).convert("RGB")
    img = img.resize((224, 224))

    img_arr = np.array(img, dtype="float32") / 255.0
    img_arr = np.expand_dims(img_arr, axis=0)

    # Model selection
    if model_type == "cnn":
        outputs = infer_cnn(tf.constant(img_arr))
        model_used = "CNN"
    elif model_type == "cnn_aug":
        outputs = infer_cnn_aug(tf.constant(img_arr))
        model_used = "CNN Augmented"
    elif model_type == "vgg":
        outputs = infer_vgg(tf.constant(img_arr))
        model_used = "VGG16"
    else:
        return jsonify({"error": "Invalid model selection"}), 400

    preds = list(outputs.values())[0].numpy()[0]
    idx = int(np.argmax(preds))

    return jsonify({
        "model_used": model_used,
        "prediction": CLASSES[idx],
        "confidence": round(float(np.max(preds) * 100), 2)
    })

# ---------------------------------
# Run server
# ---------------------------------
if __name__ == "__main__":
    app.run(debug=True)
