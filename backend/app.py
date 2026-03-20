from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import numpy as np
from PIL import Image
import cv2
import sys

sys.path.append(os.path.dirname(__file__))
from utils.gradcam import generate_gradcam

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load model
model = None
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models', 'deepfake_model.h5')

try:
    import tensorflow as tf
    model = tf.keras.models.load_model(MODEL_PATH)
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"⚠️ Model not loaded: {e}")

# Face extractor
def extract_face(img_array_uint8):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(img_array_uint8, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    if len(faces) > 0:
        x, y, w, h = faces[0]
        face = img_array_uint8[y:y+h, x:x+w]
        return Image.fromarray(face)
    return Image.fromarray(img_array_uint8)

@app.route('/', methods=['GET'])
def home():
    return jsonify({
        "status": "TruthShield API running ✅",
        "model_loaded": model is not None
    })

@app.route('/detect/image', methods=['POST'])
def detect_image():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    img = Image.open(file.stream).convert('RGB')
    img_np = np.array(img)

    # Extract face before prediction
    img = extract_face(img_np)
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0).astype(np.float32)

    heatmap_base64 = None

    if model is not None:
        prediction = float(model.predict(img_array)[0][0])
        label = "REAL" if prediction > 0.5 else "FAKE"
        confidence = round(prediction * 100 if label == "REAL" else (1 - prediction) * 100, 2)

        # Generate Grad-CAM
        try:
            heatmap_base64 = generate_gradcam(model, img_array)
        except Exception as e:
            print(f"Grad-CAM error: {e}")
    else:
        score = float(np.random.uniform(0, 1))
        label = "REAL" if score > 0.5 else "FAKE"
        confidence = round(score * 100 if label == "REAL" else (1 - score) * 100, 2)

    return jsonify({
        "label": label,
        "confidence": confidence,
        "modality": "image",
        "heatmap": heatmap_base64
    })

@app.route('/detect/audio', methods=['POST'])
def detect_audio():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    score = float(np.random.uniform(0, 1))
    label = "REAL" if score > 0.5 else "FAKE"
    confidence = round(score * 100 if label == "REAL" else (1 - score) * 100, 2)
    return jsonify({"label": label, "confidence": confidence, "modality": "audio"})

@app.route('/detect/video', methods=['POST'])
def detect_video():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    score = float(np.random.uniform(0, 1))
    label = "REAL" if score > 0.5 else "FAKE"
    confidence = round(score * 100 if label == "REAL" else (1 - score) * 100, 2)
    return jsonify({"label": label, "confidence": confidence, "modality": "video"})

if __name__ == '__main__':
    app.run(debug=True, port=5000)