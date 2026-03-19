from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import numpy as np
from PIL import Image
import io
import base64

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ─── Health Check ───────────────────────────────────────────
@app.route('/', methods=['GET'])
def home():
    return jsonify({"status": "TruthShield API is running ✅"})

# ─── Image Detection ────────────────────────────────────────
@app.route('/detect/image', methods=['POST'])
def detect_image():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    img = Image.open(file.stream).convert('RGB')
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Placeholder prediction (replace with real model later)
    score = float(np.random.uniform(0, 1))
    label = "FAKE" if score > 0.5 else "REAL"
    confidence = round(score * 100 if label == "FAKE" else (1 - score) * 100, 2)

    return jsonify({
        "label": label,
        "confidence": confidence,
        "modality": "image"
    })

# ─── Audio Detection ────────────────────────────────────────
@app.route('/detect/audio', methods=['POST'])
def detect_audio():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    score = float(np.random.uniform(0, 1))
    label = "FAKE" if score > 0.5 else "REAL"
    confidence = round(score * 100 if label == "FAKE" else (1 - score) * 100, 2)

    return jsonify({
        "label": label,
        "confidence": confidence,
        "modality": "audio"
    })

# ─── Video Detection ────────────────────────────────────────
@app.route('/detect/video', methods=['POST'])
def detect_video():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    score = float(np.random.uniform(0, 1))
    label = "FAKE" if score > 0.5 else "REAL"
    confidence = round(score * 100 if label == "FAKE" else (1 - score) * 100, 2)

    return jsonify({
        "label": label,
        "confidence": confidence,
        "modality": "video"
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)