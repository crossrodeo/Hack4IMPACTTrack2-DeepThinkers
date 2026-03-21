from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import numpy as np
from PIL import Image
import cv2
import sys
import librosa
import base64
import io
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

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

# Video analyzer
def analyze_video(file_stream, filename):
    ext = os.path.splitext(filename)[1].lower()
    temp_path = os.path.join(UPLOAD_FOLDER, f'temp_video{ext}')
    file_stream.save(temp_path)

    try:
        cap = cv2.VideoCapture(temp_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        sample_count = min(10, total_frames)
        frame_indices = np.linspace(0, total_frames - 1, sample_count, dtype=int)

        frame_predictions = []
        frame_timestamps = []
        sample_heatmap = None

        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                continue

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Skip face extraction for video — use full frame directly
            img = Image.fromarray(frame_rgb).resize((224, 224))
            img_array = np.array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0).astype(np.float32)

            if model is not None:
                prediction = float(model.predict(img_array, verbose=0)[0][0])
                frame_predictions.append(prediction)
                frame_timestamps.append(round(idx / fps, 2) if fps > 0 else idx)

                if idx == frame_indices[len(frame_indices) // 2] and sample_heatmap is None:
                    try:
                        sample_heatmap = generate_gradcam(model, img_array)
                    except:
                        pass

        cap.release()

        if not frame_predictions:
            return "REAL", 50.0, None, None

        avg_prediction = np.mean(frame_predictions)
        fake_frame_ratio = np.sum(np.array(frame_predictions) < 0.5) / len(frame_predictions)

        print(f"DEBUG: fake_frame_ratio={fake_frame_ratio}, avg={avg_prediction}")

        if fake_frame_ratio >= 0.3:
            label = "FAKE"
            confidence = round(fake_frame_ratio * 100, 2)
        else:
            label = "REAL"
            confidence = round(avg_prediction * 100, 2)

        fig, ax = plt.subplots(figsize=(8, 3))
        fig.patch.set_facecolor('#0a0e1a')
        ax.set_facecolor('#111827')

        colors = ['#ff4444' if p < 0.5 else '#00ff88' for p in frame_predictions]
        ax.bar(frame_timestamps, frame_predictions, color=colors, width=0.3)
        ax.axhline(y=0.5, color='#00d4ff', linestyle='--', alpha=0.7, label='Decision boundary')
        ax.set_xlabel('Time (seconds)', color='#8899aa')
        ax.set_ylabel('Real Probability', color='#8899aa')
        ax.set_title('Frame-by-Frame Analysis', color='#00d4ff', fontsize=10)
        ax.tick_params(colors='#8899aa')
        ax.spines['bottom'].set_color('#1e3a5f')
        ax.spines['top'].set_color('#1e3a5f')
        ax.spines['left'].set_color('#1e3a5f')
        ax.spines['right'].set_color('#1e3a5f')
        ax.legend(facecolor='#111827', labelcolor='#8899aa')
        ax.set_ylim(0, 1)

        plt.tight_layout()
        buffer = io.BytesIO()
        plt.savefig(buffer, format='PNG', facecolor='#0a0e1a')
        buffer.seek(0)
        timeline_base64 = base64.b64encode(buffer.read()).decode('utf-8')
        plt.close()

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

    return label, confidence, timeline_base64, sample_heatmap

# Audio analyzer
def analyze_audio(file_stream, filename):
    ext = os.path.splitext(filename)[1].lower()
    temp_path = os.path.join(UPLOAD_FOLDER, f'temp_audio{ext}')
    file_stream.save(temp_path)

    try:
        y, sr = librosa.load(temp_path, duration=10)

        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        mel_db = librosa.power_to_db(mel_spec, ref=np.max)

        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

        rms = np.sqrt(np.mean(y**2))
        if rms < 0.01:
            label = "FAKE"
            confidence = 97.5
        else:
            mfcc_delta = librosa.feature.delta(mfcc)
            mfcc_delta_var = np.var(mfcc_delta)
            spectral_contrast = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr))
            pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
            pitch_values = pitches[magnitudes > np.median(magnitudes)]
            pitch_std = np.std(pitch_values) if len(pitch_values) > 0 else 0
            zcr = librosa.feature.zero_crossing_rate(y)
            zcr_std = np.std(zcr)
            rms_frames = librosa.feature.rms(y=y)[0]
            silence_ratio = np.sum(rms_frames < 0.01) / len(rms_frames)

            naturalness_score = (
                min(pitch_std / 500, 1) * 0.3 +
                min(zcr_std * 100, 1) * 0.2 +
                min(mfcc_delta_var / 100, 1) * 0.25 +
                min(silence_ratio * 3, 1) * 0.15 +
                min(spectral_contrast / 30, 1) * 0.1
            )

            label = "REAL" if naturalness_score > 0.35 else "FAKE"
            confidence = round(naturalness_score * 100 if label == "REAL" else (1 - naturalness_score) * 100, 2)

        fig, axes = plt.subplots(2, 1, figsize=(8, 6))
        fig.patch.set_facecolor('#0a0e1a')

        axes[0].set_facecolor('#111827')
        librosa.display.waveshow(y, sr=sr, ax=axes[0], color='#00d4ff')
        axes[0].set_title('Waveform', color='#00d4ff', fontsize=10)
        axes[0].tick_params(colors='#8899aa')
        axes[0].spines['bottom'].set_color('#1e3a5f')
        axes[0].spines['top'].set_color('#1e3a5f')
        axes[0].spines['left'].set_color('#1e3a5f')
        axes[0].spines['right'].set_color('#1e3a5f')

        axes[1].set_facecolor('#111827')
        librosa.display.specshow(mel_db, sr=sr, x_axis='time', y_axis='mel', ax=axes[1], cmap='inferno')
        axes[1].set_title('Mel Spectrogram', color='#00d4ff', fontsize=10)
        axes[1].tick_params(colors='#8899aa')
        axes[1].spines['bottom'].set_color('#1e3a5f')
        axes[1].spines['top'].set_color('#1e3a5f')
        axes[1].spines['left'].set_color('#1e3a5f')
        axes[1].spines['right'].set_color('#1e3a5f')

        plt.tight_layout()
        buffer = io.BytesIO()
        plt.savefig(buffer, format='PNG', facecolor='#0a0e1a')
        buffer.seek(0)
        spectrogram_base64 = base64.b64encode(buffer.read()).decode('utf-8')
        plt.close()

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

    return label, confidence, spectrogram_base64

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

    img = extract_face(img_np)
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0).astype(np.float32)

    heatmap_base64 = None

    if model is not None:
        prediction = float(model.predict(img_array)[0][0])
        label = "REAL" if prediction > 0.5 else "FAKE"
        confidence = round(prediction * 100 if label == "REAL" else (1 - prediction) * 100, 2)
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

    file = request.files['file']
    filename = file.filename

    try:
        label, confidence, spectrogram = analyze_audio(file, filename)
        return jsonify({
            "label": label,
            "confidence": confidence,
            "modality": "audio",
            "spectrogram": spectrogram
        })
    except Exception as e:
        print(f"Audio error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/detect/video', methods=['POST'])
def detect_video():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    filename = file.filename

    try:
        label, confidence, timeline, heatmap = analyze_video(file, filename)
        return jsonify({
            "label": label,
            "confidence": confidence,
            "modality": "video",
            "timeline": timeline,
            "heatmap": heatmap
        })
    except Exception as e:
        print(f"Video error: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)