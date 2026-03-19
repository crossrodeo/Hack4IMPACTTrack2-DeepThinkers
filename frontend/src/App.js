import React, { useState } from 'react';
import axios from 'axios';
import './App.css';

function App() {
  const [file, setFile] = useState(null);
  const [modality, setModality] = useState('image');
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [preview, setPreview] = useState(null);

  const handleFileChange = (e) => {
    const selected = e.target.files[0];
    setFile(selected);
    setResult(null);
    if (selected && modality === 'image') {
      setPreview(URL.createObjectURL(selected));
    } else {
      setPreview(null);
    }
  };

  const handleDetect = async () => {
    if (!file) return alert('Please upload a file first!');
    setLoading(true);
    setResult(null);
    const formData = new FormData();
    formData.append('file', file);
    try {
      const res = await axios.post(
        `http://127.0.0.1:5000/detect/${modality}`,
        formData,
        { headers: { 'Content-Type': 'multipart/form-data' } }
      );
      setResult(res.data);
    } catch (err) {
      alert('Error connecting to backend!');
    }
    setLoading(false);
  };

  return (
    <div className="app">
      <header className="header">
        <h1>🛡️ TruthShield</h1>
        <p>Real-Time Multimodal Deepfake & AI Misinformation Detection</p>
        <span className="team">Team Deep Thinkers · KIIT · HACK4IMPACT 2026</span>
      </header>

      <main className="main">
        <div className="card">
          <h2>Select Detection Mode</h2>
          <div className="modality-btns">
            {['image', 'audio', 'video'].map((m) => (
              <button
                key={m}
                className={`mod-btn ${modality === m ? 'active' : ''}`}
                onClick={() => {
                  setModality(m);
                  setFile(null);
                  setResult(null);
                  setPreview(null);
                }}
              >
                {m === 'image' ? '🖼️ Image' : m === 'audio' ? '🎵 Audio' : '🎬 Video'}
              </button>
            ))}
          </div>

          <div className="upload-box">
            <input
              type="file"
              accept={modality === 'image' ? 'image/*' : modality === 'audio' ? 'audio/*' : 'video/*'}
              onChange={handleFileChange}
              id="fileInput"
            />
            <label htmlFor="fileInput" className="upload-label">
              {file ? `✅ ${file.name}` : `⬆️ Upload ${modality} file`}
            </label>
          </div>

          {preview && (
            <div className="preview">
              <img src={preview} alt="Preview" />
            </div>
          )}

          <button className="detect-btn" onClick={handleDetect} disabled={loading}>
            {loading ? '🔍 Analyzing...' : '🔍 Detect'}
          </button>
        </div>

        {result && (
          <div className={`result-card ${result.label === 'FAKE' ? 'fake' : 'real'}`}>
            <h2>{result.label === 'FAKE' ? '⚠️ FAKE DETECTED' : '✅ CONTENT IS REAL'}</h2>
            <p className="confidence">Confidence: <strong>{result.confidence}%</strong></p>
            <p className="modality-tag">Modality: {result.modality.toUpperCase()}</p>
            <div className="confidence-bar">
              <div
                className="confidence-fill"
                style={{ width: `${result.confidence}%` }}
              />
            </div>
          </div>
        )}
      </main>

      <footer className="footer">
        <p>Powered by TensorFlow · Flask · React · Grad-CAM XAI</p>
      </footer>
    </div>
  );
}

export default App;
