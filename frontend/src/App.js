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
    } else if (selected && modality === 'video') {
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

  const getBiasAudit = (result) => {
    if (!result) return null;
    const confidence = result.confidence;
    return {
      model: 'MobileNetV2 (Transfer Learning)',
      dataset: 'FaceForensics++ / 140k Real & Fake Faces',
      threshold: '50% decision boundary',
      transparency: 'Grad-CAM XAI enabled',
      dataRetention: 'No user data stored',
      fairness: confidence > 85 ? 'High confidence — low ambiguity' : confidence > 60 ? 'Moderate confidence — review recommended' : 'Low confidence — human review advised',
      auditStatus: '✅ Open source & auditable',
    };
  };

  const bias = getBiasAudit(result);

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

          {preview && modality === 'image' && (
            <div className="preview">
              <img src={preview} alt="Preview" />
            </div>
          )}

          {preview && modality === 'video' && (
            <div className="preview">
              <video src={preview} controls style={{ width: '100%', borderRadius: '8px' }} />
            </div>
          )}

          <button className="detect-btn" onClick={handleDetect} disabled={loading}>
            {loading ? '🔍 Analyzing...' : '🔍 Detect'}
          </button>

          {loading && modality === 'video' && (
            <p className="loading-note">⏳ Analyzing video frames — this may take 30-60 seconds...</p>
          )}
        </div>

        {result && (
          <>
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

              {result.heatmap && (
                <div className="heatmap-section">
                  <h3>🔥 Grad-CAM Explainability Heatmap</h3>
                  <p className="heatmap-desc">Areas highlighted in red/yellow show regions the AI focused on to make its decision.</p>
                  <img
                    src={`data:image/png;base64,${result.heatmap}`}
                    alt="Grad-CAM Heatmap"
                    className="heatmap-img"
                  />
                </div>
              )}

              {result.spectrogram && (
                <div className="heatmap-section">
                  <h3>🎵 Audio Spectrogram Analysis</h3>
                  <p className="heatmap-desc">Waveform and Mel Spectrogram anomaly visualization for voice clone detection.</p>
                  <img
                    src={`data:image/png;base64,${result.spectrogram}`}
                    alt="Audio Spectrogram"
                    className="heatmap-img"
                    style={{ maxWidth: '100%' }}
                  />
                </div>
              )}

              {result.timeline && (
                <div className="heatmap-section">
                  <h3>🎬 Frame-by-Frame Analysis</h3>
                  <p className="heatmap-desc">Green bars = REAL frames, Red bars = FAKE frames. Blue line = decision boundary.</p>
                  <img
                    src={`data:image/png;base64,${result.timeline}`}
                    alt="Video Timeline"
                    className="heatmap-img"
                    style={{ maxWidth: '100%' }}
                  />
                </div>
              )}
            </div>

            {bias && (
              <div className="bias-card">
                <h2>🔎 Bias Audit & Transparency Panel</h2>
                <p className="bias-subtitle">TruthShield is committed to ethical, explainable AI. Here's how this decision was made.</p>
                <div className="bias-grid">
                  <div className="bias-item">
                    <span className="bias-label">🤖 Model</span>
                    <span className="bias-value">{bias.model}</span>
                  </div>
                  <div className="bias-item">
                    <span className="bias-label">📦 Training Data</span>
                    <span className="bias-value">{bias.dataset}</span>
                  </div>
                  <div className="bias-item">
                    <span className="bias-label">⚖️ Decision Threshold</span>
                    <span className="bias-value">{bias.threshold}</span>
                  </div>
                  <div className="bias-item">
                    <span className="bias-label">🔍 Explainability</span>
                    <span className="bias-value">{bias.transparency}</span>
                  </div>
                  <div className="bias-item">
                    <span className="bias-label">🔒 Privacy</span>
                    <span className="bias-value">{bias.dataRetention}</span>
                  </div>
                  <div className="bias-item">
                    <span className="bias-label">📊 Confidence Assessment</span>
                    <span className="bias-value">{bias.fairness}</span>
                  </div>
                  <div className="bias-item full-width">
                    <span className="bias-label">🌐 Audit Status</span>
                    <span className="bias-value">{bias.auditStatus}</span>
                  </div>
                </div>
              </div>
            )}
          </>
        )}
      </main>

      <footer className="footer">
        <p>Powered by TensorFlow · Flask · React · Grad-CAM XAI</p>
      </footer>
    </div>
  );
}

export default App;