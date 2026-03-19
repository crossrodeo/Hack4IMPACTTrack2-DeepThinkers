# Hack4IMPACTTrack2-DeepThinkers

TruthShield — Deepfake & AI Misinformation Detection | HACK4IMPACT Track 2 | Team Deep Thinkers | KIIT Bhubaneswar

---

# TruthShield 🛡️
**Real-Time Multimodal Deepfake & AI Misinformation Detection**

We're Team Deep Thinkers from KIIT, and this is our submission for HACK4IMPACT Track 2.
The problem we're tackling is simple to state but hard to solve — how do you know
if what you're seeing online is real?

---

## The Team

| Name |
|------|
| Prakhar Patel |
| Amrutha Jampala |
| Abhiamrit Veera |

**College:** KIIT Deemed to be University, Bhubaneswar
**Hackathon:** HACK4IMPACT Track 2 — Smart Technology for a Sustainable World
**Domain:** Cybersecurity & Ethical AI Systems

---

## The Problem

Deepfakes are getting scary good. AI-generated videos, cloned voices, and synthetic
images are already being used to spread misinformation, commit fraud, and manipulate
public opinion — and most people have no way to tell the difference.

Existing detection tools are either too expensive, too slow, or just a black box that
gives you a "fake" label with zero explanation. That's not good enough, especially
when the stakes involve elections, legal evidence, or someone's reputation.

---

## What We're Building

TruthShield is a web app that lets you upload a video clip, audio file, or image and
get a verdict — Real or Fake — along with a visual explanation of *why* the model
thinks so.

Under the hood:
- **Video** → CNN + LSTM on frame sequences
- **Audio** → Spectrogram-based anomaly classifier
- **Image** → ResNet with GAN artifact detection
- **Explainability** → Grad-CAM heatmaps so you can see exactly what triggered the flag

The XAI layer is the part we're most proud of. Too many AI tools just say "trust me."
We wanted ours to show its work.

---

## Tech Stack

- **Languages:** Python, JavaScript
- **Frameworks:** TensorFlow/Keras, Flask, React.js
- **Libraries:** OpenCV, Librosa, Grad-CAM, NumPy
- **Datasets:** FaceForensics++, Celeb-DF, ASVspoof
- **Tools:** Docker, Hugging Face, GitHub

---

## Status

🚧 Actively building — HACK4IMPACT 2026, 43-hour sprint
