# SignifyPlus Baseline - ASL Gesture Recognition

> Breaking down communication barriers through real-time American Sign Language translation

**SignifyPlus** is an AI-powered system that translates American Sign Language (ASL) gestures into text in real-time. This project addresses the critical communication gap between deaf and hearing communities by providing instant, accurate gesture recognition using computer vision and deep learning.

🚀 **[Live Demo](https://asl-translation-production.up.railway.app/)** | 📱 **[Mobile Version]([mobile-repo-link](https://github.com/kamilaalv/ASL-Gesture-Recognition-Mobile))** | 🏆 **Innovation Award Winner**

## The Problem

Over 70 million deaf people worldwide use sign language as their primary communication method, yet most hearing individuals cannot understand sign language. This creates significant barriers in:
- Healthcare settings
- Educational environments  
- Professional interactions
- Emergency situations
- Daily social interactions

## Our Solution

SignifyPlus uses advanced computer vision and neural networks to:
- **Real-time recognition**: Translate ASL gestures as they happen
- **High accuracy**: 52% mean average precision on gesture classification
- **Accessible technology**: Works with standard webcams and mobile devices
- **Scalable system**: Supports expanding vocabulary of signs

## How It Works

1. **Hand Detection**: MediaPipe extracts 21 3D hand landmarks from video feed
2. **Feature Engineering**: Converts landmark coordinates into meaningful geometric features
3. **Sequence Processing**: LSTM neural network captures temporal dynamics of gestures
4. **Real-time Prediction**: Outputs recognized signs with confidence scores

### Why LSTM for Sign Language?

Unlike static images, **sign language is inherently dynamic** - gestures unfold over time with specific movements, speeds, and trajectories that carry meaning. A single frame cannot capture the full context of a sign. This is why we use **LSTM (Long Short-Term Memory)** networks:

- **Temporal Memory**: LSTM remembers previous hand positions and movements, understanding how gestures evolve over time
- **Sequential Learning**: Captures the flow and rhythm of sign language, distinguishing between similar signs based on their temporal patterns
- **Context Awareness**: Understands that the same hand shape can mean different things depending on the movement sequence

For example, signs like "HELP" and "THANK YOU" might have similar hand shapes but completely different movement patterns - only a memory-enabled model can distinguish between them reliably.

![Architecture Diagram](docs/architecture_diagram.png)

## Key Features
- **Real-time processing** with webcam input
- **Confidence scoring** for prediction reliability
- **Sequence buffering** for temporal gesture recognition
- **Extensible architecture** for adding new signs

## Quick Start

### Online Demo
Try the system immediately: **[Live Demo](https://asl-translation-production.up.railway.app/)**

### Local Setup
```bash
git clone https://github.com/kamilaalv/ASL-Gesture-Recognition
cd ASL-Gesture-Recognition
pip install -r requirements.txt
python src/inference/dynamic_test_camera.py
```
