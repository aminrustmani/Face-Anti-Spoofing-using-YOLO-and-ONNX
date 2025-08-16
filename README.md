## 🔹 Project Description

This project implements a real-time Face Anti-Spoofing system that detects whether a face is real or spoofed (e.g., printed photo, replay attack).
It combines two ONNX models:

YOLOv5 Face Detector (ONNX) → Detects and localizes faces.

Anti-Spoofing Classifier (ONNX) → Classifies each detected face as Real or Spoof.

The system is fully extended for real-time webcam input using OpenCV + ONNX Runtime, making it lightweight, portable, and ready for deployment.

# 🔹 Features

Real-time webcam-based face anti-spoofing

Accurate face detection with YOLOv5 ONNX

Anti-spoofing classification with pre-trained ONNX model

End-to-end Python pipeline

Annotated results with bounding boxes & labels (Real/Spoof)

# 🔹 Tech Stack

Python 3.x

OpenCV

NumPy

ONNX Runtime

YOLOv5 (exported to ONNX)

🔹 Installation

Clone repository:

git clone https://github.com/yourusername/face-anti-spoofing-yolo.git
cd face-anti-spoofing-yolo


Install dependencies:

pip install -r requirements.txt


requirements.txt

opencv-python
numpy
onnxruntime

# 🔹 Usage
Run real-time webcam detection
python src/main.py


Press Q to quit the webcam window.

# 🔹 Example Output

Bounding box drawn around detected face.

Label above face: Real (0.95) or Spoof (0.87).

Runs in real-time using webcam feed.

# 🔹 How It Works

Face Detection → YOLOv5 ONNX detects face bounding boxes.

Face Cropping → Extract detected face region.

Spoof Classification → ONNX model predicts Real or Spoof.

Annotation → Results displayed live with bounding boxes & labels.

# 🔹 Future Improvements

Add multi-face support for crowded scenes.

Optimize for GPU inference.

Export to TensorRT / TFLite / CoreML for deployment on edge devices.

# 🔹 License

MIT License – Free to use and modify.
