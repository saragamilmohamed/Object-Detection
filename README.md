# Object-Detection

# Streamlit Video Detection App

This project is a Streamlit web application that allows users to upload a video and choose between two real-time video detection models:

1. Face and Eye Detection using OpenCV Haar Cascades
2. Object Detection using MediaPipe EfficientDet Lite

## Features

- Upload any `.mp4` video file
- Choose between face/eye detection or object detection
- Stream processed video with detection boxes and labels
- Easy-to-use web interface with Streamlit

## Project Structure
video-detection-app/
├── app.py # Main Streamlit app
├── app_face_detection.py # Face & Eye detection logic
├── app_object_detection.py # Object detection logic
├── efficientdet_lite0.tflite # MediaPipe model file
├── requirements.txt # Dependencies
└── README.md



---

### ✅ `requirements.txt`

```txt
streamlit
opencv-python
mediapipe
numpy

