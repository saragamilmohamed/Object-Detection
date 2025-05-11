import streamlit as st
import cv2
import numpy as np
import tempfile
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Load Haar cascade classifiers
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

# Object detector model
@st.cache_resource
def load_mediapipe_detector():
    model_path = "efficientdet_lite0.tflite"
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.ObjectDetectorOptions(base_options=base_options, score_threshold=0.5)
    return vision.ObjectDetector.create_from_options(options)

# Object detection visualization
def visualize_objects(image, detection_result):
    for detection in detection_result.detections:
        bbox = detection.bounding_box
        start = (bbox.origin_x, bbox.origin_y)
        end = (bbox.origin_x + bbox.width, bbox.origin_y + bbox.height)
        cv2.rectangle(image, start, end, (0, 255, 0), 2)

        category = detection.categories[0]
        label = f"{category.category_name} ({round(category.score, 2)})"
        cv2.putText(image, label, (bbox.origin_x, max(0, bbox.origin_y - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return image

# Face & eye detection
def detect_faces_eyes(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 4)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 0, 255), 2)
    return frame

# Streamlit app
st.title("🎥 Video Detection App")
st.markdown("Upload a video and choose a model for detection.")

# Choose model
model_choice = st.selectbox("Choose Detection Model:", ["Face & Eye Detection", "Object Detection"])

uploaded_file = st.file_uploader("Upload MP4 video", type=["mp4"])

if uploaded_file:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    cap = cv2.VideoCapture(tfile.name)
    st.video(tfile.name)  # Optional: show original

    st.markdown("### Processed Frames:")
    frame_count = 0
    max_frames = 100  # Limit to prevent overload

    # Load object detector if needed
    if model_choice == "Object Detection":
        detector = load_mediapipe_detector()

    while cap.isOpened() and frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        if model_choice == "Face & Eye Detection":
            processed_frame = detect_faces_eyes(frame)
        else:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            detection_result = detector.detect(mp_image)
            processed_frame = visualize_objects(frame.copy(), detection_result)

        st.image(cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB), channels="RGB")
        frame_count += 1

    cap.release()
