import streamlit as st
import numpy as np
import cv2
import tempfile
import os

# Load the Haar cascades
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eyes_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

def detect_faces_and_eyes(video_path):
    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, 1.3, 4)
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w]
            eyes = eyes_cascade.detectMultiScale(roi_gray)
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 0, 255), 2)

        # Convert BGR to RGB for display
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        yield frame_rgb

    cap.release()

# Streamlit UI
st.title("🎥 Face and Eye Detection App")
st.markdown("Upload a video and detect faces and eyes using Haar Cascades.")

uploaded_file = st.file_uploader("Upload an MP4 video", type=["mp4"])

if uploaded_file is not None:
    # Save uploaded video to a temp file
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())

    st.video(tfile.name)  # Optional: show the original video

    st.markdown("### Processed Frames:")

    # Display processed frames
    frame_generator = detect_faces_and_eyes(tfile.name)
    for frame in frame_generator:
        st.image(frame, channels="RGB")
