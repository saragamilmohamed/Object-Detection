import streamlit as st
import cv2
import numpy as np
import tempfile
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# =========================
# Visualization function
# =========================
def visualize(image, detection_result):
    for detection in detection_result.detections:
        bbox = detection.bounding_box
        start_point = (bbox.origin_x, bbox.origin_y)
        end_point = (bbox.origin_x + bbox.width, bbox.origin_y + bbox.height)
        cv2.rectangle(image, start_point, end_point, (0, 255, 0), 2)

        # Draw label
        category = detection.categories[0]
        category_name = category.category_name
        score = round(category.score, 2)
        result_text = f'{category_name} ({score})'
        cv2.putText(image, result_text, (bbox.origin_x, max(0, bbox.origin_y - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return image

# =========================
# Streamlit App
# =========================
st.title("📦 Object Detection with MediaPipe EfficientDet")
st.markdown("Upload a video to detect objects using MediaPipe's EfficientDet-Lite0 model.")

uploaded_file = st.file_uploader("Upload MP4 video", type=["mp4"])

if uploaded_file:
    # Save uploaded video to a temp file
    temp_video = tempfile.NamedTemporaryFile(delete=False)
    temp_video.write(uploaded_file.read())

    # Load MediaPipe model
    model_path = "efficientdet_lite0.tflite"
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.ObjectDetectorOptions(base_options=base_options, score_threshold=0.5)
    detector = vision.ObjectDetector.create_from_options(options)

    # Read and process video
    cap = cv2.VideoCapture(temp_video.name)
    st.video(temp_video.name)  # show original

    st.markdown("### Processed Frames:")
    frame_count = 0
    max_frames = 100  # limit number of displayed frames to avoid overload

    while cap.isOpened() and frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        # Run detection
        detection_result = detector.detect(mp_image)

        # Draw detections
        annotated_frame = visualize(frame.copy(), detection_result)

        # Show in Streamlit
        st.image(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB), channels="RGB")
        frame_count += 1

    cap.release()
