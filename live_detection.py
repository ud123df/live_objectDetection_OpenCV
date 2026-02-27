import streamlit as st
import cv2
from ultralytics import YOLO

st.title("Live YOLO Object Detection")

model = YOLO("yolov8n.pt")

run = st.checkbox("Start Camera")

FRAME_WINDOW = st.image([])

camera = cv2.VideoCapture(0)

while run:
    ret, frame = camera.read()
    if not ret:
        st.error("Failed to access camera.")
        break

    # Run YOLO
    results = model(frame)

    # Draw bounding boxes
    annotated_frame = results[0].plot()

    # Convert BGR to RGB
    annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

    FRAME_WINDOW.image(annotated_frame)

camera.release()