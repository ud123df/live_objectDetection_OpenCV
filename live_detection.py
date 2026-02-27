import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av

st.title("🚀 Live YOLO Object Detection")

# Load lightweight YOLO model
model = YOLO("yolov8n.pt")


class YOLOVideoProcessor(VideoProcessorBase):
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")

        results = model(img)

        annotated_frame = results[0].plot()

        return av.VideoFrame.from_ndarray(annotated_frame, format="bgr24")


RTC_CONFIGURATION = {
    "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
}

webrtc_streamer(
    key="yolo-live",
    video_processor_factory=YOLOVideoProcessor,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={"video": True, "audio": False},
)
