import time
import streamlit as st
import cv2
import tempfile
import os
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from ultralytics import YOLO
from mpl_toolkits.mplot3d import Axes3D

# -----------------------------
# Load YOLOv8 Model
# -----------------------------
MODEL_PATH = "best.pt"

model = YOLO(MODEL_PATH)

# -----------------------------
# Streamlit Page Config
# -----------------------------
st.set_page_config(page_title="Traffic Detection App", layout="centered")

st.title("🚦 Traffic Detection with YOLOv8")
st.write("Test your trained model on uploaded video or live webcam")

# Sidebar Options
option = st.sidebar.selectbox(
    "Choose an option",
    ["Video Upload", "Live Webcam"]
)

conf_threshold = st.sidebar.slider(
    "Confidence Threshold",
    0.1, 1.0, 0.25, 0.05
)

# -----------------------------
# Process Frame Function
# -----------------------------
def process_frame(
    frame,
    frame_count,
    start_time,
    detection_frames,
    detection_times,
    detection_confidences,
    bbox_areas,
    frame_numbers,
    x_centers,
    y_centers,
    class_names_list
):

    results = model.predict(frame, conf=conf_threshold)

    annotated_frame = results[0].plot()

    current_time = time.time() - start_time

    for box in results[0].boxes:

        conf = float(box.conf[0])
        cls = int(box.cls[0])

        x1, y1, x2, y2 = map(int, box.xyxy[0])

        # Bounding Box Area
        area = (x2 - x1) * (y2 - y1)

        # Center Coordinates
        x_center = (x1 + x2) // 2
        y_center = (y1 + y2) // 2

        detection_frames.append(frame_count)
        detection_times.append(current_time)
        detection_confidences.append(conf)
        bbox_areas.append(area)
        frame_numbers.append(frame_count)
        x_centers.append(x_center)
        y_centers.append(y_center)

        class_name = model.names[cls]
        class_names_list.append(class_name)

    return annotated_frame

# -----------------------------
# Plot Graphs Function
# -----------------------------
def plot_all_graphs(
    confidences,
    x_centers,
    y_centers,
    frame_numbers,
    bbox_areas,
    total_frames,
    class_names_list
):

    # 1. Confidence Histogram
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(confidences, bins=10)
    ax.set_title("Confidence Distribution")
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Frequency")
    st.pyplot(fig)

    # 2. Detection Timeline
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(frame_numbers, confidences, marker='o')
    ax.set_title("Detection Confidence Over Frames")
    ax.set_xlabel("Frame Number")
    ax.set_ylabel("Confidence")
    st.pyplot(fig)

    # 3. Bounding Box Area Distribution
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(bbox_areas, bins=10)
    ax.set_title("Bounding Box Area Distribution")
    ax.set_xlabel("Area")
    ax.set_ylabel("Count")
    st.pyplot(fig)

    # 4. Object Trajectory Plot
    fig, ax = plt.subplots(figsize=(6, 4))
    scatter = ax.scatter(
        x_centers,
        y_centers,
        c=range(len(x_centers)),
        cmap="viridis"
    )

    ax.set_xlabel("X Center")
    ax.set_ylabel("Y Center")
    ax.set_title("Object Trajectory (XY Position)")
    plt.colorbar(scatter, ax=ax, label="Frame Progression")
    st.pyplot(fig)

    # 5. 3D Scatter Plot
    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(
        x_centers,
        y_centers,
        confidences,
        c=confidences,
        cmap="coolwarm"
    )

    ax.set_xlabel("X Center")
    ax.set_ylabel("Y Center")
    ax.set_zlabel("Confidence")
    ax.set_title("3D Scatter of Detections")

    st.pyplot(fig)

    # 6. Confidence Boxplot
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.boxplot(confidences, vert=True, patch_artist=True)

    ax.set_ylabel("Confidence")
    ax.set_title("Confidence Value Distribution")
    st.pyplot(fig)

    # 7. Detection Coverage Pie Chart
    detected_frames = len(set(frame_numbers))
    non_detected = total_frames - detected_frames

    labels = ["Frames with Detections", "Frames without Detections"]
    values = [detected_frames, max(0, non_detected)]

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.pie(
        values,
        labels=labels,
        autopct='%1.1f%%',
        colors=["#4CAF50", "#FFC107"]
    )

    ax.set_title("Detection Coverage Across Video")
    st.pyplot(fig)

    # 8. Class Occurrence Chart
    if len(class_names_list) > 0:

        class_counts = Counter(class_names_list)

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(class_counts.keys(), class_counts.values(), color="teal")

        ax.set_xlabel("Class")
        ax.set_ylabel("Occurrences")
        ax.set_title("Object Occurrences")

        st.pyplot(fig)

        fig, ax = plt.subplots(figsize=(5, 5))

        ax.pie(
            class_counts.values(),
            labels=class_counts.keys(),
            autopct='%1.1f%%'
        )

        ax.set_title("Class Distribution")
        st.pyplot(fig)

# -----------------------------
# VIDEO UPLOAD
# -----------------------------
if option == "Video Upload":

    st.subheader("Upload Image or Video")

    uploaded_file = st.file_uploader(
        "Choose an image or video file",
        type=["jpg", "jpeg", "png", "mp4", "avi", "mov"]
    )

    if uploaded_file is not None:

        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())

        detection_confidences = []
        x_centers, y_centers = [], []
        detection_frames, detection_times = [], []
        bbox_areas, frame_numbers = [], []
        class_names_list = []

        frame_count = 0
        start_time = time.time()
        total_frames = 0

        if uploaded_file.type.startswith("video"):

            cap = cv2.VideoCapture(tfile.name)

            stframe = st.empty()

            while cap.isOpened():

                ret, frame = cap.read()

                if not ret:
                    break

                total_frames += 1

                annotated_frame = process_frame(
                    frame,
                    frame_count,
                    start_time,
                    detection_frames,
                    detection_times,
                    detection_confidences,
                    bbox_areas,
                    frame_numbers,
                    x_centers,
                    y_centers,
                    class_names_list
                )

                stframe.image(annotated_frame, channels="BGR")

                frame_count += 1

            cap.release()

        else:

            img = cv2.imread(tfile.name)

            annotated_frame = process_frame(
                img,
                frame_count,
                start_time,
                detection_frames,
                detection_times,
                detection_confidences,
                bbox_areas,
                frame_numbers,
                x_centers,
                y_centers,
                class_names_list
            )

            st.image(annotated_frame, channels="BGR")

            total_frames = 1

        # Show Graphs
        if len(detection_confidences) > 0:

            st.subheader("📊 Detection Analysis Graphs")

            plot_all_graphs(
                detection_confidences,
                x_centers,
                y_centers,
                frame_numbers,
                bbox_areas,
                total_frames,
                class_names_list
            )

# -----------------------------
# LIVE WEBCAM
# -----------------------------
elif option == "Live Webcam":

    st.subheader("🎥 Test on Live Webcam")

    run_webcam = st.checkbox("Start Webcam")

    if run_webcam:

        cap = cv2.VideoCapture(0)

        stframe = st.empty()

        detection_confidences = []
        x_centers, y_centers = [], []
        frame_numbers = []

        detection_frames = []
        detection_times = []
        bbox_areas = []

        class_names_list = []

        frame_count = 0
        start_time = time.time()
        total_frames = 0

        while cap.isOpened():

            ret, frame = cap.read()

            if not ret:
                st.error("❌ Failed to capture webcam frame.")
                break

            total_frames += 1

            annotated_frame = process_frame(
                frame,
                frame_count,
                start_time,
                detection_frames,
                detection_times,
                detection_confidences,
                bbox_areas,
                frame_numbers,
                x_centers,
                y_centers,
                class_names_list
            )

            annotated_frame = cv2.cvtColor(
                annotated_frame,
                cv2.COLOR_BGR2RGB
            )

            stframe.image(annotated_frame, channels="RGB")

            frame_count += 1

            # Stop webcam when unchecked
            if not run_webcam:
                break

        cap.release()

        # Show Graphs
        if len(detection_confidences) > 0:

            st.subheader("📊 Detection Analysis Graphs (Webcam)")

            plot_all_graphs(
                detection_confidences,
                x_centers,
                y_centers,
                frame_numbers,
                bbox_areas,
                total_frames,
                class_names_list
            )