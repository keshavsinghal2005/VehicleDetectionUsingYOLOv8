#  Vehicle Detection Using YOLOv8

##  Overview

Vehicle Detection Using YOLOv8 is a real-time computer vision system designed for intelligent traffic monitoring and vehicle analytics. The project leverages the **YOLOv8 deep learning object detection framework** with OpenCV and Streamlit to detect, classify, and visualize vehicles from images, videos, and live streams with high accuracy and low latency.

The system provides an interactive AI-powered detection interface capable of processing traffic surveillance footage and generating real-time detection outputs for transportation analytics and smart monitoring applications.


# Deployed Project Link

https://vehicledetectionusingyolov8.streamlit.app/


#  Features

* Real-time vehicle detection using YOLOv8
* Video, image, and webcam inference support
* Deep learning–based object localization and classification
* Interactive Streamlit web application
* Bounding box visualization with confidence scoring
* Support for traffic surveillance analytics
* GPU/CPU-compatible inference pipeline
* Custom-trained YOLOv8 model integration
* Detection output visualization and processing



#  Technologies Used

* Python
* YOLOv8
* Ultralytics
* OpenCV
* Streamlit
* Deep Learning
* Computer Vision
* NumPy
* Matplotlib
* PyTorch


#  System Architecture

```text id="jlwm2b"
Input Video/Image/Webcam
            ↓
Frame Extraction using OpenCV
            ↓
YOLOv8 Deep Learning Inference
            ↓
Vehicle Detection & Classification
            ↓
Bounding Box Rendering
            ↓
Real-Time Visualization via Streamlit
```



#  Project Structure

```text id="jlwmy0"
VehicleDetectionUsingYOLOv8/
│
├── app.py
├── best.pt
├── requirements.txt
├── dataset.yaml
├── runs/
├── DataSet YOLO/
├── TestVideo/
└── README.md
```



#  Model Training

The YOLOv8 model was trained on a custom vehicle detection dataset using the Ultralytics framework.

### Training Pipeline

* Dataset preprocessing and annotation
* YOLO-format dataset preparation
* Transfer learning using pretrained YOLOv8n weights
* Custom training using Ultralytics API

### Training Configuration

```python id="jlwm94"
model.train(
    data="dataset.yaml",
    imgsz=64,
    epochs=10,
    batch=8
)
```



#  Running the Project

## Clone Repository

```bash id="jlwmdg"
git clone https://github.com/keshavsinghal2005/VehicleDetectionUsingYOLOv8.git
```

Move into directory:

```bash id="jlwmz1"
cd VehicleDetectionUsingYOLOv8
```

Install dependencies:

```bash id="jlwml6"
pip install -r requirements.txt
```

Run Streamlit application:

```bash id="jlwmx5"
streamlit run app.py
```



#  Applications

* Intelligent Traffic Monitoring
* Smart City Surveillance
* Vehicle Analytics
* Transportation Management
* Automated Traffic Inspection
* AI-Based Security Monitoring



#  Results

* Accurate vehicle localization and detection
* Real-time object inference using YOLOv8
* Efficient processing of traffic video streams
* Interactive visualization through Streamlit deployment



#  Future Enhancements

* Multi-class vehicle classification
* Vehicle counting and traffic density estimation
* Speed estimation and tracking
* License plate recognition integration
* Real-time cloud deployment optimization
* Advanced analytics dashboard



#  Repository Topics

```text id="jlwms0"
yolov8
computer-vision
deep-learning
object-detection
vehicle-detection
opencv
streamlit
traffic-monitoring
pytorch
ultralytics
artificial-intelligence
real-time-detection
```



#  Author

**Keshav Singhal**
AI/ML | Computer Vision & Deep Learning 

