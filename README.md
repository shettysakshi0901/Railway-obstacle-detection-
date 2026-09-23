# Railway Obstacle Detection

## 📌 Project Overview

Railway tracks need continuous monitoring to identify obstacles that may create unsafe conditions for trains. This project presents a computer vision-based Railway Obstacle Detection System that uses a YOLOv8 object detection model to identify different types of obstacles on railway tracks.

The system can detect the following six categories:

- 🐄 Animal
- 🚗 Car
- 👤 Person
- 🪨 Rock
- 🗑️ Trash
- 🌳 Tree

The detected objects are displayed with bounding boxes and confidence scores. The system can also generate an obstacle warning when an object is detected.

---

## 🎯 Objectives

The main objectives of this project are:

1. To detect obstacles present on railway tracks using computer vision.
2. To train a YOLOv8 object detection model using a railway obstacle dataset.
3. To classify detected obstacles into different categories.
4. To display the location of detected obstacles using bounding boxes.
5. To provide a warning when an obstacle is detected.
6. To demonstrate the system using images, videos, or a live camera.

---

## 🧠 Technology Used

| Technology | Purpose |
|---|---|
| Python | Programming language |
| YOLOv8 | Object detection |
| OpenCV | Image and video processing |
| Ultralytics | YOLO implementation |
| Roboflow | Dataset preparation |
| Computer Vision | Obstacle detection |
| Machine Learning | Model training |

---

## 📊 Dataset

The project uses the **Railway Track Obstacle Detection** dataset.

The dataset contains:

- **619 images**
- **6 object classes**
- YOLOv8 annotation format
- Training, validation and testing datasets

### Classes

```text
0 - Animal
1 - Car
2 - Person
3 - Rock
4 - Trash
5 - Tree
