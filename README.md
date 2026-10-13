# 👤 Whole Face, Ears, Eyes, Nose & Lips Live Tracking Studio

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8%2B-green.svg)](https://opencv.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B.svg)](https://streamlit.io/)
[![GitHub Pages](https://img.shields.io/badge/GitHub%20Pages-Live%20Showcase-brightgreen.svg)](#-github-pages-setup)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An end-to-end Computer Vision system for **Live Real-Time Webcam Tracking and Video Analysis** of **Whole Face, Left & Right Ears, Left & Right Eyes, Nose, and Lips**, featuring:
- **Real-Time Live Webcam Support**: Direct OpenCV camera capture (index 0) with live preview window and browser streaming.
- **Whole Face Tracking**: Primary bounding box `(x, y, w, h)` with center coordinates and exponential smoothing.
- **Left & Right Ear Tracking**: Cranial lateral geometry calculation, bounding boxes, and real-time `ear_span` measurement (px).
- **Dual Eye Tracking & Inter-Eye Distance**: Left Eye and Right Eye pupil centers, ROI boxes, and dynamic inter-eye distance metric.
- **Nose Landmark Anchor**: Central landmark tip and ROI box.
- **Lips Optical Flow**: Shi-Tomasi feature seeding inside lip ROI, Lucas-Kanade optical flow, motion trails, and parabolic curvature fit.
- **Cyber HUD Telemetry**: Translucent on-screen coordinate badges attached directly to landmarks plus a live cyber HUD telemetry box.
- **5-Second Video Chunking (`shutil`)**: Automatically partitions video chunks and cropped lip chunks into zip archives.
- **Comprehensive CSV Telemetry**: Logs frame indices, timestamps, and full spatial coordinates for every tracked landmark.
- **Interactive Streamlit Web Studio (`app.py`)**: Real-time browser app with live webcam streaming, photo snapshots, video uploader, and Plotly analytics.
- **Showcase Landing Page (`index.html`)**: Responsive dark glassmorphic showcase with interactive Chart.js charts.

---

## 🏗 Pipeline Architecture

```mermaid
flowchart TD
    A[Live Webcam (Camera 0) / Video File] --> B[MTCNN Face & Landmark Detection]
    B --> C[Compute Face BBox & Center Coordinates]
    B --> D[Left & Right Eyes Detection + Inter-Eye Distance]
    B --> E[Cranial Ear Geometry: Left & Right Ears + Ear Span]
    B --> F[Nose Landmark Keypoint]
    B --> G[Lip ROI, Corners, Shi-Tomasi Seeding & Parabola Fit]
    G --> H[Lucas-Kanade Optical Flow Motion Propagation]
    H -->|Every N Frames| B
    H --> I[Cyber HUD Telemetry & Coordinate Badges Overlay]
    I --> J[Save Full Video: recorded_face_lips_eyes_tracked.mp4]
    I --> K[Segment 5s Chunks & Zip Archive using shutil]
    I --> L[Export Comprehensive CSV Telemetry Log]
```

---

## 🚀 Quick Start Guide

### 1. Prerequisites & Installation

Clone the repository and install the dependencies:

```bash
git clone https://github.com/username/repository-name.git
cd "Open cv models"

pip install -r requirements.txt
```

### 2. Run Live Webcam Tracking (Camera 0)

Launch direct real-time webcam tracking from your terminal:

```bash
# Direct live webcam mode
python track_features.py --live

# Or specify webcam index 0
python track_features.py --input 0
```

#### Interactive Window Hotkeys:
| Key | Action |
| --- | --- |
| `ESC` / `q` | Stop session and finalize video/archive/CSV outputs |
| `h` | Toggle Cyber HUD telemetry overlay on/off |
| `b` | Toggle bounding boxes and coordinate badges on/off |
| `r` | Force instant MTCNN landmark re-detection |
| `s` | Save high-resolution snapshot image |

### 3. Run with Recorded Video File

Run detection & tracking on an existing video file:

```bash
# Process recorded video with live preview window
python track_features.py --input tracked_lips.mp4 --show --chunk-sec 5.0

# Process video in headless mode
python track_features.py --input tracked_lips.mp4 --no-show
```

### 4. Launch the Streamlit Web Application

Launch the interactive Streamlit Web Studio locally:

```bash
streamlit run app.py
```

Access the app in your browser at `http://localhost:8501`.

Modes available in the Streamlit app:
1. **🔴 Live Webcam Studio**: Real-time OpenCV video stream right in the browser + Camera photo snapshot analyzer.
2. **🎥 Recorded Video Processor**: Upload any video to track all landmarks and download annotated video & CSV.
3. **📊 Analytics & Showcase**: Plotly telemetry charts for Ear Span, Eye Distance, Lip Width, and spatial landmark trajectories.
4. **📁 CSV Metrics Explorer**: Inspect and filter recorded CSV logs.

---

## 📊 CSV Telemetry Columns

The exported `recorded_log.csv` contains complete spatial telemetry:

| Column Group | Metrics Included |
| --- | --- |
| **Time & Index** | `frame`, `chunk_id`, `time_s` |
| **Face** | `face_box_x`, `face_box_y`, `face_box_w`, `face_box_h`, `face_center_x`, `face_center_y` |
| **Ears** | `left_ear_x`, `left_ear_y`, `left_ear_w`, `left_ear_h`, `right_ear_x`, `right_ear_y`, `right_ear_w`, `right_ear_h`, `ear_span_px` |
| **Eyes** | `left_eye_x`, `left_eye_y`, `left_eye_w`, `left_eye_h`, `right_eye_x`, `right_eye_y`, `right_eye_w`, `right_eye_h`, `inter_eye_dist_px` |
| **Nose** | `nose_x`, `nose_y`, `nose_w`, `nose_h` |
| **Lips** | `left_lip_x`, `left_lip_y`, `right_lip_x`, `right_lip_y`, `lip_center_x`, `lip_center_y`, `lip_box_x`, `lip_box_y`, `lip_box_w`, `lip_box_h`, `lip_width_px`, `active_lip_points` |
| **Parabola Fit** | `parabola_a`, `parabola_b`, `parabola_c`, `parabola_apex_x`, `parabola_apex_y` |

---

## 🌐 Live Web Deployment Options

### Option 1: GitHub Pages Live Showcase (Static Showcase)

1. Push your project repository to GitHub:
   ```bash
   git remote add origin https://github.com/username/repository-name.git
   git branch -M main
   git push -u origin main
   ```
2. On GitHub, navigate to **Settings** > **Pages**.
3. Under **Build and deployment** > **Branch**, select `main` and `/ (root)`.
4. Click **Save**.

### Option 2: Streamlit Community Cloud

1. Push your repository to GitHub.
2. Sign in to [Streamlit Community Cloud](https://streamlit.io/cloud).
3. Connect your repository and set `app.py` as the entry point.
