import os
import sys
import tempfile
import csv
import time
import cv2
import numpy as np
import streamlit as st

try:
    import pandas as pd
    HAS_PANDAS = True
except Exception:
    HAS_PANDAS = False

try:
    import plotly.express as px
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except Exception:
    HAS_PLOTLY = False

from track_features import (
    FaceLipsEyesTracker, LipTracker, seed_lip_points,
    draw_coordinate_badge, draw_hud_panel, draw_all_feature_overlays,
    LK_PARAMS, fit_parabola_to_lips, draw_parabolic_lip_symbol,
    get_aligned_lip_box
)

# Page Configuration
st.set_page_config(
    page_title="Facial Feature & Live Tracking Studio",
    page_icon="👤",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling (Dark Glassmorphic UI)
st.markdown("""
<style>
    .main {
        background: linear-gradient(135deg, #0a0d14 0%, #121824 100%);
    }
    .stAppHeader {
        background: rgba(10, 13, 20, 0.85);
        backdrop-filter: blur(10px);
    }
    .metric-card {
        background: rgba(18, 24, 36, 0.75);
        border: 1px solid rgba(0, 229, 255, 0.2);
        border-radius: 12px;
        padding: 18px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.45);
        backdrop-filter: blur(6px);
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        background: linear-gradient(90deg, #00e5ff 0%, #00ff88 50%, #e040fb 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .metric-label {
        font-size: 0.82rem;
        color: #8b9bb4;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .badge {
        background: rgba(0, 229, 255, 0.12);
        color: #00e5ff;
        border: 1px solid rgba(0, 229, 255, 0.3);
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.82rem;
        display: inline-block;
        margin-right: 6px;
        margin-bottom: 6px;
        font-weight: 500;
    }
    .badge-live {
        background: rgba(255, 60, 60, 0.15);
        color: #ff4d4d;
        border: 1px solid rgba(255, 60, 60, 0.4);
    }
    .badge-amber {
        background: rgba(255, 180, 0, 0.15);
        color: #ffb400;
        border: 1px solid rgba(255, 180, 0, 0.4);
    }
</style>
""", unsafe_allow_html=True)

# Header Section
st.title("👤 Whole Face, Ears, Eyes, Nose & Lips Tracking Studio")
st.markdown("""
<div style="margin-bottom: 22px;">
    <span class="badge badge-live">🔴 Live Webcam Tracking</span>
    <span class="badge">👤 Face BBox & Center</span>
    <span class="badge badge-amber">👂 Left & Right Ears</span>
    <span class="badge">👁️ Eyes & Distance</span>
    <span class="badge">👃 Nose Landmark</span>
    <span class="badge">👄 Lips Optical Flow</span>
    <span class="badge">📍 Cyber HUD Telemetry</span>
    <span class="badge">📁 Full CSV Exporter</span>
</div>
""", unsafe_allow_html=True)

# Sidebar Navigation & Settings
st.sidebar.image("https://img.icons8.com/isometric/96/000000/artificial-intelligence.png", width=65)
st.sidebar.title("Studio Controls")
app_mode = st.sidebar.radio(
    "Select Operating Mode",
    [
        "🔴 Live Webcam Studio",
        "🎥 Recorded Video Processor",
        "📊 Analytics & Showcase",
        "📁 CSV Metrics Explorer"
    ]
)

# Helper: Load Log Data
def load_csv_data(filepath):
    if not os.path.exists(filepath):
        return None
    if HAS_PANDAS:
        return pd.read_csv(filepath)
    else:
        rows = []
        with open(filepath, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                rows.append(r)
        return rows


# =====================================================================
# 1. LIVE WEBCAM STUDIO MODE
# =====================================================================
if app_mode == "🔴 Live Webcam Studio":
    st.subheader("🔴 Real-Time Live Webcam Facial Tracker")
    st.markdown("Track **Whole Face, Left/Right Ears, Eyes, Nose, and Lips** live from your system camera with instant telemetry coordinates.")

    cam_method = st.radio(
        "Select Camera Interface",
        ["⚡ Live OpenCV Camera Stream", "📸 In-Browser Camera Capture (Snapshot & Photo)"],
        horizontal=True
    )

    if cam_method == "⚡ Live OpenCV Camera Stream":
        st.markdown("#### Live Desktop Camera Stream")
        col_ctrl, col_view = st.columns([1, 2])

        with col_ctrl:
            st.markdown("##### Stream Settings")
            cam_idx = st.number_input("Camera Index", min_value=0, max_value=5, value=0, step=1)
            redetect_frames = st.slider("MTCNN Redetection Rate", min_value=3, max_value=25, value=8)
            show_hud_toggle = st.checkbox("Show Cyber HUD Telemetry", value=True)
            show_boxes_toggle = st.checkbox("Show Bounding Boxes & Badges", value=True)
            run_live = st.toggle("🔴 Start Live Camera Stream", value=False)

            st.info("""
            **Tracked Features:**
            - **Face**: Cyan Box & Center Crosshair
            - **Ears**: Left & Right Ear Targets + Ear Span
            - **Eyes**: Pupils, ROI Boxes & Distance
            - **Nose**: Central Keypoint
            - **Lips**: Corners, Optical Flow & Parabola
            """)

            st.markdown("---")
            st.markdown("**Native Window Mode:**")
            st.caption("You can also run the ultra-fast native OpenCV window from terminal:")
            st.code("python track_features.py --live", language="bash")

        with col_view:
            frame_placeholder = st.empty()
            warning_placeholder = st.empty()
            metric_cols = st.columns(4)
            m_fps = metric_cols[0].empty()
            m_eyes = metric_cols[1].empty()
            m_ears = metric_cols[2].empty()
            m_lips = metric_cols[3].empty()

            if run_live:
                cap = None
                if sys.platform.startswith('win'):
                    cap = cv2.VideoCapture(int(cam_idx), cv2.CAP_DSHOW)
                if cap is None or not cap.isOpened():
                    cap = cv2.VideoCapture(int(cam_idx))

                if not cap.isOpened():
                    st.error(f"Error: Could not open Camera index {cam_idx}. Make sure your camera is connected and not in use by another app.")
                else:
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                    cap.set(cv2.CAP_PROP_FPS, 30)

                    # Sensor warmup: read and discard first 5 frames for auto-exposure stabilization
                    for _ in range(5):
                        cap.read()

                    tracker = FaceLipsEyesTracker()
                    prev_gray = None
                    lip_pts = None
                    features_data = None
                    frame_num = 0
                    consecutive_fails = 0
                    t_last = time.time()
                    fps_val = 30.0

                    try:
                        while run_live:
                            ret, frame = cap.read()
                            if not ret or frame is None:
                                consecutive_fails += 1
                                if consecutive_fails > 15:
                                    st.warning("Camera stream interrupted. Please verify your camera is connected.")
                                    break
                                time.sleep(0.04)
                                continue
                            consecutive_fails = 0

                            # Check for covered camera / dark frames
                            if frame.mean() < 12.0:
                                warning_placeholder.warning("⚠️ Camera frame is pitch black. If your laptop/webcam has a physical privacy slider or shutter, please open it!")
                            else:
                                warning_placeholder.empty()

                            t_now = time.time()
                            dt = t_now - t_last
                            t_last = t_now
                            if dt > 0:
                                fps_val = 0.85 * fps_val + 0.15 * (1.0 / dt)

                            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                            # Optical flow update
                            if prev_gray is not None and lip_pts is not None and len(lip_pts) >= 3 and features_data is not None:
                                new_pts, st_pts, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, lip_pts, None, **LK_PARAMS)
                                if new_pts is not None and st_pts is not None:
                                    good_new = new_pts[st_pts == 1]
                                    good_old = lip_pts[st_pts == 1]
                                    if len(good_new) >= 3:
                                        dx, dy = np.median(good_new - good_old, axis=0)
                                        # Shift landmarks
                                        fbx, fby, fbw, fbh = features_data["face_box"]
                                        fcx, fcy = features_data["face_center"]
                                        lear_x, lear_y = features_data["left_ear"]
                                        rear_x, rear_y = features_data["right_ear"]
                                        lex, ley = features_data["left_eye"]
                                        rex, rey = features_data["right_eye"]
                                        nx, ny = features_data["nose"]
                                        llx, lly = features_data["left_corner"]
                                        rlx, rly = features_data["right_corner"]
                                        lcx, lcy = features_data["center"]
                                        lbx, lby, lbw, lbh = features_data["box"]

                                        features_data["face_box"] = (int(fbx + dx), int(fby + dy), fbw, fbh)
                                        features_data["face_center"] = (fcx + dx, fcy + dy)
                                        features_data["left_ear"] = (lear_x + dx, lear_y + dy)
                                        features_data["right_ear"] = (rear_x + dx, rear_y + dy)
                                        features_data["left_eye"] = (lex + dx, ley + dy)
                                        features_data["right_eye"] = (rex + dx, rey + dy)
                                        features_data["nose"] = (nx + dx, ny + dy)
                                        features_data["left_corner"] = (llx + dx, lly + dy)
                                        features_data["right_corner"] = (rlx + dx, rly + dy)
                                        features_data["center"] = (lcx + dx, lcy + dy)
                                        features_data["box"] = (int(lbx + dx), int(lby + dy), lbw, lbh)
                                        lip_pts = good_new.reshape(-1, 1, 2)
                                    else:
                                        lip_pts = None
                                else:
                                    lip_pts = None

                            # Re-detection
                            if frame_num % redetect_frames == 0 or features_data is None or lip_pts is None or len(lip_pts) < 5:
                                det = tracker.detect_features_mtcnn(frame)
                                if det is not None:
                                    features_data = det
                                    lip_pts = seed_lip_points(gray, features_data["box"])

                            parabola_info = None
                            if features_data is not None:
                                parabola_info = fit_parabola_to_lips(
                                    features_data["left_corner"],
                                    features_data["center"],
                                    features_data["right_corner"],
                                    lip_pts=lip_pts
                                )

                            annotated = frame.copy()
                            draw_all_feature_overlays(
                                annotated,
                                features_data,
                                lip_pts=lip_pts,
                                parabola_info=parabola_info,
                                show_badges=show_boxes_toggle,
                                show_boxes=show_boxes_toggle
                            )

                            if show_hud_toggle:
                                draw_hud_panel(
                                    annotated,
                                    frame_num,
                                    fps_val,
                                    0,
                                    features_data,
                                    parabola_info=parabola_info,
                                    is_live=True
                                )

                            # Convert to RGB for Streamlit rendering
                            frame_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                            frame_placeholder.image(frame_rgb, use_container_width=True)

                            # Update metrics
                            m_fps.metric("Live FPS", f"{fps_val:.1f}")
                            if features_data is not None:
                                m_eyes.metric("Eye Distance", f"{features_data['eye_distance']:.1f} px")
                                m_ears.metric("Ear Span", f"{features_data.get('ear_span', 0.0):.1f} px")
                                m_lips.metric("Lip Width", f"{features_data['width']:.1f} px")
                            else:
                                m_eyes.metric("Eye Distance", "Searching...")
                                m_ears.metric("Ear Span", "Searching...")
                                m_lips.metric("Lip Width", "Searching...")

                            prev_gray = gray
                            frame_num += 1
                            time.sleep(0.01)

                    finally:
                        cap.release()
            else:
                frame_placeholder.markdown("""
                <div style="border: 2px dashed #30363d; border-radius: 12px; padding: 60px; text-align: center;">
                    <h3 style="color: #8b949e;">Camera Offline</h3>
                    <p style="color: #6e7681;">Toggle <b>'🔴 Start Live Camera Stream'</b> on the left to begin real-time facial landmark tracking.</p>
                </div>
                """, unsafe_allow_html=True)

    else:
        st.markdown("#### High-Resolution Camera Snapshot Tracker")
        img_file_buffer = st.camera_input("Take a photo to extract all facial coordinates")

        if img_file_buffer is not None:
            bytes_data = img_file_buffer.getvalue()
            cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

            tracker = FaceLipsEyesTracker()
            gray = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2GRAY)
            features_data = tracker.detect_features_mtcnn(cv2_img)

            if features_data is not None:
                lip_pts = seed_lip_points(gray, features_data["box"])
                parabola_info = fit_parabola_to_lips(
                    features_data["left_corner"],
                    features_data["center"],
                    features_data["right_corner"],
                    lip_pts=lip_pts
                )

                annotated_snap = cv2_img.copy()
                draw_all_feature_overlays(
                    annotated_snap,
                    features_data,
                    lip_pts=lip_pts,
                    parabola_info=parabola_info,
                    show_badges=True,
                    show_boxes=True
                )
                draw_hud_panel(annotated_snap, 1, 30.0, 0, features_data, parabola_info=parabola_info, is_live=True)

                c_img, c_table = st.columns([3, 2])
                with c_img:
                    st.image(cv2.cvtColor(annotated_snap, cv2.COLOR_BGR2RGB), caption="Analyzed Landmark Frame", use_container_width=True)

                with c_table:
                    st.markdown("### 📋 Coordinates Extracted")
                    coords_summary = {
                        "Feature": ["Face Box", "Face Center", "Left Ear", "Right Ear", "Ear Span", "Left Eye", "Right Eye", "Eye Distance", "Nose", "Lips Center", "Lip Width"],
                        "Coordinate / Metric": [
                            str(features_data["face_box"]),
                            f"({features_data['face_center'][0]:.1f}, {features_data['face_center'][1]:.1f})",
                            f"({features_data['left_ear'][0]:.1f}, {features_data['left_ear'][1]:.1f})",
                            f"({features_data['right_ear'][0]:.1f}, {features_data['right_ear'][1]:.1f})",
                            f"{features_data.get('ear_span', 0.0):.1f} px",
                            f"({features_data['left_eye'][0]:.1f}, {features_data['left_eye'][1]:.1f})",
                            f"({features_data['right_eye'][0]:.1f}, {features_data['right_eye'][1]:.1f})",
                            f"{features_data['eye_distance']:.1f} px",
                            f"({features_data['nose'][0]:.1f}, {features_data['nose'][1]:.1f})",
                            f"({features_data['center'][0]:.1f}, {features_data['center'][1]:.1f})",
                            f"{features_data['width']:.1f} px"
                        ]
                    }
                    if HAS_PANDAS:
                        st.dataframe(pd.DataFrame(coords_summary), use_container_width=True)
                    else:
                        st.table(coords_summary)
            else:
                st.warning("No face detected in the photo. Please ensure good lighting and face camera directly.")


# =====================================================================
# 2. RECORDED VIDEO PROCESSOR
# =====================================================================
elif app_mode == "🎥 Recorded Video Processor":
    st.subheader("Process Video File with Full Face, Ears, Eyes, Nose & Lips Tracking")
    uploaded_file = st.file_uploader("Upload an MP4, AVI, or MOV video file", type=["mp4", "avi", "mov"])

    c_s1, c_s2 = st.columns(2)
    with c_s1:
        redetect_freq = st.slider("MTCNN Re-detection Interval (frames)", min_value=5, max_value=30, value=10)
    with c_s2:
        chunk_dur = st.number_input("Chunk Duration (seconds)", min_value=1.0, max_value=20.0, value=5.0, step=1.0)

    if uploaded_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded_file.read())
        tfile.close()

        if st.button("🚀 Process Video Now", type="primary"):
            st.info("Processing video with Whole Face, Left/Right Ears, Eyes, Nose & Lips Tracking...")
            cap = cv2.VideoCapture(tfile.name)
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            out_temp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out_writer = cv2.VideoWriter(out_temp.name, fourcc, fps, (width, height))

            tracker = FaceLipsEyesTracker()
            prev_gray = None
            lip_pts = None
            features_data = None
            processed_data = []

            progress_bar = st.progress(0)
            frame_idx = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                active_pts_count = 0

                if frame_idx % redetect_freq == 0 or lip_pts is None or len(lip_pts) < 5:
                    features_data = tracker.detect_features_mtcnn(frame)
                    if features_data is not None:
                        seeded = seed_lip_points(gray, features_data["box"])
                        if seeded is not None:
                            lip_pts = seeded
                elif prev_gray is not None and lip_pts is not None and len(lip_pts) > 0 and features_data is not None:
                    p1, st_pts, err = cv2.calcOpticalFlowPyrLK(prev_gray, gray, lip_pts, None, **LK_PARAMS)
                    if p1 is not None and st_pts is not None:
                        good_new = p1[st_pts == 1]
                        good_old = lip_pts[st_pts == 1]
                        if len(good_new) >= 3:
                            dx, dy = np.median(good_new - good_old, axis=0)
                            features_data["face_box"] = (int(features_data["face_box"][0] + dx), int(features_data["face_box"][1] + dy), features_data["face_box"][2], features_data["face_box"][3])
                            features_data["left_ear"] = (features_data["left_ear"][0] + dx, features_data["left_ear"][1] + dy)
                            features_data["right_ear"] = (features_data["right_ear"][0] + dx, features_data["right_ear"][1] + dy)
                            features_data["left_eye"] = (features_data["left_eye"][0] + dx, features_data["left_eye"][1] + dy)
                            features_data["right_eye"] = (features_data["right_eye"][0] + dx, features_data["right_eye"][1] + dy)
                            features_data["nose"] = (features_data["nose"][0] + dx, features_data["nose"][1] + dy)
                            features_data["left_corner"] = (features_data["left_corner"][0] + dx, features_data["left_corner"][1] + dy)
                            features_data["right_corner"] = (features_data["right_corner"][0] + dx, features_data["right_corner"][1] + dy)
                            features_data["center"] = (features_data["center"][0] + dx, features_data["center"][1] + dy)
                            lip_pts = good_new.reshape(-1, 1, 2)
                        else:
                            lip_pts = None

                # Render annotations
                annotated = frame.copy()
                parabola_info = None

                if features_data is not None:
                    parabola_info = fit_parabola_to_lips(
                        features_data["left_corner"],
                        features_data["center"],
                        features_data["right_corner"],
                        lip_pts=lip_pts
                    )
                    draw_all_feature_overlays(annotated, features_data, lip_pts=lip_pts, parabola_info=parabola_info)
                    draw_hud_panel(annotated, frame_idx, fps, 0, features_data, parabola_info=parabola_info, is_live=False)

                active_pts_count = len(lip_pts) if lip_pts is not None else 0
                out_writer.write(annotated)

                if features_data is not None:
                    processed_data.append({
                        "frame": frame_idx,
                        "time_s": frame_idx / fps,
                        "face_box_x": features_data["face_box"][0],
                        "face_box_y": features_data["face_box"][1],
                        "face_box_w": features_data["face_box"][2],
                        "face_box_h": features_data["face_box"][3],
                        "left_ear_x": features_data["left_ear"][0],
                        "left_ear_y": features_data["left_ear"][1],
                        "right_ear_x": features_data["right_ear"][0],
                        "right_ear_y": features_data["right_ear"][1],
                        "ear_span_px": features_data.get("ear_span", 0.0),
                        "left_eye_x": features_data["left_eye"][0],
                        "left_eye_y": features_data["left_eye"][1],
                        "right_eye_x": features_data["right_eye"][0],
                        "right_eye_y": features_data["right_eye"][1],
                        "inter_eye_dist_px": features_data["eye_distance"],
                        "nose_x": features_data["nose"][0],
                        "nose_y": features_data["nose"][1],
                        "left_lip_x": features_data["left_corner"][0],
                        "left_lip_y": features_data["left_corner"][1],
                        "right_lip_x": features_data["right_corner"][0],
                        "right_lip_y": features_data["right_corner"][1],
                        "lip_center_x": features_data["center"][0],
                        "lip_center_y": features_data["center"][1],
                        "lip_width_px": features_data["width"],
                        "active_points": active_pts_count
                    })

                prev_gray = gray.copy()
                frame_idx += 1
                if total_frames > 0:
                    progress_bar.progress(min(1.0, frame_idx / total_frames))

            cap.release()
            out_writer.release()

            st.success("✅ Video Processing Complete!")
            st.video(out_temp.name)

            if processed_data and HAS_PANDAS:
                res_df = pd.DataFrame(processed_data)
                st.dataframe(res_df.head(25))
                st.download_button(
                    label="⬇️ Download Tracking Metrics CSV",
                    data=res_df.to_csv(index=False),
                    file_name="processed_face_ears_eyes_lips_coordinates.csv",
                    mime="text/csv"
                )


# =====================================================================
# 3. ANALYTICS & SAMPLE SHOWCASE
# =====================================================================
elif app_mode == "📊 Analytics & Showcase":
    st.subheader("Facial Landmark Showcase & Telemetry Analytics")
    col1, col2 = st.columns([3, 2])

    with col1:
        st.markdown("### 🎬 Processed Video Output")
        video_options = []
        if os.path.exists("recorded_face_lips_eyes_tracked.mp4"):
            video_options.append("recorded_face_lips_eyes_tracked.mp4")
        if os.path.exists("tracked_lips.mp4"):
            video_options.append("tracked_lips.mp4")

        selected_video = st.selectbox("Select Video Output", video_options) if video_options else None

        if selected_video and os.path.exists(selected_video):
            st.video(selected_video)
            with open(selected_video, "rb") as vf:
                st.download_button(
                    label=f"⬇️ Download Video ({selected_video})",
                    data=vf,
                    file_name=selected_video,
                    mime="video/mp4"
                )
        else:
            st.warning("No sample video found in workspace root.")

    with col2:
        st.markdown("### 📈 Comprehensive Pipeline Mechanics")
        st.info("""
        **Integrated Tracking Architecture:**
        1. **Whole Face BBox:** Detects & tracks complete face bounding box `(x, y, w, h)`.
        2. **Left & Right Ears:** Cranial geometry calculation & lateral ear tragus tracking with `ear_span` (px).
        3. **Eyes Landmark Tracking:** Left Eye & Right Eye landmarks, bounding boxes, & inter-eye distance (`px`).
        4. **Nose Landmark:** Tip coordinate and spatial anchor.
        5. **Lips Optical Flow:** Mouth corners, lip center, Shi-Tomasi points & Lucas-Kanade optical flow.
        6. **Cyber HUD Telemetry:** Translucent real-time coordinate badges on live video.
        """)

    # Metric Charts Section
    st.markdown("---")
    st.subheader("📊 Tracking Telemetry Charts")

    log_files = ["recorded_log.csv", "lips_tracking_log.csv", "features_log.csv", "track.csv"]
    existing_logs = [f for f in log_files if os.path.exists(f)]

    if existing_logs:
        selected_log = st.selectbox("Select Metrics Log File", existing_logs)
        df = load_csv_data(selected_log)

        if df is not None:
            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.metric("Total Frames", len(df))
            with m2:
                st.metric("Tracking Target", "Face + Ears + Eyes + Lips")
            with m3:
                st.metric("Optical Flow", "Active")
            with m4:
                st.metric("Status", "Operational")

            if HAS_PANDAS and HAS_PLOTLY and isinstance(df, pd.DataFrame):
                c1, c2 = st.columns(2)
                with c1:
                    fig_w = go.Figure()
                    if 'lip_width_px' in df.columns:
                        fig_w.add_trace(go.Scatter(x=df['frame'], y=df['lip_width_px'], mode='lines', name='Lip Width (px)', line=dict(color='#ff4081')))
                    if 'inter_eye_dist_px' in df.columns:
                        fig_w.add_trace(go.Scatter(x=df['frame'], y=df['inter_eye_dist_px'], mode='lines', name='Inter-Eye Dist (px)', line=dict(color='#00e5ff')))
                    if 'ear_span_px' in df.columns:
                        fig_w.add_trace(go.Scatter(x=df['frame'], y=df['ear_span_px'], mode='lines', name='Ear Span (px)', line=dict(color='#ffb300')))
                    fig_w.update_layout(title="Distance & Feature Spans Over Time", template="plotly_dark", margin=dict(l=20, r=20, t=40, b=20))
                    st.plotly_chart(fig_w, use_container_width=True)

                with c2:
                    fig_center = go.Figure()
                    if 'left_ear_x' in df.columns and 'left_ear_y' in df.columns:
                        fig_center.add_trace(go.Scatter(x=df['left_ear_x'], y=df['left_ear_y'], mode='markers', name='Left Ear', marker=dict(color='#ffb300', size=5)))
                    if 'right_ear_x' in df.columns and 'right_ear_y' in df.columns:
                        fig_center.add_trace(go.Scatter(x=df['right_ear_x'], y=df['right_ear_y'], mode='markers', name='Right Ear', marker=dict(color='#ffb300', size=5)))
                    if 'left_eye_x' in df.columns and 'left_eye_y' in df.columns:
                        fig_center.add_trace(go.Scatter(x=df['left_eye_x'], y=df['left_eye_y'], mode='markers', name='Left Eye', marker=dict(color='#00e5ff', size=5)))
                    if 'right_eye_x' in df.columns and 'right_eye_y' in df.columns:
                        fig_center.add_trace(go.Scatter(x=df['right_eye_x'], y=df['right_eye_y'], mode='markers', name='Right Eye', marker=dict(color='#00e5ff', size=5)))
                    if 'nose_x' in df.columns and 'nose_y' in df.columns:
                        fig_center.add_trace(go.Scatter(x=df['nose_x'], y=df['nose_y'], mode='markers', name='Nose', marker=dict(color='#e040fb', size=6)))
                    if 'lip_center_x' in df.columns and 'lip_center_y' in df.columns:
                        fig_center.add_trace(go.Scatter(x=df['lip_center_x'], y=df['lip_center_y'], mode='markers', name='Lip Center', marker=dict(color='#ff4081', size=5)))
                    fig_center.update_yaxes(autorange="reversed")
                    fig_center.update_layout(title="Facial Landmark Spatial Coordinates (X vs Y)", template="plotly_dark", margin=dict(l=20, r=20, t=40, b=20))
                    st.plotly_chart(fig_center, use_container_width=True)


# =====================================================================
# 4. CSV METRICS EXPLORER
# =====================================================================
elif app_mode == "📁 CSV Metrics Explorer":
    st.subheader("Raw Metrics Data Table & CSV Exporter")
    log_files = ["recorded_log.csv", "lips_tracking_log.csv", "features_log.csv", "track.csv"]
    existing_logs = [f for f in log_files if os.path.exists(f)]

    if existing_logs:
        selected_log = st.selectbox("Choose CSV File to View", existing_logs)
        if HAS_PANDAS:
            df = pd.read_csv(selected_log)
            st.write(f"Showing **{len(df)}** rows from `{selected_log}`:")
            st.dataframe(df, use_container_width=True)

            st.download_button(
                label=f"⬇️ Download {selected_log}",
                data=df.to_csv(index=False),
                file_name=selected_log,
                mime="text/csv"
            )
        else:
            with open(selected_log, "r", encoding="utf-8") as f:
                content = f.read()
            st.code(content[:2500] + "\n... (truncated)")
            st.download_button(
                label=f"⬇️ Download {selected_log}",
                data=content,
                file_name=selected_log,
                mime="text/csv"
            )
    else:
        st.info("No CSV logs found in workspace root.")
