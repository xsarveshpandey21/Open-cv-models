import os
import tempfile
import cv2
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from track_features import LipTracker, seed_lip_points, LK_PARAMS

# Page Configuration
st.set_page_config(
    page_title="Lip Tracking & Optical Flow Studio",
    page_icon="👄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    /* Dark glassmorphism theme styling */
    .main {
        background: linear-gradient(135deg, #0d1117 0%, #161b22 100%);
    }
    .stAppHeader {
        background: rgba(13, 17, 23, 0.8);
        backdrop-filter: blur(10px);
    }
    .metric-card {
        background: rgba(22, 27, 34, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        backdrop-filter: blur(4px);
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        background: linear-gradient(90deg, #58a6ff 0%, #bc8cff 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #8b949e;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .badge {
        background: rgba(88, 166, 255, 0.15);
        color: #58a6ff;
        border: 1px solid rgba(88, 166, 255, 0.3);
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.85rem;
        display: inline-block;
        margin-right: 8px;
    }
</style>
""", unsafe_allow_html=True)

# Header Section
st.title("👄 Lip Tracking & Optical Flow Studio")
st.markdown("""
<div style="margin-bottom: 25px;">
    <span class="badge">MTCNN Face Detection</span>
    <span class="badge">Shi-Tomasi Feature Seeding</span>
    <span class="badge">Lucas-Kanade Optical Flow</span>
    <span class="badge">5-Second Video Chunking</span>
</div>
""", unsafe_allow_html=True)

# Sidebar Navigation & Settings
st.sidebar.image("https://img.icons8.com/isometric/96/000000/artificial-intelligence.png", width=70)
st.sidebar.title("Navigation & Config")
app_mode = st.sidebar.radio("Select Mode", ["📊 Analytics & Sample Showcase", "🎥 Live Video Processor", "📁 CSV Metrics Explorer"])

# Helper: Load Log Data
@st.cache_data
def load_csv_data(filepath):
    if os.path.exists(filepath):
        return pd.read_csv(filepath)
    return None

if app_mode == "📊 Analytics & Sample Showcase":
    st.subheader("Interactive Tracking Demo & Real-Time Metrics")
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.markdown("### 🎬 Processed Video Output")
        sample_video_path = "tracked_lips.mp4"
        if os.path.exists(sample_video_path):
            st.video(sample_video_path)
            with open(sample_video_path, "rb") as vf:
                st.download_button(
                    label="⬇️ Download Annotated Video",
                    data=vf,
                    file_name="tracked_lips.mp4",
                    mime="video/mp4"
                )
        else:
            st.warning("Sample video `tracked_lips.mp4` not found in root directory.")

    with col2:
        st.markdown("### 📈 Pipeline Overview")
        st.info("""
        **Pipeline Mechanics:**
        1. **MTCNN Landmarks:** Extracts left and right mouth corner coordinates.
        2. **Feature Seeding:** Seeds up to 25 Shi-Tomasi corner points inside the lip ROI.
        3. **LK Optical Flow:** Tracks feature point movement frame-to-frame at high FPS.
        4. **Chunking & Logs:** Exports 5-second video chunks and logs frame metrics to CSV.
        """)

    # Metric Charts Section
    st.markdown("---")
    st.subheader("📊 Tracking Telemetry Charts")
    
    log_files = ["lips_tracking_log.csv", "features_log.csv", "track.csv", "recorded_log.csv"]
    existing_logs = [f for f in log_files if os.path.exists(f)]
    
    if existing_logs:
        selected_log = st.selectbox("Select Metrics Log File", existing_logs)
        df = load_csv_data(selected_log)
        
        if df is not None and not df.empty:
            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.metric("Total Frames Processed", len(df))
            with m2:
                if 'lip_width_px' in df.columns:
                    st.metric("Avg Lip Width (px)", f"{df['lip_width_px'].mean():.2f}")
                elif 'lip_width' in df.columns:
                    st.metric("Avg Lip Width (px)", f"{df['lip_width'].mean():.2f}")
                else:
                    st.metric("Avg Lip Width", "N/A")
            with m3:
                if 'active_points' in df.columns:
                    st.metric("Avg Active Points", f"{df['active_points'].mean():.1f}")
                elif 'num_features' in df.columns:
                    st.metric("Avg Active Points", f"{df['num_features'].mean():.1f}")
                else:
                    st.metric("Active Points", "N/A")
            with m4:
                if 'chunk_id' in df.columns:
                    st.metric("Total Chunks", df['chunk_id'].nunique())
                else:
                    st.metric("Total Chunks", "1")

            c1, c2 = st.columns(2)
            with c1:
                # Lip Width Over Time
                width_col = 'lip_width_px' if 'lip_width_px' in df.columns else ('lip_width' if 'lip_width' in df.columns else None)
                if width_col:
                    fig_w = px.line(
                        df, x='frame' if 'frame' in df.columns else df.index, y=width_col,
                        title="Lip Width Variation Over Time (Pixels)",
                        labels={'x': 'Frame Number', width_col: 'Width (px)'},
                        color_discrete_sequence=['#58a6ff']
                    )
                    fig_w.update_layout(template="plotly_dark", margin=dict(l=20, r=20, t=40, b=20))
                    st.plotly_chart(fig_w, use_container_width=True)
            
            with c2:
                # Lip Center Trajectory
                cx_col = 'lip_center_x' if 'lip_center_x' in df.columns else ('center_x' if 'center_x' in df.columns else None)
                cy_col = 'lip_center_y' if 'lip_center_y' in df.columns else ('center_y' if 'center_y' in df.columns else None)
                if cx_col and cy_col:
                    fig_center = px.scatter(
                        df, x=cx_col, y=cy_col, color='frame' if 'frame' in df.columns else df.index,
                        title="Lip Center Spatial Trajectory (X vs Y)",
                        labels={cx_col: 'Center X', cy_col: 'Center Y'},
                        color_continuous_scale='Viridis'
                    )
                    fig_center.update_yaxes(autorange="reversed")  # Image Y starts at top
                    fig_center.update_layout(template="plotly_dark", margin=dict(l=20, r=20, t=40, b=20))
                    st.plotly_chart(fig_center, use_container_width=True)

elif app_mode == "🎥 Live Video Processor":
    st.subheader("Upload Video & Run Lip Optical Flow Tracking")
    uploaded_file = st.file_uploader("Upload an MP4, AVI, or MOV video file", type=["mp4", "avi", "mov"])
    
    redetect_freq = st.slider("MTCNN Re-detection Interval (frames)", min_value=5, max_value=30, value=10)
    chunk_dur = st.number_input("Chunk Duration (seconds)", min_value=1.0, max_value=20.0, value=5.0, step=1.0)
    
    if uploaded_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded_file.read())
        tfile.close()

        if st.button("🚀 Process Video Now", type="primary"):
            st.info("Processing video stream with MTCNN & Lucas-Kanade Optical Flow...")
            cap = cv2.VideoCapture(tfile.name)
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            out_temp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out_writer = cv2.VideoWriter(out_temp.name, fourcc, fps, (width, height))

            tracker = LipTracker()
            prev_gray = None
            lip_pts = None
            lip_data = None
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
                    lip_data = tracker.detect_lips_mtcnn(frame)
                    if lip_data is not None:
                        seeded = seed_lip_points(gray, lip_data["box"])
                        if seeded is not None:
                            lip_pts = seeded
                elif prev_gray is not None and lip_pts is not None and len(lip_pts) > 0:
                    p1, st_pts, err = cv2.calcOpticalFlowPyrLK(prev_gray, gray, lip_pts, None, **LK_PARAMS)
                    if p1 is not None and st_pts is not None:
                        good_new = p1[st_pts == 1]
                        if len(good_new) > 0:
                            lip_pts = good_new.reshape(-1, 1, 2)
                        else:
                            lip_pts = None

                # Render annotations
                annotated = frame.copy()
                if lip_data is not None:
                    bx, by, bw, bh = lip_data["box"]
                    cv2.rectangle(annotated, (bx, by), (bx + bw, by + bh), (0, 255, 0), 2)
                    lx, ly = map(int, lip_data["left_corner"])
                    rx, ry = map(int, lip_data["right_corner"])
                    cv2.circle(annotated, (lx, ly), 4, (0, 0, 255), -1)
                    cv2.circle(annotated, (rx, ry), 4, (0, 0, 255), -1)

                if lip_pts is not None:
                    active_pts_count = len(lip_pts)
                    for pt in lip_pts:
                        px, py = map(int, pt.ravel())
                        cv2.circle(annotated, (px, py), 3, (255, 255, 0), -1)

                out_writer.write(annotated)
                
                if lip_data is not None:
                    processed_data.append({
                        "frame": frame_idx,
                        "time_s": frame_idx / fps,
                        "left_lip_x": lip_data["left_corner"][0],
                        "left_lip_y": lip_data["left_corner"][1],
                        "right_lip_x": lip_data["right_corner"][0],
                        "right_lip_y": lip_data["right_corner"][1],
                        "lip_center_x": lip_data["center"][0],
                        "lip_center_y": lip_data["center"][1],
                        "lip_width_px": lip_data["width"],
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

            if processed_data:
                res_df = pd.DataFrame(processed_data)
                st.dataframe(res_df.head(20))
                st.download_button(
                    label="⬇️ Download Tracking Metrics CSV",
                    data=res_df.to_csv(index=False),
                    file_name="processed_lips_metrics.csv",
                    mime="text/csv"
                )

elif app_mode == "📁 CSV Metrics Explorer":
    st.subheader("Raw Metrics Data Table & CSV Exporter")
    log_files = ["lips_tracking_log.csv", "features_log.csv", "track.csv", "recorded_log.csv"]
    existing_logs = [f for f in log_files if os.path.exists(f)]
    
    if existing_logs:
        selected_log = st.selectbox("Choose CSV File to View", existing_logs)
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
        st.info("No CSV logs found in workspace root.")
