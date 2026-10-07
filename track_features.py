"""
MTCNN Lips-Only Detection & Optical Flow Tracking with 5-Second Video Chunking (shutil).

Pipeline:
  1. Input Selection: Select a recorded video file via native Tkinter file picker,
     command-line argument (--input), or webcam stream.
  2. Directory Setup (shutil): Manage output chunk directories using shutil.rmtree()
     and os.makedirs() to store 5-second video chunks and cropped lip video chunks.
  3. MTCNN Detection: Detect face landmarks using MTCNN (facenet_pytorch).
     Extract Left Mouth Corner and Right Mouth Corner landmarks.
  4. Lips Region ROI & Feature Seeding: Compute lip center, mouth width, and lip bounding box.
     Seed Shi-Tomasi corner points within the lip ROI for precise tracking.
  5. Hybrid Lucas-Kanade Optical Flow: Track lip feature points frame-to-frame between
     MTCNN re-detection intervals to maintain smooth, high-FPS lip tracking.
  6. Lips-Only Visualization: Render bounding box, lip corners, lip center, optical flow points,
     and lip motion path. (Strictly no face, eye, nose, or ear bounding boxes).
  7. Export & Archiving (shutil): Save full annotated video, 5-second video chunks,
     cropped 5-second lip video chunks, zip archive (shutil.make_archive), and CSV metrics log.

Install Dependencies:
  pip install opencv-python facenet-pytorch torch numpy
"""

import argparse
import csv
from collections import deque
import os
import sys
import shutil
import cv2
import numpy as np
from PIL import Image

# Try importing torch and facenet_pytorch MTCNN
try:
    import torch
    from facenet_pytorch import MTCNN
    HAS_MTCNN = True
except ImportError:
    HAS_MTCNN = False

# Try importing tkinter for native file selection dialog
try:
    import tkinter as tk
    from tkinter import filedialog
    HAS_TK = True
except ImportError:
    HAS_TK = False

# Fallback Haar Cascade for mouth/smile if MTCNN unavailable
SMILE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_smile.xml")
FACE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Lucas-Kanade optical flow parameters
LK_PARAMS = dict(
    winSize=(15, 15),
    maxLevel=2,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03)
)

# Shi-Tomasi corner detector parameters for lips region
CORNER_PARAMS = dict(
    maxCorners=25,
    qualityLevel=0.01,
    minDistance=4,
    blockSize=3
)


def select_video_file_gui():
    """Open a native Windows file selection dialog to choose a recorded video."""
    if not HAS_TK:
        return None
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    file_path = filedialog.askopenfilename(
        title="Select Video File for Lips Tracking",
        filetypes=[("Video Files", "*.mp4 *.avi *.mov *.mkv *.wmv"), ("All Files", "*.*")]
    )
    root.destroy()
    return file_path if file_path else None


class LipTracker:
    def __init__(self, min_face_size=30, thresholds=(0.5, 0.6, 0.6)):
        self.device = 'cuda' if HAS_MTCNN and torch.cuda.is_available() else 'cpu'
        if HAS_MTCNN:
            self.mtcnn = MTCNN(
                keep_all=True,
                min_face_size=min_face_size,
                thresholds=list(thresholds),
                device=self.device
            )
        else:
            self.mtcnn = None
            print("Warning: facenet_pytorch MTCNN not found. Falling back to Haar cascades.")

    def detect_lips_mtcnn(self, frame_bgr):
        """
        Detect mouth/lips using MTCNN landmarks.
        Returns:
            dict with 'left_corner', 'right_corner', 'center', 'width', 'box'
            or None if no face/mouth detected.
        """
        if self.mtcnn is None:
            return self.detect_lips_haar(frame_bgr)

        h_img, w_img = frame_bgr.shape[:2]
        img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)

        try:
            boxes, probs, landmarks = self.mtcnn.detect(pil_img, landmarks=True)
        except Exception:
            return self.detect_lips_haar(frame_bgr)

        if landmarks is None or len(landmarks) == 0:
            return self.detect_lips_haar(frame_bgr)

        # Select primary face (highest detection probability or largest box)
        best_idx = 0
        if boxes is not None and len(boxes) > 1:
            areas = [(b[2] - b[0]) * (b[3] - b[1]) for b in boxes]
            best_idx = int(np.argmax(areas))

        lm = landmarks[best_idx]  # 5 landmarks: [left_eye, right_eye, nose, left_mouth, right_mouth]
        left_m = lm[3]   # (x, y)
        right_m = lm[4]  # (x, y)

        center_x = (left_m[0] + right_m[0]) / 2.0
        center_y = (left_m[1] + right_m[1]) / 2.0
        lip_width = float(np.linalg.norm(right_m - left_m))

        if lip_width < 5:
            return None

        # Compute Lip ROI Box centered on lip center
        box_w = int(lip_width * 1.6)
        box_h = int(lip_width * 1.0)
        box_x = max(0, int(center_x - box_w / 2.0))
        box_y = max(0, int(center_y - box_h * 0.45))
        box_w = min(w_img - box_x, box_w)
        box_h = min(h_img - box_y, box_h)

        return {
            "left_corner": (float(left_m[0]), float(left_m[1])),
            "right_corner": (float(right_m[0]), float(right_m[1])),
            "center": (float(center_x), float(center_y)),
            "width": lip_width,
            "box": (box_x, box_y, box_w, box_h)
        }

    def detect_lips_haar(self, frame_bgr):
        """Fallback detection using OpenCV Haar Cascades."""
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        h_img, w_img = gray.shape[:2]
        faces = FACE_CASCADE.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))
        if len(faces) == 0:
            return None

        fx, fy, fw, fh = max(faces, key=lambda f: f[2] * f[3])
        roi_lower = gray[fy + int(fh * 0.6): fy + fh, fx: fx + fw]
        smiles = SMILE_CASCADE.detectMultiScale(roi_lower, 1.5, 10, minSize=(25, 15))

        if len(smiles):
            sx, sy, sw, sh = max(smiles, key=lambda b: b[2] * b[3])
            bx, by, bw, bh = fx + sx, fy + int(fh * 0.6) + sy, sw, sh
        else:
            bx = fx + int(fw * 0.22)
            by = fy + int(fh * 0.68)
            bw = int(fw * 0.56)
            bh = int(fh * 0.24)

        lx, ly = bx + int(bw * 0.15), by + int(bh * 0.5)
        rx, ry = bx + int(bw * 0.85), by + int(bh * 0.5)
        cx, cy = bx + int(bw * 0.5), by + int(bh * 0.5)
        w = float(bw * 0.7)

        return {
            "left_corner": (float(lx), float(ly)),
            "right_corner": (float(rx), float(ry)),
            "center": (float(cx), float(cy)),
            "width": w,
            "box": (bx, by, bw, bh)
        }


def seed_lip_points(gray, lip_box):
    """Seed Shi-Tomasi feature points within the lip ROI box."""
    x, y, w, h = lip_box
    h_img, w_img = gray.shape[:2]
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(w_img, x + w), min(h_img, y + h)

    if x2 <= x1 or y2 <= y1:
        return None

    mask = np.zeros_like(gray)
    mask[y1:y2, x1:x2] = 255
    pts = cv2.goodFeaturesToTrack(gray, mask=mask, **CORNER_PARAMS)
    return pts


def main():
    ap = argparse.ArgumentParser(description="Lips-Only Detection & Tracking using MTCNN with 5-Second Shutil Video Chunking")
    ap.add_argument("--input", default=None, help="Input video file path or webcam index (0). Default opens GUI picker.")
    ap.add_argument("--output", default="tracked_lips.mp4", help="Output annotated main video path.")
    ap.add_argument("--chunks-dir", default="lips_chunks_5s", help="Output directory for 5-second video chunks.")
    ap.add_argument("--crop-dir", default="lips_cropped_chunks_5s", help="Output directory for cropped 5s lip video chunks.")
    ap.add_argument("--csv", default="lips_tracking_log.csv", help="Output CSV metrics log path.")
    ap.add_argument("--chunk-sec", type=float, default=5.0, help="Chunk duration in seconds (default: 5.0s).")
    ap.add_argument("--redetect", type=int, default=10, help="Re-detect with MTCNN every N frames.")
    ap.add_argument("--show", action="store_true", help="Display live preview window during processing.")
    args = ap.parse_args()

    input_source = args.input
    if input_source is None:
        print("Opening video file selection dialog... Please select your video file.")
        input_source = select_video_file_gui()
        if not input_source:
            print("No video file selected. Exiting.")
            sys.exit(0)

    src = int(input_source) if isinstance(input_source, str) and input_source.isdigit() else input_source
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise SystemExit(f"Error: Cannot open video source '{input_source}'")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames_per_chunk = max(1, int(fps * args.chunk_sec))

    print(f"\n========================================================")
    print(f" MTCNN LIPS-ONLY TRACKER")
    print(f" Video Source    : {input_source}")
    print(f" Resolution      : {width}x{height} @ {fps:.2f} FPS")
    print(f" Chunk Duration  : {args.chunk_sec} sec ({frames_per_chunk} frames/chunk)")
    print(f" Output Video    : {args.output}")
    print(f" Chunks Dir      : {args.chunks_dir} (managed with shutil)")
    print(f" Cropped Dir     : {args.crop_dir} (managed with shutil)")
    print(f" Metrics CSV     : {args.csv}")
    print(f"========================================================\n")

    # Clean and recreate chunk directories using shutil
    if os.path.exists(args.chunks_dir):
        shutil.rmtree(args.chunks_dir, ignore_errors=True)
    os.makedirs(args.chunks_dir, exist_ok=True)

    if os.path.exists(args.crop_dir):
        shutil.rmtree(args.crop_dir, ignore_errors=True)
    os.makedirs(args.crop_dir, exist_ok=True)

    # Main output video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_main = cv2.VideoWriter(args.output, fourcc, fps, (width, height))

    # Initialize tracker
    tracker = LipTracker()

    # Data structures for tracking & smoothing
    prev_gray = None
    lip_data = None
    lip_pts = None
    smoothed_box = None
    motion_trail = deque(maxlen=30)

    # Chunk writers
    chunk_idx = 1
    chunk_writer = None
    crop_writer = None
    crop_size = (256, 256)

    # Open CSV writer
    csv_headers = [
        "frame", "chunk_id", "time_s",
        "left_lip_x", "left_lip_y",
        "right_lip_x", "right_lip_y",
        "lip_center_x", "lip_center_y",
        "lip_box_x", "lip_box_y", "lip_box_w", "lip_box_h",
        "lip_width_px", "active_points"
    ]

    with open(args.csv, "w", newline="") as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(csv_headers)

        n = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            # Handle 5-second chunk creation with shutil directory structure
            if n % frames_per_chunk == 0:
                if chunk_writer is not None:
                    chunk_writer.release()
                if crop_writer is not None:
                    crop_writer.release()

                chunk_filename = os.path.join(args.chunks_dir, f"chunk_{chunk_idx:03d}_5s.mp4")
                crop_filename = os.path.join(args.crop_dir, f"chunk_{chunk_idx:03d}_lip_crop_5s.mp4")

                chunk_writer = cv2.VideoWriter(chunk_filename, fourcc, fps, (width, height))
                crop_writer = cv2.VideoWriter(crop_filename, fourcc, fps, crop_size)

                print(f" -> Writing Chunk #{chunk_idx:03d} (Time: {n / fps:.1f}s - {(n + frames_per_chunk) / fps:.1f}s)")
                chunk_idx += 1

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # 1. Optical Flow update between MTCNN re-detections
            if prev_gray is not None and lip_pts is not None and len(lip_pts) >= 3 and lip_data is not None:
                new_pts, st, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, lip_pts, None, **LK_PARAMS)
                good_new = new_pts[st == 1]
                good_old = lip_pts[st == 1]

                if len(good_new) >= 3:
                    dx, dy = np.median(good_new - good_old, axis=0)

                    # Shift lip landmarks and center by median motion delta
                    lx, ly = lip_data["left_corner"]
                    rx, ry = lip_data["right_corner"]
                    cx, cy = lip_data["center"]
                    bx, by, bw, bh = lip_data["box"]

                    lip_data["left_corner"] = (lx + dx, ly + dy)
                    lip_data["right_corner"] = (rx + dx, ry + dy)
                    lip_data["center"] = (cx + dx, cy + dy)
                    lip_data["box"] = (int(bx + dx), int(by + dy), bw, bh)
                    lip_pts = good_new.reshape(-1, 1, 2)
                else:
                    lip_pts = None

            # 2. Periodically or when points drop, re-detect with MTCNN
            need_redetect = (n % args.redetect == 0) or (lip_data is None) or (lip_pts is None) or (len(lip_pts) < 5)

            if need_redetect:
                det_res = tracker.detect_lips_mtcnn(frame)
                if det_res is not None:
                    lip_data = det_res
                    lip_pts = seed_lip_points(gray, lip_data["box"])

            # 3. Smooth bounding box with Exponential Moving Average (EMA)
            if lip_data is not None:
                bx, by, bw, bh = lip_data["box"]
                if smoothed_box is None:
                    smoothed_box = np.array([bx, by, bw, bh], dtype=np.float32)
                else:
                    alpha = 0.65
                    curr_box = np.array([bx, by, bw, bh], dtype=np.float32)
                    smoothed_box = alpha * curr_box + (1 - alpha) * smoothed_box

                sbx, sby, sbw, sbh = [int(v) for v in smoothed_box]

                # Update motion trail
                cx, cy = int(lip_data["center"][0]), int(lip_data["center"][1])
                motion_trail.append((cx, cy))

                # --- DRAW LIPS-ONLY VISUALIZATIONS ---
                # A. Lips Bounding Box (Vibrant Coral/Red)
                cv2.rectangle(frame, (sbx, sby), (sbx + sbw, sby + sbh), (0, 0, 255), 2)

                # B. Label & Header Badge
                badge_text = f"Lips (MTCNN) {lip_data['width']:.1f}px"
                cv2.rectangle(frame, (sbx, max(0, sby - 22)), (sbx + 180, sby), (0, 0, 255), -1)
                cv2.putText(frame, badge_text, (sbx + 4, max(12, sby - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

                # C. Left Corner, Right Corner & Center Markers
                lx, ly = int(lip_data["left_corner"][0]), int(lip_data["left_corner"][1])
                rx, ry = int(lip_data["right_corner"][0]), int(lip_data["right_corner"][1])

                cv2.circle(frame, (lx, ly), 5, (0, 255, 255), -1)  # Cyan Left Corner
                cv2.circle(frame, (rx, ry), 5, (0, 255, 255), -1)  # Cyan Right Corner
                cv2.circle(frame, (cx, cy), 5, (255, 0, 255), -1)  # Magenta Center

                # D. Optical flow feature points inside lip region
                active_pts_count = 0
                if lip_pts is not None:
                    active_pts_count = len(lip_pts)
                    for pt in lip_pts.reshape(-1, 2):
                        cv2.circle(frame, (int(pt[0]), int(pt[1])), 2, (0, 255, 0), -1)

                # E. Motion trail of lip center
                trail = list(motion_trail)
                for i in range(1, len(trail)):
                    thickness = int(np.sqrt(20 / float(i + 1)) * 1.5)
                    cv2.line(frame, trail[i - 1], trail[i], (0, 165, 255), thickness)

                # Crop Lip ROI for cropped video chunk output
                cbx1, cby1 = max(0, sbx), max(0, sby)
                cbx2, cby2 = min(width, sbx + sbw), min(height, sby + sbh)
                if cbx2 > cbx1 and cby2 > cby1:
                    crop_roi = frame[cby1:cby2, cbx1:cbx2]
                    resized_crop = cv2.resize(crop_roi, crop_size)
                else:
                    resized_crop = np.zeros((crop_size[1], crop_size[0], 3), dtype=np.uint8)
            else:
                active_pts_count = 0
                resized_crop = np.zeros((crop_size[1], crop_size[0], 3), dtype=np.uint8)

            # HUD Overlay
            cv2.rectangle(frame, (10, 10), (320, 75), (0, 0, 0), -1)
            cv2.rectangle(frame, (10, 10), (320, 75), (0, 0, 255), 1)
            cv2.putText(frame, "MTCNN LIPS TRACKER (5s CHUNKS)", (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(frame, f"Chunk #{chunk_idx - 1:02d} | Frame: {n} | Time: {n / fps:.2f}s", (20, 48),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(frame, f"Shutil Managed Chunks: {args.chunks_dir}", (20, 65),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1, cv2.LINE_AA)

            # Write outputs
            out_main.write(frame)
            if chunk_writer is not None:
                chunk_writer.write(frame)
            if crop_writer is not None:
                crop_writer.write(resized_crop)

            # CSV Logging
            if lip_data is not None:
                lx, ly = lip_data["left_corner"]
                rx, ry = lip_data["right_corner"]
                cx, cy = lip_data["center"]
                bx, by, bw, bh = lip_data["box"]
                w_px = lip_data["width"]
            else:
                lx = ly = rx = ry = cx = cy = bx = by = bw = bh = w_px = ""

            csv_writer.writerow([
                n, chunk_idx - 1, round(n / fps, 3),
                lx, ly, rx, ry, cx, cy,
                bx, by, bw, bh, w_px, active_pts_count
            ])

            if args.show:
                cv2.imshow("MTCNN Lips Tracking", frame)
                if cv2.waitKey(1) & 0xFF == 27:  # ESC key
                    break

            prev_gray = gray
            n += 1

    # Cleanup open file handles
    cap.release()
    out_main.release()
    if chunk_writer is not None:
        chunk_writer.release()
    if crop_writer is not None:
        crop_writer.release()
    cv2.destroyAllWindows()

    # Package chunks using shutil
    zip_archive_path = shutil.make_archive("lips_5sec_chunks_archive", "zip", args.chunks_dir)

    print(f"\n========================================================")
    print(f" TRACKING & CHUNKING COMPLETE!")
    print(f" Processed Frames    : {n}")
    print(f" Main Annotated Video: '{os.path.abspath(args.output)}'")
    print(f" 5s Chunks Directory : '{os.path.abspath(args.chunks_dir)}'")
    print(f" Cropped Lips Chunks : '{os.path.abspath(args.crop_dir)}'")
    print(f" Shutil Zip Archive  : '{os.path.abspath(zip_archive_path)}'")
    print(f" Tracking Data CSV   : '{os.path.abspath(args.csv)}'")
    print(f"========================================================\n")


if __name__ == "__main__":
    main()
