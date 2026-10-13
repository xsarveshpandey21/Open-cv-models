"""
MTCNN & Optical Flow Facial Tracker: Whole Face, Left/Right Ears, Eyes, Nose, and Lips.
Supports Live Real-Time Webcam Tracking, Video File Processing, On-Screen Coordinate Badges,
Interactive Cyber HUD Telemetry, 5-Second Video Chunks (shutil), and Structured CSV Logging.
"""

import argparse
import csv
from collections import deque
import math
import os
import sys
import shutil
import time
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

# Fallback Haar Cascades for Face, Eyes, and Smile/Mouth
FACE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
EYE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
SMILE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_smile.xml")

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
    """Open a native file selection dialog to choose a recorded video."""
    if not HAS_TK:
        return None
    try:
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        file_path = filedialog.askopenfilename(
            title="Select Video File for Face, Ears, Eyes & Lips Tracking",
            filetypes=[("Video Files", "*.mp4 *.avi *.mov *.mkv *.wmv"), ("All Files", "*.*")]
        )
        root.destroy()
        return file_path if file_path else None
    except Exception:
        return None


class FaceLipsEyesTracker:
    """
    Tracking engine for Whole Face, Left & Right Ears, Left & Right Eyes, Nose, and Lips.
    Extracts landmark coordinates, computes cranial geometry ROIs, and tracks using
    hybrid MTCNN + Lucas-Kanade Optical Flow.
    """
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
            print("Warning: facenet_pytorch MTCNN not found. Using Haar Cascade fallbacks.")

    def detect_features_mtcnn(self, frame_bgr):
        """
        Detect face, ears, eyes, nose, and mouth landmarks using MTCNN and cranial geometry.
        Returns dict containing all coordinates, bounding boxes, and distances.
        """
        if self.mtcnn is None:
            return self.detect_features_haar(frame_bgr)

        h_img, w_img = frame_bgr.shape[:2]
        img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)

        try:
            boxes, probs, landmarks = self.mtcnn.detect(pil_img, landmarks=True)
        except Exception:
            return self.detect_features_haar(frame_bgr)

        if boxes is None or landmarks is None or len(boxes) == 0 or len(landmarks) == 0:
            return self.detect_features_haar(frame_bgr)

        # Select primary face (largest bounding box area)
        areas = [(b[2] - b[0]) * (b[3] - b[1]) for b in boxes]
        best_idx = int(np.argmax(areas))

        box = boxes[best_idx]
        fbx1, fby1, fbx2, fby2 = [int(v) for v in box]
        fbx1, fby1 = max(0, fbx1), max(0, fby1)
        fbx2, fby2 = min(w_img, fbx2), min(h_img, fby2)
        fbw, fbh = max(1, fbx2 - fbx1), max(1, fby2 - fby1)
        face_center_x, face_center_y = fbx1 + fbw / 2.0, fby1 + fbh / 2.0

        # MTCNN 5 Landmarks: [0: left_eye, 1: right_eye, 2: nose, 3: left_mouth, 4: right_mouth]
        lm = landmarks[best_idx]
        left_eye_pt = (float(lm[0][0]), float(lm[0][1]))
        right_eye_pt = (float(lm[1][0]), float(lm[1][1]))
        nose_pt = (float(lm[2][0]), float(lm[2][1]))
        left_m = (float(lm[3][0]), float(lm[3][1]))
        right_m = (float(lm[4][0]), float(lm[4][1]))

        # Inter-eye distance & eye orientation vector
        eye_dx = right_eye_pt[0] - left_eye_pt[0]
        eye_dy = right_eye_pt[1] - left_eye_pt[1]
        eye_dist = max(1.0, float(math.hypot(eye_dx, eye_dy)))

        # Normalized direction vector along eye line and perpendicular downward vector
        ux, uy = eye_dx / eye_dist, eye_dy / eye_dist
        px, py = -uy, ux  # perpendicular pointing downwards towards chin

        # Eye ROI Bounding Boxes
        eye_box_w = int(max(15.0, eye_dist * 0.45))
        eye_box_h = int(max(12.0, eye_dist * 0.35))

        le_box_x = max(0, int(left_eye_pt[0] - eye_box_w / 2.0))
        le_box_y = max(0, int(left_eye_pt[1] - eye_box_h / 2.0))
        le_box_w = min(w_img - le_box_x, eye_box_w)
        le_box_h = min(h_img - le_box_y, eye_box_h)

        re_box_x = max(0, int(right_eye_pt[0] - eye_box_w / 2.0))
        re_box_y = max(0, int(right_eye_pt[1] - eye_box_h / 2.0))
        re_box_w = min(w_img - re_box_x, eye_box_w)
        re_box_h = min(h_img - re_box_y, eye_box_h)

        # -------------------------------------------------------------
        # Left Ear & Right Ear Cranial Geometry Calculation:
        # Anatomically, the ear tragus is located laterally on each side of the head,
        # at the level between the eye line and the upper nasal ala.
        # -------------------------------------------------------------
        # Left Ear (viewer's left side): lateral offset outward from left eye
        raw_lear_x = left_eye_pt[0] - 0.78 * eye_dist * ux + 0.22 * eye_dist * px
        raw_lear_y = left_eye_pt[1] - 0.78 * eye_dist * uy + 0.22 * eye_dist * py
        # Constrain naturally within cranial perimeter
        lear_x = float(np.clip(raw_lear_x, max(0, fbx1 - fbw * 0.12), fbx1 + fbw * 0.25))
        lear_y = float(np.clip(raw_lear_y, fby1 + fbh * 0.22, fby1 + fbh * 0.70))

        # Right Ear (viewer's right side): lateral offset outward from right eye
        raw_rear_x = right_eye_pt[0] + 0.78 * eye_dist * ux + 0.22 * eye_dist * px
        raw_rear_y = right_eye_pt[1] + 0.78 * eye_dist * uy + 0.22 * eye_dist * py
        rear_x = float(np.clip(raw_rear_x, fbx1 + fbw * 0.75, min(w_img, fbx1 + fbw * 1.12)))
        rear_y = float(np.clip(raw_rear_y, fby1 + fbh * 0.22, fby1 + fbh * 0.70))

        ear_span = float(math.hypot(rear_x - lear_x, rear_y - lear_y))

        # Ear ROI Bounding Boxes
        ear_box_w = int(max(14.0, eye_dist * 0.35))
        ear_box_h = int(max(20.0, eye_dist * 0.65))

        lear_bx = max(0, int(lear_x - ear_box_w / 2.0))
        lear_by = max(0, int(lear_y - ear_box_h / 2.0))
        lear_bw = min(w_img - lear_bx, ear_box_w)
        lear_bh = min(h_img - lear_by, ear_box_h)

        rear_bx = max(0, int(rear_x - ear_box_w / 2.0))
        rear_by = max(0, int(rear_y - ear_box_h / 2.0))
        rear_bw = min(w_img - rear_bx, ear_box_w)
        rear_bh = min(h_img - rear_by, ear_box_h)

        # -------------------------------------------------------------
        # Nose ROI Bounding Box:
        # -------------------------------------------------------------
        nose_box_w = int(max(16.0, eye_dist * 0.38))
        nose_box_h = int(max(16.0, eye_dist * 0.42))
        nose_bx = max(0, int(nose_pt[0] - nose_box_w / 2.0))
        nose_by = max(0, int(nose_pt[1] - nose_box_h / 2.0))
        nose_bw = min(w_img - nose_bx, nose_box_w)
        nose_bh = min(h_img - nose_by, nose_box_h)

        # -------------------------------------------------------------
        # Lip Center, Width & ROI Bounding Box:
        # -------------------------------------------------------------
        lip_center_x = (left_m[0] + right_m[0]) / 2.0
        lip_center_y = (left_m[1] + right_m[1]) / 2.0
        lip_width = float(math.hypot(right_m[0] - left_m[0], right_m[1] - left_m[1]))

        lip_box_w = int(max(20.0, lip_width * 1.6))
        lip_box_h = int(max(15.0, lip_width * 1.0))
        lip_box_x = max(0, int(lip_center_x - lip_box_w / 2.0))
        lip_box_y = max(0, int(lip_center_y - lip_box_h * 0.45))
        lip_box_w = min(w_img - lip_box_x, lip_box_w)
        lip_box_h = min(h_img - lip_box_y, lip_box_h)

        return {
            "face_box": (fbx1, fby1, fbw, fbh),
            "face_center": (float(face_center_x), float(face_center_y)),
            "left_ear": (lear_x, lear_y),
            "left_ear_box": (lear_bx, lear_by, lear_bw, lear_bh),
            "right_ear": (rear_x, rear_y),
            "right_ear_box": (rear_bx, rear_by, rear_bw, rear_bh),
            "ear_span": ear_span,
            "left_eye": left_eye_pt,
            "left_eye_box": (le_box_x, le_box_y, le_box_w, le_box_h),
            "right_eye": right_eye_pt,
            "right_eye_box": (re_box_x, re_box_y, re_box_w, re_box_h),
            "eye_distance": eye_dist,
            "nose": nose_pt,
            "nose_box": (nose_bx, nose_by, nose_bw, nose_bh),
            "left_corner": left_m,
            "right_corner": right_m,
            "center": (float(lip_center_x), float(lip_center_y)),
            "width": lip_width,
            "box": (lip_box_x, lip_box_y, lip_box_w, lip_box_h)
        }

    def detect_features_haar(self, frame_bgr):
        """Fallback detection using OpenCV Haar Cascades."""
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        h_img, w_img = gray.shape[:2]

        faces = FACE_CASCADE.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))
        if len(faces) == 0:
            return None

        fx, fy, fw, fh = max(faces, key=lambda f: f[2] * f[3])
        face_center_x, face_center_y = fx + fw / 2.0, fy + fh / 2.0

        # Estimate eye locations inside upper half of face
        roi_upper = gray[fy: fy + int(fh * 0.55), fx: fx + fw]
        eyes = EYE_CASCADE.detectMultiScale(roi_upper, 1.1, 3, minSize=(20, 20))

        if len(eyes) >= 2:
            eyes_sorted = sorted(eyes, key=lambda e: e[0])
            ex1, ey1, ew1, eh1 = eyes_sorted[0]
            ex2, ey2, ew2, eh2 = eyes_sorted[-1]
            le_x, le_y = fx + ex1 + ew1 / 2.0, fy + ey1 + eh1 / 2.0
            re_x, re_y = fx + ex2 + ew2 / 2.0, fy + ey2 + eh2 / 2.0
        else:
            le_x, le_y = fx + fw * 0.32, fy + fh * 0.35
            re_x, re_y = fx + fw * 0.68, fy + fh * 0.35

        eye_dist = max(1.0, float(math.hypot(re_x - le_x, re_y - le_y)))
        eye_box_w, eye_box_h = int(fw * 0.25), int(fh * 0.18)

        le_box = (max(0, int(le_x - eye_box_w / 2)), max(0, int(le_y - eye_box_h / 2)), eye_box_w, eye_box_h)
        re_box = (max(0, int(re_x - eye_box_w / 2)), max(0, int(re_y - eye_box_h / 2)), eye_box_w, eye_box_h)

        # Ears estimation from face box
        lear_x = fx + fw * 0.05
        lear_y = fy + fh * 0.45
        rear_x = fx + fw * 0.95
        rear_y = fy + fh * 0.45
        ear_span = float(math.hypot(rear_x - lear_x, rear_y - lear_y))
        ear_w, ear_h = int(fw * 0.15), int(fh * 0.28)
        lear_box = (max(0, int(lear_x - ear_w / 2)), max(0, int(lear_y - ear_h / 2)), ear_w, ear_h)
        rear_box = (max(0, int(rear_x - ear_w / 2)), max(0, int(rear_y - ear_h / 2)), ear_w, ear_h)

        # Nose estimation
        nx, ny = fx + fw * 0.5, fy + fh * 0.52
        nw, nh = int(fw * 0.18), int(fh * 0.20)
        nbox = (max(0, int(nx - nw / 2)), max(0, int(ny - nh / 2)), nw, nh)

        # Estimate lips inside lower region of face
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
            "face_box": (fx, fy, fw, fh),
            "face_center": (float(face_center_x), float(face_center_y)),
            "left_ear": (float(lear_x), float(lear_y)),
            "left_ear_box": lear_box,
            "right_ear": (float(rear_x), float(rear_y)),
            "right_ear_box": rear_box,
            "ear_span": ear_span,
            "left_eye": (float(le_x), float(le_y)),
            "left_eye_box": le_box,
            "right_eye": (float(re_x), float(re_y)),
            "right_eye_box": re_box,
            "eye_distance": eye_dist,
            "nose": (float(nx), float(ny)),
            "nose_box": nbox,
            "left_corner": (float(lx), float(ly)),
            "right_corner": (float(rx), float(ry)),
            "center": (float(cx), float(cy)),
            "width": w,
            "box": (bx, by, bw, bh)
        }

    # Backward compatibility alias
    def detect_lips_mtcnn(self, frame_bgr):
        return self.detect_features_mtcnn(frame_bgr)


# Backward compatibility alias
LipTracker = FaceLipsEyesTracker


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


def draw_coordinate_badge(img, text, pos, bg_color=(12, 16, 24), text_color=(255, 255, 255), scale=0.34):
    """Render a clean text badge with translucent high-contrast background."""
    x, y = int(pos[0]), int(pos[1])
    font = cv2.FONT_HERSHEY_SIMPLEX
    (text_w, text_h), baseline = cv2.getTextSize(text, font, scale, 1)
    pad = 3
    x1 = max(0, x - pad)
    y1 = max(0, y - text_h - pad)
    x2 = min(img.shape[1], x + text_w + pad)
    y2 = min(img.shape[0], y + baseline + pad)

    # Dark background pill
    cv2.rectangle(img, (x1, y1), (x2, y2), bg_color, -1)
    cv2.rectangle(img, (x1, y1), (x2, y2), (60, 75, 95), 1)
    cv2.putText(img, text, (x, y), font, scale, text_color, 1, cv2.LINE_AA)


def draw_corner_rect(img, pt1, pt2, color, thickness=1, corner_len=14, corner_thickness=2):
    """Draw a bounding box with futuristic high-tech highlighted corner accents."""
    x1, y1 = pt1
    x2, y2 = pt2
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

    c_len = min(corner_len, abs(x2 - x1) // 3, abs(y2 - y1) // 3)
    if c_len < 2:
        return

    # Top-Left Corner
    cv2.line(img, (x1, y1), (x1 + c_len, y1), color, corner_thickness)
    cv2.line(img, (x1, y1), (x1, y1 + c_len), color, corner_thickness)

    # Top-Right Corner
    cv2.line(img, (x2, y1), (x2 - c_len, y1), color, corner_thickness)
    cv2.line(img, (x2, y1), (x2, y1 + c_len), color, corner_thickness)

    # Bottom-Left Corner
    cv2.line(img, (x1, y2), (x1 + c_len, y2), color, corner_thickness)
    cv2.line(img, (x1, y2), (x1, y2 - c_len), color, corner_thickness)

    # Bottom-Right Corner
    cv2.line(img, (x2, y2), (x2 - c_len, y2), color, corner_thickness)
    cv2.line(img, (x2, y2), (x2, y2 - c_len), color, corner_thickness)


def fit_parabola_to_lips(left_corner, center, right_corner, lip_pts=None):
    """
    Fits quadratic curve y = a*x^2 + b*x + c to lip landmarks.
    Returns dict with a, b, c, vertex apex, and generated curve points.
    """
    x1, y1 = left_corner
    xc, yc = center
    x2, y2 = right_corner

    xs = [x1, xc, x2]
    ys = [y1, yc, y2]

    if lip_pts is not None and len(lip_pts) >= 3:
        pts = lip_pts.reshape(-1, 2)
        xs.extend(pts[:, 0].tolist())
        ys.extend(pts[:, 1].tolist())

    xs_arr = np.array(xs, dtype=np.float64)
    ys_arr = np.array(ys, dtype=np.float64)

    try:
        poly_coeffs = np.polyfit(xs_arr, ys_arr, 2)
        a, b, c = poly_coeffs[0], poly_coeffs[1], poly_coeffs[2]
        apex_x = -b / (2.0 * a) if abs(a) > 1e-9 else xc
        apex_y = a * (apex_x ** 2) + b * apex_x + c

        min_x = int(min(xs))
        max_x = int(max(xs))
        if max_x <= min_x:
            max_x = min_x + 10

        x_vals = np.linspace(min_x, max_x, 40)
        y_vals = a * (x_vals ** 2) + b * x_vals + c
        curve_pts = np.vstack((x_vals, y_vals)).T.astype(np.int32)

        return {
            "a": float(a),
            "b": float(b),
            "c": float(c),
            "apex_x": float(apex_x),
            "apex_y": float(apex_y),
            "curve_pts": curve_pts
        }
    except Exception:
        return None


def get_aligned_lip_box(left_corner, center, right_corner, lip_pts=None, frame_shape=None):
    """Computes tightly-bounded lip box around all active lip points."""
    all_x = [left_corner[0], center[0], right_corner[0]]
    all_y = [left_corner[1], center[1], right_corner[1]]

    if lip_pts is not None and len(lip_pts) > 0:
        pts = lip_pts.reshape(-1, 2)
        all_x.extend(pts[:, 0].tolist())
        all_y.extend(pts[:, 1].tolist())

    min_x, max_x = min(all_x), max(all_x)
    min_y, max_y = min(all_y), max(all_y)

    width = max_x - min_x
    height = max_y - min_y

    pad_x = max(6, int(width * 0.12))
    pad_y = max(6, int(height * 0.35))

    bx1 = int(round(min_x - pad_x))
    by1 = int(round(min_y - pad_y))
    bx2 = int(round(max_x + pad_x))
    by2 = int(round(max_y + pad_y))

    if frame_shape is not None:
        h_img, w_img = frame_shape[:2]
        bx1 = max(0, bx1)
        by1 = max(0, by1)
        bx2 = min(w_img, bx2)
        by2 = min(h_img, by2)

    bw = max(1, bx2 - bx1)
    bh = max(1, by2 - by1)
    return bx1, by1, bw, bh


def draw_parabolic_lip_symbol(frame, parabola_info, lip_box=None):
    """Renders Parabolic Curve line over lips."""
    if parabola_info is None:
        return
    curve_pts = parabola_info["curve_pts"]
    cv2.polylines(frame, [curve_pts], isClosed=False, color=(255, 230, 0), thickness=2, lineType=cv2.LINE_AA)


def draw_hud_panel(frame, n, fps, chunk_idx, features, parabola_info=None, is_live=False):
    """
    Render sleek cyber Telemetry HUD panel on top-left of video frame
    displaying real-time tracking details for Face, Ears, Eyes, Nose, Lips.
    """
    hud_w, hud_h = 320, 205
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (10 + hud_w, 10 + hud_h), (8, 12, 18), -1)
    cv2.addWeighted(overlay, 0.82, frame, 0.18, 0, frame)
    cv2.rectangle(frame, (10, 10), (10 + hud_w, 10 + hud_h), (0, 229, 255), 1)

    # Status & Mode Indicator
    mode_text = "LIVE WEBCAM (CAM 0)" if is_live else "VIDEO PLAYBACK"
    mode_color = (0, 0, 255) if is_live else (0, 215, 255)
    cv2.circle(frame, (22, 26), 4, mode_color, -1)
    cv2.putText(frame, mode_text, (32, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, f"FPS: {fps:.1f} | Frame: {n}", (190, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (180, 200, 220), 1, cv2.LINE_AA)
    cv2.line(frame, (20, 36), (10 + hud_w - 10, 36), (40, 55, 75), 1)

    if features is not None:
        fbx, fby, fbw, fbh = features["face_box"]
        fcx, fcy = features["face_center"]
        lear_x, lear_y = features["left_ear"]
        rear_x, rear_y = features["right_ear"]
        ear_span = features.get("ear_span", 0.0)
        lex, ley = features["left_eye"]
        rex, rey = features["right_eye"]
        eye_d = features["eye_distance"]
        nx, ny = features.get("nose", (0.0, 0.0))
        llx, lly = features["left_corner"]
        rlx, rly = features["right_corner"]
        lcx, lcy = features["center"]
        lip_w = features["width"]

        rows = [
            ("FACE", f"[{fbx},{fby},{fbw},{fbh}] Ctr:({fcx:.0f},{fcy:.0f})", (0, 229, 255)),
            ("EARS", f"L:({lear_x:.0f},{lear_y:.0f}) R:({rear_x:.0f},{rear_y:.0f}) | {ear_span:.1f}px", (0, 180, 255)),
            ("EYES", f"L:({lex:.0f},{ley:.0f}) R:({rex:.0f},{rey:.0f}) | {eye_d:.1f}px", (255, 170, 0)),
            ("NOSE", f"({nx:.0f},{ny:.0f})", (220, 60, 255)),
            ("LIPS", f"Ctr:({lcx:.0f},{lcy:.0f}) | W:{lip_w:.1f}px", (100, 100, 255)),
        ]

        if parabola_info is not None:
            a_val = parabola_info["a"]
            ax_p = parabola_info["apex_x"]
            ay_p = parabola_info["apex_y"]
            rows.append(("PARABOLA", f"a:{a_val:+.4f} Apex:({ax_p:.0f},{ay_p:.0f})", (255, 230, 0)))

        y_offset = 53
        for label, val, color in rows:
            cv2.putText(frame, f"{label}:", (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.33, (150, 170, 190), 1, cv2.LINE_AA)
            cv2.putText(frame, val, (85, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.33, color, 1, cv2.LINE_AA)
            y_offset += 17

        # Live tracking indicator
        cv2.putText(frame, "STATUS: LOCKED [MTCNN + OPTICAL FLOW]", (20, 195),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, (0, 255, 120), 1, cv2.LINE_AA)
    else:
        cv2.putText(frame, "SEARCHING FOR FACIAL FEATURES...", (20, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 80, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, "Please face camera directly", (20, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, (150, 150, 150), 1, cv2.LINE_AA)


def draw_all_feature_overlays(frame, features, lip_pts=None, parabola_info=None, show_badges=True, show_boxes=True):
    """
    Renders high-contrast, beautiful visual bounding boxes, crosshairs,
    and coordinate badges for Face, Left/Right Ears, Left/Right Eyes, Nose, and Lips.
    """
    if features is None:
        return

    # 1. Whole Face Bounding Box & Center Badge
    fbx, fby, fbw, fbh = features["face_box"]
    fcx, fcy = features["face_center"]
    if show_boxes:
        draw_corner_rect(frame, (fbx, fby), (fbx + fbw, fby + fbh), (0, 229, 255), thickness=1, corner_len=18, corner_thickness=3)
        cv2.drawMarker(frame, (int(fcx), int(fcy)), (0, 229, 255), cv2.MARKER_CROSS, 10, 1)
    if show_badges:
        draw_coordinate_badge(frame, f"FACE [{fbx},{fby},{fbw},{fbh}]", (fbx, max(14, fby - 6)), bg_color=(10, 20, 30), text_color=(0, 229, 255))

    # 2. Left Ear & Right Ear Overlays
    lear_x, lear_y = features["left_ear"]
    rear_x, rear_y = features["right_ear"]
    lear_bx, lear_by, lear_bw, lear_bh = features["left_ear_box"]
    rear_bx, rear_by, rear_bw, rear_bh = features["right_ear_box"]
    ear_color = (0, 190, 255)  # Amber gold (BGR)

    if show_boxes:
        # Left Ear Box & Marker
        draw_corner_rect(frame, (lear_bx, lear_by), (lear_bx + lear_bw, lear_by + lear_bh), ear_color, thickness=1, corner_len=8, corner_thickness=2)
        cv2.circle(frame, (int(lear_x), int(lear_y)), 4, ear_color, -1)
        # Right Ear Box & Marker
        draw_corner_rect(frame, (rear_bx, rear_by), (rear_bx + rear_bw, rear_by + rear_bh), ear_color, thickness=1, corner_len=8, corner_thickness=2)
        cv2.circle(frame, (int(rear_x), int(rear_y)), 4, ear_color, -1)
        # Optional ear span connection line
        cv2.line(frame, (int(lear_x), int(lear_y)), (int(rear_x), int(rear_y)), (0, 120, 180), 1, cv2.LINE_AA)

    if show_badges:
        draw_coordinate_badge(frame, f"L-EAR ({lear_x:.0f},{lear_y:.0f})", (lear_bx, max(12, lear_by - 4)), text_color=ear_color)
        draw_coordinate_badge(frame, f"R-EAR ({rear_x:.0f},{rear_y:.0f})", (rear_bx, max(12, rear_by - 4)), text_color=ear_color)

    # 3. Left Eye & Right Eye Overlays
    lex, ley = features["left_eye"]
    rex, rey = features["right_eye"]
    le_bx, le_by, le_bw, le_bh = features["left_eye_box"]
    re_bx, re_by, re_bw, re_bh = features["right_eye_box"]
    eye_color = (255, 170, 0)  # Electric Blue (BGR)

    if show_boxes:
        draw_corner_rect(frame, (le_bx, le_by), (le_bx + le_bw, le_by + le_bh), eye_color, thickness=1, corner_len=8, corner_thickness=2)
        cv2.circle(frame, (int(lex), int(ley)), 3, (255, 255, 255), -1)
        cv2.circle(frame, (int(lex), int(ley)), 6, eye_color, 1)

        draw_corner_rect(frame, (re_bx, re_by), (re_bx + re_bw, re_by + re_bh), eye_color, thickness=1, corner_len=8, corner_thickness=2)
        cv2.circle(frame, (int(rex), int(rey)), 3, (255, 255, 255), -1)
        cv2.circle(frame, (int(rex), int(rey)), 6, eye_color, 1)

        # Inter-eye distance dashed line
        cv2.line(frame, (int(lex), int(ley)), (int(rex), int(rey)), (120, 180, 255), 1, cv2.LINE_AA)

    if show_badges:
        draw_coordinate_badge(frame, f"L-EYE ({lex:.0f},{ley:.0f})", (le_bx, max(12, le_by - 4)), text_color=eye_color)
        draw_coordinate_badge(frame, f"R-EYE ({rex:.0f},{rey:.0f})", (re_bx, max(12, re_by - 4)), text_color=eye_color)
        mid_eye_x = (lex + rex) / 2.0
        mid_eye_y = (ley + rey) / 2.0 - 10
        draw_coordinate_badge(frame, f"DIST:{features['eye_distance']:.1f}px", (mid_eye_x - 30, mid_eye_y), text_color=(180, 255, 120))

    # 4. Nose Landmark Overlay
    if "nose" in features:
        nx, ny = features["nose"]
        nbx, nby, nbw, nbh = features["nose_box"]
        nose_color = (220, 60, 255)  # Magenta (BGR)
        if show_boxes:
            cv2.drawMarker(frame, (int(nx), int(ny)), nose_color, cv2.MARKER_TRIANGLE_UP, 8, 1)
        if show_badges:
            draw_coordinate_badge(frame, f"NOSE ({nx:.0f},{ny:.0f})", (nx + 10, ny - 6), text_color=nose_color)

    # 5. Lips Overlays & Optical Flow Points
    llx, lly = features["left_corner"]
    rlx, rly = features["right_corner"]
    lcx, lcy = features["center"]
    lbx, lby, lbw, lbh = features["box"]
    lip_color = (80, 80, 255)  # Coral pink (BGR)

    if show_boxes:
        draw_corner_rect(frame, (lbx, lby), (lbx + lbw, lby + lbh), lip_color, thickness=1, corner_len=10, corner_thickness=2)
        cv2.circle(frame, (int(llx), int(lly)), 4, (0, 255, 120), -1)  # Left corner green
        cv2.circle(frame, (int(rlx), int(rly)), 4, (0, 255, 120), -1)  # Right corner green
        cv2.circle(frame, (int(lcx), int(lcy)), 3, (255, 255, 255), -1)

    # Shi-Tomasi feature points inside lips
    if lip_pts is not None and len(lip_pts) > 0:
        for pt in lip_pts:
            px_val, py_val = pt.ravel()
            cv2.circle(frame, (int(px_val), int(py_val)), 2, (0, 255, 0), -1)

    # Parabola fit line
    if parabola_info is not None:
        draw_parabolic_lip_symbol(frame, parabola_info, (lbx, lby, lbw, lbh))

    if show_badges:
        draw_coordinate_badge(frame, f"LIPS ({lcx:.0f},{lcy:.0f}) W:{features['width']:.1f}px", (lbx, max(12, lby - 6)), text_color=lip_color)


def main():
    ap = argparse.ArgumentParser(description="Whole Face, Left/Right Ears, Eyes, Nose & Lips Feature Tracker with Live Webcam")
    ap.add_argument("--input", default=None, help="Input recorded video path or webcam index (0). Default opens choice prompt.")
    ap.add_argument("--live", "--webcam", action="store_true", help="Launch live webcam tracking directly (Camera 0).")
    ap.add_argument("--output", default="recorded_face_lips_eyes_tracked.mp4", help="Output annotated main video path.")
    ap.add_argument("--chunks-dir", default="lips_chunks_5s", help="Output directory for 5-second video chunks.")
    ap.add_argument("--crop-dir", default="lips_cropped_chunks_5s", help="Output directory for cropped lip video chunks.")
    ap.add_argument("--csv", default="recorded_log.csv", help="Output CSV metrics log path with coordinates.")
    ap.add_argument("--chunk-sec", type=float, default=5.0, help="Chunk duration in seconds (default: 5.0s).")
    ap.add_argument("--redetect", type=int, default=10, help="Re-detect with MTCNN every N frames.")
    ap.add_argument("--show", action="store_true", default=True, help="Display live preview window during processing.")
    ap.add_argument("--no-show", action="store_false", dest="show", help="Run in headless mode without preview window.")
    args = ap.parse_args()

    input_source = args.input
    is_live = False

    if args.live:
        input_source = 0
        is_live = True
    elif input_source is None:
        print("\n========================================================")
        print(" WHOLE FACE, EARS, EYES & LIPS TRACKING STUDIO")
        print("========================================================")
        print(" Choose an input source:")
        print("   [1] Live Webcam Tracking (Camera 0)")
        print("   [2] Select Video File via GUI Picker")
        print("   [3] Default Recorded Video ('tracked_lips.mp4')")
        print("========================================================")

        user_choice = None
        # Check if stdin is interactive
        if sys.stdin.isatty():
            try:
                user_choice = input("Enter choice [1/2/3] (default: 1): ").strip()
            except Exception:
                user_choice = "1"
        else:
            user_choice = "1"

        if user_choice == "2":
            input_source = select_video_file_gui()
            if not input_source:
                if os.path.exists("tracked_lips.mp4"):
                    input_source = "tracked_lips.mp4"
                else:
                    input_source = 0
                    is_live = True
        elif user_choice == "3":
            if os.path.exists("tracked_lips.mp4"):
                input_source = "tracked_lips.mp4"
            else:
                input_source = 0
                is_live = True
        else:
            input_source = 0
            is_live = True

    # Check if input source is a webcam index
    if isinstance(input_source, str) and input_source.isdigit():
        input_source = int(input_source)
        is_live = True
    elif input_source == 0:
        is_live = True

    if is_live and sys.platform.startswith('win'):
        cap = cv2.VideoCapture(int(input_source), cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap = cv2.VideoCapture(int(input_source))
    else:
        cap = cv2.VideoCapture(input_source)

    if not cap.isOpened():
        print(f"Error: Unable to open video source '{input_source}'. Trying webcam 0...")
        if sys.platform.startswith('win'):
            cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        else:
            cap = cv2.VideoCapture(0)
        is_live = True
        if not cap.isOpened():
            raise SystemExit("Error: Could not open any video source or webcam.")

    # Camera resolution & FPS
    if is_live:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if fps <= 0 or fps > 120:
        fps = 30.0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames_per_chunk = max(1, int(fps * args.chunk_sec))

    print(f"\n========================================================")
    print(f" SOURCE          : {'[LIVE WEBCAM] (Camera 0)' if is_live else f'[VIDEO FILE] {input_source}'}")
    print(f" RESOLUTION      : {width}x{height} @ {fps:.1f} FPS")
    print(f" FEATURES        : Face, Left/Right Ears, Eyes, Nose, Lips")
    print(f" CHUNK INTERVAL  : {args.chunk_sec}s ({frames_per_chunk} frames/chunk)")
    print(f" OUTPUT VIDEO    : {args.output}")
    print(f" CSV TELEMETRY   : {args.csv}")
    print(f" LIVE WINDOW     : {'Enabled (Press ESC or Q to quit)' if args.show else 'Headless'}")
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
    tracker = FaceLipsEyesTracker()

    prev_gray = None
    features_data = None
    lip_pts = None
    smoothed_face_box = None
    smoothed_lip_box = None
    motion_trail = deque(maxlen=25)

    chunk_idx = 1
    chunk_writer = None
    crop_writer = None
    crop_size = (256, 256)

    show_hud = True
    show_boxes = True
    fps_smooth = fps
    t_prev = time.time()

    # Complete CSV headers including Face, Ears, Eyes, Nose, Lips, and Parabolic Curve
    csv_headers = [
        "frame", "chunk_id", "time_s",
        "face_box_x", "face_box_y", "face_box_w", "face_box_h",
        "face_center_x", "face_center_y",
        "left_ear_x", "left_ear_y", "left_ear_w", "left_ear_h",
        "right_ear_x", "right_ear_y", "right_ear_w", "right_ear_h", "ear_span_px",
        "left_eye_x", "left_eye_y", "left_eye_w", "left_eye_h",
        "right_eye_x", "right_eye_y", "right_eye_w", "right_eye_h", "inter_eye_dist_px",
        "nose_x", "nose_y", "nose_w", "nose_h",
        "left_lip_x", "left_lip_y",
        "right_lip_x", "right_lip_y",
        "lip_center_x", "lip_center_y",
        "lip_box_x", "lip_box_y", "lip_box_w", "lip_box_h",
        "lip_width_px", "active_lip_points",
        "parabola_a", "parabola_b", "parabola_c", "parabola_apex_x", "parabola_apex_y"
    ]

    with open(args.csv, "w", newline="") as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(csv_headers)

        n = 0
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break

                t_curr = time.time()
                dt = t_curr - t_prev
                t_prev = t_curr
                if dt > 0:
                    fps_instant = 1.0 / dt
                    fps_smooth = 0.9 * fps_smooth + 0.1 * fps_instant

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                # Initialize or rotate 5-second chunk writers
                if n % frames_per_chunk == 0:
                    if chunk_writer is not None:
                        chunk_writer.release()
                    if crop_writer is not None:
                        crop_writer.release()

                    c_file = os.path.join(args.chunks_dir, f"chunk_{chunk_idx:04d}.mp4")
                    cr_file = os.path.join(args.crop_dir, f"lip_crop_{chunk_idx:04d}.mp4")
                    chunk_writer = cv2.VideoWriter(c_file, fourcc, fps, (width, height))
                    crop_writer = cv2.VideoWriter(cr_file, fourcc, fps, crop_size)
                    chunk_idx += 1

                # 1. Optical Flow tracking update between MTCNN re-detections
                if prev_gray is not None and lip_pts is not None and len(lip_pts) >= 3 and features_data is not None:
                    new_pts, st_flow, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, lip_pts, None, **LK_PARAMS)
                    if new_pts is not None and st_flow is not None:
                        good_new = new_pts[st_flow == 1]
                        good_old = lip_pts[st_flow == 1]

                        if len(good_new) >= 3:
                            dx, dy = np.median(good_new - good_old, axis=0)

                            # Shift all facial features by optical flow displacement
                            fbx, fby, fbw, fbh = features_data["face_box"]
                            fc_x, fc_y = features_data["face_center"]
                            lear_x, lear_y = features_data["left_ear"]
                            rear_x, rear_y = features_data["right_ear"]
                            le_bx, le_by, le_bw, le_bh = features_data["left_ear_box"]
                            re_bx, re_by, re_bw, re_bh = features_data["right_ear_box"]

                            lex, ley = features_data["left_eye"]
                            rex, rey = features_data["right_eye"]
                            eye_bx1, eye_by1, eye_bw1, eye_bh1 = features_data["left_eye_box"]
                            eye_bx2, eye_by2, eye_bw2, eye_bh2 = features_data["right_eye_box"]

                            nx, ny = features_data["nose"]
                            nbx, nby, nbw, nbh = features_data["nose_box"]

                            llx, lly = features_data["left_corner"]
                            rlx, rly = features_data["right_corner"]
                            lcx, lcy = features_data["center"]
                            lbx, lby, lbw, lbh = features_data["box"]

                            features_data["face_box"] = (int(fbx + dx), int(fby + dy), fbw, fbh)
                            features_data["face_center"] = (fc_x + dx, fc_y + dy)
                            features_data["left_ear"] = (lear_x + dx, lear_y + dy)
                            features_data["right_ear"] = (rear_x + dx, rear_y + dy)
                            features_data["left_ear_box"] = (int(le_bx + dx), int(le_by + dy), le_bw, le_bh)
                            features_data["right_ear_box"] = (int(re_bx + dx), int(re_by + dy), re_bw, re_bh)

                            features_data["left_eye"] = (lex + dx, ley + dy)
                            features_data["right_eye"] = (rex + dx, rey + dy)
                            features_data["left_eye_box"] = (int(eye_bx1 + dx), int(eye_by1 + dy), eye_bw1, eye_bh1)
                            features_data["right_eye_box"] = (int(eye_bx2 + dx), int(eye_by2 + dy), eye_bw2, eye_bh2)

                            features_data["nose"] = (nx + dx, ny + dy)
                            features_data["nose_box"] = (int(nbx + dx), int(nby + dy), nbw, nbh)

                            features_data["left_corner"] = (llx + dx, lly + dy)
                            features_data["right_corner"] = (rlx + dx, rly + dy)
                            features_data["center"] = (lcx + dx, lcy + dy)
                            features_data["box"] = (int(lbx + dx), int(lby + dy), lbw, lbh)

                            lip_pts = good_new.reshape(-1, 1, 2)
                        else:
                            lip_pts = None
                    else:
                        lip_pts = None

                # 2. Periodic MTCNN re-detection
                need_redetect = (n % args.redetect == 0) or (features_data is None) or (lip_pts is None) or (len(lip_pts) < 5)
                if need_redetect:
                    det_res = tracker.detect_features_mtcnn(frame)
                    if det_res is not None:
                        features_data = det_res
                        lip_pts = seed_lip_points(gray, features_data["box"])

                parabola_info = None
                active_pts_count = len(lip_pts) if lip_pts is not None else 0

                # 3. Fit Parabola and motion trails
                if features_data is not None:
                    parabola_info = fit_parabola_to_lips(
                        features_data["left_corner"],
                        features_data["center"],
                        features_data["right_corner"],
                        lip_pts=lip_pts
                    )
                    motion_trail.append(features_data["center"])

                # Render annotated visuals
                annotated_frame = frame.copy()

                # Draw motion trail of lips
                for i in range(1, len(motion_trail)):
                    pt_a = (int(motion_trail[i - 1][0]), int(motion_trail[i - 1][1]))
                    pt_b = (int(motion_trail[i][0]), int(motion_trail[i][1]))
                    cv2.line(annotated_frame, pt_a, pt_b, (0, 255, 120), 1, cv2.LINE_AA)

                # Draw all facial feature bounding boxes, crosshairs, and badges
                draw_all_feature_overlays(
                    annotated_frame,
                    features_data,
                    lip_pts=lip_pts,
                    parabola_info=parabola_info,
                    show_badges=show_boxes,
                    show_boxes=show_boxes
                )

                # Draw real-time HUD panel
                if show_hud:
                    draw_hud_panel(
                        annotated_frame,
                        n,
                        fps_smooth,
                        chunk_idx - 1,
                        features_data,
                        parabola_info=parabola_info,
                        is_live=is_live
                    )

                # Crop lip region for 5s archive
                if features_data is not None:
                    lbx, lby, lbw, lbh = features_data["box"]
                    cbx1, cby1 = max(0, lbx), max(0, lby)
                    cbx2, cby2 = min(width, lbx + lbw), min(height, lby + lbh)
                    if cbx2 > cbx1 and cby2 > cby1:
                        crop_roi = frame[cby1:cby2, cbx1:cbx2]
                        resized_crop = cv2.resize(crop_roi, crop_size)
                    else:
                        resized_crop = np.zeros((crop_size[1], crop_size[0], 3), dtype=np.uint8)
                else:
                    resized_crop = np.zeros((crop_size[1], crop_size[0], 3), dtype=np.uint8)

                # Write outputs
                out_main.write(annotated_frame)
                if chunk_writer is not None:
                    chunk_writer.write(annotated_frame)
                if crop_writer is not None:
                    crop_writer.write(resized_crop)

                # Write CSV log entry
                if features_data is not None:
                    fbx, fby, fbw, fbh = features_data["face_box"]
                    fcx, fcy = features_data["face_center"]
                    lear_x, lear_y = features_data["left_ear"]
                    rear_x, rear_y = features_data["right_ear"]
                    lear_bx, lear_by, lear_bw, lear_bh = features_data["left_ear_box"]
                    rear_bx, rear_by, rear_bw, rear_bh = features_data["right_ear_box"]
                    ear_span = features_data.get("ear_span", 0.0) if features_data else 0.0

                    lex, ley = features_data["left_eye"]
                    rex, rey = features_data["right_eye"]
                    le_bx, le_by, le_bw, le_bh = features_data["left_eye_box"]
                    re_bx, re_by, re_bw, re_bh = features_data["right_eye_box"]
                    eye_d = features_data["eye_distance"]

                    nx, ny = features_data["nose"]
                    nbx, nby, nbw, nbh = features_data["nose_box"]

                    llx, lly = features_data["left_corner"]
                    rlx, rly = features_data["right_corner"]
                    lcx, lcy = features_data["center"]
                    lbx, lby, lbw, lbh = features_data["box"]
                    w_px = features_data["width"]

                    if parabola_info is not None:
                        p_a = round(parabola_info["a"], 6)
                        p_b = round(parabola_info["b"], 4)
                        p_c = round(parabola_info["c"], 2)
                        p_ax = round(parabola_info["apex_x"], 1)
                        p_ay = round(parabola_info["apex_y"], 1)
                    else:
                        p_a = p_b = p_c = p_ax = p_ay = ""
                else:
                    fbx = fby = fbw = fbh = fcx = fcy = ""
                    lear_x = lear_y = lear_bw = lear_bh = rear_x = rear_y = rear_bw = rear_bh = ear_span = ""
                    lex = ley = le_bw = le_bh = rex = rey = re_bw = re_bh = eye_d = ""
                    nx = ny = nbw = nbh = ""
                    llx = lly = rlx = rly = lcx = lcy = lbx = lby = lbw = lbh = w_px = ""
                    p_a = p_b = p_c = p_ax = p_ay = ""

                csv_writer.writerow([
                    n, chunk_idx - 1, round(n / fps, 3),
                    fbx, fby, fbw, fbh, fcx, fcy,
                    lear_x, lear_y, lear_bw, lear_bh,
                    rear_x, rear_y, rear_bw, rear_bh, ear_span,
                    lex, ley, le_bw, le_bh,
                    rex, rey, re_bw, re_bh, eye_d,
                    nx, ny, nbw, nbh,
                    llx, lly, rlx, rly, lcx, lcy,
                    lbx, lby, lbw, lbh, w_px, active_pts_count,
                    p_a, p_b, p_c, p_ax, p_ay
                ])

                # Live preview window
                if args.show:
                    win_title = "Facial Studio: Face, Ears, Eyes, Nose & Lips [Live Tracking]"
                    cv2.imshow(win_title, annotated_frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (27, ord('q'), ord('Q')):
                        print("\nStopping tracking session...")
                        break
                    elif key in (ord('h'), ord('H')):
                        show_hud = not show_hud
                    elif key in (ord('b'), ord('B')):
                        show_boxes = not show_boxes
                    elif key in (ord('r'), ord('R')):
                        features_data = None
                        lip_pts = None
                    elif key in (ord('s'), ord('S')):
                        snap_name = f"snapshot_frame_{n:05d}.jpg"
                        cv2.imwrite(snap_name, annotated_frame)
                        print(f"Snapshot saved: '{snap_name}'")

                prev_gray = gray
                n += 1

        except KeyboardInterrupt:
            print("\nSession interrupted by user.")

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
    print(f" TRACKING COMPLETE!")
    print(f" Processed Frames    : {n}")
    print(f" Main Annotated Video: '{os.path.abspath(args.output)}'")
    print(f" 5s Chunks Directory : '{os.path.abspath(args.chunks_dir)}'")
    print(f" Cropped Lips Chunks : '{os.path.abspath(args.crop_dir)}'")
    print(f" Shutil Zip Archive  : '{os.path.abspath(zip_archive_path)}'")
    print(f" Tracking Data CSV   : '{os.path.abspath(args.csv)}'")
    print(f"========================================================\n")


if __name__ == "__main__":
    main()
