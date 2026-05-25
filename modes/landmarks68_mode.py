import os

import cv2
import numpy as np

try:
    import dlib
except ImportError:
    dlib = None

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SHAPE_PREDICTOR_PATH = os.path.join(BASE_DIR, "shape_predictor_68_face_landmarks.dat")


def mode_status():
    missing = []
    if dlib is None:
        missing.append("Missing dlib package")
    if not os.path.exists(SHAPE_PREDICTOR_PATH):
        missing.append("Missing 68-point predictor model")
    return {
        "mode": "landmarks68",
        "title": "Face 68 + Zoom",
        "subtitle": "dlib landmarks",
        "description": "Tracks the largest face and opens dedicated zoom panels for both eyes and the mouth.",
        "ready": not missing,
        "missing": missing,
    }


def get_padded_box(points, img_w, img_h, pad=12):
    x, y, w, h = cv2.boundingRect(points)
    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(img_w, x + w + pad)
    y2 = min(img_h, y + h + pad)
    return x1, y1, x2, y2


def select_largest_face(rects):
    if not rects:
        return None
    return max(rects, key=lambda r: r.width() * r.height())


def create_landmark_tools():
    if dlib is None:
        raise ImportError("dlib is not installed. Install it to use the landmarks68 mode.")
    if not os.path.exists(SHAPE_PREDICTOR_PATH):
        raise FileNotFoundError(
            "Missing shape predictor model file: "
            f"{SHAPE_PREDICTOR_PATH}\n"
            "Download shape_predictor_68_face_landmarks.dat and place it in this project root."
        )

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(SHAPE_PREDICTOR_PATH)
    return detector, predictor


def magnify_region_inplace(frame, bbox, magnification=1.85, softness=0.82):
    x1, y1, x2, y2 = bbox
    if x2 <= x1 or y2 <= y1:
        return frame

    src = frame[y1:y2, x1:x2]
    if src.size == 0:
        return frame

    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    dst_w = min(frame.shape[1], max(x2 - x1 + 12, int((x2 - x1) * magnification)))
    dst_h = min(frame.shape[0], max(y2 - y1 + 12, int((y2 - y1) * magnification)))

    dx1 = max(0, cx - dst_w // 2)
    dy1 = max(0, cy - dst_h // 2)
    dx2 = min(frame.shape[1], dx1 + dst_w)
    dy2 = min(frame.shape[0], dy1 + dst_h)
    dst_w = dx2 - dx1
    dst_h = dy2 - dy1
    if dst_w <= 0 or dst_h <= 0:
        return frame

    enlarged = cv2.resize(src, (dst_w, dst_h), interpolation=cv2.INTER_CUBIC)
    target = frame[dy1:dy2, dx1:dx2]
    blended = cv2.addWeighted(enlarged, softness, target, 1.0 - softness, 0)
    frame[dy1:dy2, dx1:dx2] = blended
    return frame


def process_landmarks68_frame(frame, detector, predictor, zoom_strength=1.0, detect_scale=0.75):
    output = frame.copy()
    if detect_scale != 1.0:
        small_frame = cv2.resize(frame, (0, 0), fx=detect_scale, fy=detect_scale, interpolation=cv2.INTER_LINEAR)
        gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
    else:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    rects = detector(gray, 0)
    face = select_largest_face(rects)
    if face is None:
        return output

    h, w = frame.shape[:2]
    shape = predictor(gray, face)
    pts = np.array([[shape.part(i).x, shape.part(i).y] for i in range(68)], dtype=np.float32)
    if detect_scale != 1.0:
        pts /= detect_scale
    pts = pts.astype(np.int32)

    left_eye_box = get_padded_box(pts[36:42], w, h, pad=10)
    right_eye_box = get_padded_box(pts[42:48], w, h, pad=10)
    mouth_box = get_padded_box(pts[48:68], w, h, pad=12)

    eye_zoom = 1.0 + (1.9 - 1.0) * zoom_strength
    mouth_zoom = 1.0 + (1.45 - 1.0) * zoom_strength

    output = magnify_region_inplace(output, left_eye_box, magnification=eye_zoom, softness=0.9)
    output = magnify_region_inplace(output, right_eye_box, magnification=eye_zoom, softness=0.9)
    output = magnify_region_inplace(output, mouth_box, magnification=mouth_zoom, softness=0.84)
    return output


def run_landmarks68_mode(cap):
    detector, predictor = create_landmark_tools()
    cv2.namedWindow("Face 68 + Zoom", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Face 68 + Zoom", 1200, 720)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        cv2.imshow("Face 68 + Zoom", process_landmarks68_frame(frame, detector, predictor, 1.0))

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cv2.destroyWindow("Face 68 + Zoom")
