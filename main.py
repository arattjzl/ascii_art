import argparse
import os
import sys
import time

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

try:
    import dlib
except ImportError:
    dlib = None

try:
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QImage, QPixmap
    from PySide6.QtWidgets import (
        QApplication,
        QDialog,
        QHBoxLayout,
        QLabel,
        QPushButton,
        QSlider,
        QVBoxLayout,
        QWidget,
    )
except ImportError:
    QApplication = None

ASCII = " .:-=+*#%@"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SEGMENTER_MODEL_PATH = os.path.join(BASE_DIR, "selfie_segmenter.tflite")
SHAPE_PREDICTOR_PATH = os.path.join(BASE_DIR, "shape_predictor_68_face_landmarks.dat")


def frame_to_ascii_lines_block(gray, mask, block_x=8, block_y=12):
    mask = np.squeeze(mask).astype(bool)
    gray = np.where(mask, gray, 0).astype(np.uint8)

    h, w = gray.shape
    h2 = (h // block_y) * block_y
    w2 = (w // block_x) * block_x
    gray = gray[:h2, :w2]

    small = gray.reshape(h2 // block_y, block_y, w2 // block_x, block_x).mean(axis=(1, 3))
    idx = (small.astype(np.int32) * (len(ASCII) - 1) // 255).astype(np.int32)
    return ["".join(ASCII[i] for i in row) for row in idx]


def ascii_lines_to_image(lines, font_scale=0.7, thickness=1, pad=10):
    font = cv2.FONT_HERSHEY_PLAIN
    (cw, ch), baseline = cv2.getTextSize("A", font, font_scale, thickness)
    line_h = ch + baseline + 2

    rows = len(lines)
    cols = max((len(x) for x in lines), default=1)
    img_h = pad * 2 + rows * line_h
    img_w = pad * 2 + cols * cw

    canvas = np.zeros((img_h, img_w, 3), dtype=np.uint8)
    y = pad + ch
    for line in lines:
        cv2.putText(canvas, line, (pad, y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_8)
        y += line_h
    return canvas


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


def create_segmenter():
    if not os.path.exists(SEGMENTER_MODEL_PATH):
        raise FileNotFoundError(f"Missing model file: {SEGMENTER_MODEL_PATH}")

    base_options = python.BaseOptions(model_asset_path=SEGMENTER_MODEL_PATH)
    options = vision.ImageSegmenterOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        output_confidence_masks=True,
    )
    return vision.ImageSegmenter.create_from_options(options)


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


def process_ascii_frame(frame, segmenter, timestamp_ms, block_x=8, block_y=12, threshold=0.6):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = segmenter.segment_for_video(mp_image, timestamp_ms)

    person_prob = np.squeeze(result.confidence_masks[0].numpy_view())
    mask_person = person_prob > threshold

    lines = frame_to_ascii_lines_block(gray, mask_person, block_x=block_x, block_y=block_y)
    return ascii_lines_to_image(lines, font_scale=0.7, thickness=1)


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
        small_frame = frame
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


def mode_statuses():
    ascii_missing = []
    if not os.path.exists(SEGMENTER_MODEL_PATH):
        ascii_missing.append("Missing selfie segmentation model")

    landmarks_missing = []
    if dlib is None:
        landmarks_missing.append("Missing dlib package")
    if not os.path.exists(SHAPE_PREDICTOR_PATH):
        landmarks_missing.append("Missing 68-point predictor model")

    return [
        {
            "mode": "ascii",
            "title": "ASCII Person",
            "subtitle": "MediaPipe segmentation",
            "description": "Live grayscale person isolation rendered as an animated ASCII display.",
            "ready": not ascii_missing,
            "missing": ascii_missing,
        },
        {
            "mode": "landmarks68",
            "title": "Face 68 + Zoom",
            "subtitle": "dlib landmarks",
            "description": "Tracks the largest face and opens dedicated zoom panels for both eyes and the mouth.",
            "ready": not landmarks_missing,
            "missing": landmarks_missing,
        },
    ]


if QApplication is not None:
    class ModeSelectorDialog(QDialog):
        def __init__(self, initial_mode=None):
            super().__init__()
            self.selected_mode = initial_mode
            self.cap = None
            self.segmenter = None
            self.detector = None
            self.predictor = None
            self.mode_buttons = {}
            self.zoom_strength = 1.0
            self.setWindowTitle("ASCII Visuals")
            self.setFixedSize(920, 680)
            self.setStyleSheet(
                """
                QDialog {
                    background: #f3f4f6;
                }
                QLabel#preview {
                    background: #0b0f19;
                    border: 1px solid #d1d5db;
                    border-radius: 18px;
                }
                QLabel#hint {
                    color: #4b5563;
                    font-size: 13px;
                }
                QPushButton#mode {
                    background: white;
                    border: 1px solid #d1d5db;
                    border-radius: 12px;
                    padding: 12px 16px;
                    font-size: 14px;
                    font-weight: 700;
                    color: #0f1728;
                }
                QPushButton#mode[active="true"] {
                    background: #111827;
                    color: white;
                    border: 1px solid #111827;
                }
                QPushButton#mode[ready="false"] {
                    background: #e5e7eb;
                    color: #98a2b3;
                    border: 1px solid #d0d5dd;
                }
                QSlider::groove:horizontal {
                    height: 6px;
                    background: #d1d5db;
                    border-radius: 3px;
                }
                QSlider::handle:horizontal {
                    background: #111827;
                    width: 18px;
                    margin: -6px 0;
                    border-radius: 9px;
                }
                QPushButton#close {
                    background: white;
                    border: 1px solid #d1d5db;
                    border-radius: 12px;
                    padding: 12px 16px;
                    font-size: 14px;
                    font-weight: 700;
                    color: #0f1728;
                }
                """
            )
            self._build_ui()

        def _build_ui(self):
            root = QVBoxLayout(self)
            root.setContentsMargins(16, 16, 16, 16)

            layout = QHBoxLayout()
            layout.setSpacing(16)
            root.addLayout(layout)

            left = QVBoxLayout()
            left.setSpacing(8)
            layout.addLayout(left, stretch=7)

            self.preview = QLabel("Opening camera preview...")
            self.preview.setObjectName("preview")
            self.preview.setAlignment(Qt.AlignCenter)
            self.preview.setMinimumSize(620, 620)
            left.addWidget(self.preview, stretch=1)

            right = QVBoxLayout()
            right.setSpacing(10)
            layout.addLayout(right, stretch=2)

            for status in mode_statuses():
                button = QPushButton(status["title"])
                button.setObjectName("mode")
                button.setCheckable(True)
                button.setProperty("ready", "true" if status["ready"] else "false")
                button.setProperty("active", "true" if status["mode"] == self.selected_mode else "false")
                button.setEnabled(status["ready"])
                button.clicked.connect(lambda checked=False, mode=status["mode"]: self._select_mode(mode))
                button.style().unpolish(button)
                button.style().polish(button)
                right.addWidget(button)
                self.mode_buttons[status["mode"]] = button

            self.zoom_controls = QWidget()
            zoom_layout = QVBoxLayout(self.zoom_controls)
            zoom_layout.setContentsMargins(0, 4, 0, 0)
            zoom_layout.setSpacing(6)

            zoom_hint = QLabel("Zoom")
            zoom_hint.setObjectName("hint")
            zoom_layout.addWidget(zoom_hint)

            self.zoom_slider = QSlider(Qt.Horizontal)
            self.zoom_slider.setMinimum(0)
            self.zoom_slider.setMaximum(200)
            self.zoom_slider.setValue(100)
            self.zoom_slider.valueChanged.connect(self._update_zoom_strength)
            zoom_layout.addWidget(self.zoom_slider)

            self.zoom_value = QLabel("1.0x")
            self.zoom_value.setObjectName("hint")
            zoom_layout.addWidget(self.zoom_value)

            right.addWidget(self.zoom_controls)
            self.zoom_controls.setVisible(self.selected_mode == "landmarks68")

            right.addStretch()

            close_button = QPushButton("Close")
            close_button.setObjectName("close")
            close_button.clicked.connect(self.close)
            right.addWidget(close_button)

            self.timer = self.startTimer(33)
            self._open_preview_camera()

        def _select_mode(self, mode):
            self.selected_mode = mode
            for button_mode, button in self.mode_buttons.items():
                button.setChecked(button_mode == mode)
                button.setProperty("active", "true" if button_mode == mode else "false")
                button.style().unpolish(button)
                button.style().polish(button)
            self.zoom_controls.setVisible(mode == "landmarks68")
            self._ensure_mode_resources()

        def _update_zoom_strength(self, value):
            self.zoom_strength = value / 100.0
            self.zoom_value.setText(f"{self.zoom_strength:.1f}x")

        def keyPressEvent(self, event):
            if event.key() == Qt.Key_Escape:
                self.close()
                return
            super().keyPressEvent(event)

        def _open_preview_camera(self):
            self.cap = cv2.VideoCapture(0)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            if not self.cap.isOpened():
                self.preview.setText("Camera preview is unavailable.")
                self.cap.release()
                self.cap = None

        def _release_preview_camera(self):
            if self.cap is not None:
                self.cap.release()
                self.cap = None

        def _ensure_mode_resources(self):
            if self.selected_mode == "ascii" and self.segmenter is None:
                self.segmenter = create_segmenter()
            if self.selected_mode == "landmarks68" and self.predictor is None:
                self.detector, self.predictor = create_landmark_tools()

        def timerEvent(self, event):
            if event.timerId() != self.timer:
                super().timerEvent(event)
                return
            if self.cap is None:
                return
            ret, frame = self.cap.read()
            if not ret:
                return

            if self.selected_mode is not None:
                self._ensure_mode_resources()
                timestamp_ms = int(time.time() * 1000)
                if self.selected_mode == "ascii":
                    frame = process_ascii_frame(frame, self.segmenter, timestamp_ms)
                elif self.selected_mode == "landmarks68":
                    frame = process_landmarks68_frame(frame, self.detector, self.predictor, self.zoom_strength)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, channels = rgb.shape
            image = QImage(rgb.data, w, h, channels * w, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(image).scaled(
                self.preview.size(),
                Qt.KeepAspectRatio,
                Qt.FastTransformation,
            )
            self.preview.setPixmap(pixmap)

        def closeEvent(self, event):
            if self.segmenter is not None:
                self.segmenter.close()
            self._release_preview_camera()
            super().closeEvent(event)


def run_qt_visual_app(initial_mode=None):
    if QApplication is None:
        return False

    app = QApplication.instance()
    owns_app = app is None
    if owns_app:
        app = QApplication(sys.argv)

    dialog = ModeSelectorDialog(initial_mode=initial_mode)
    dialog.show()
    if owns_app:
        app.exec()
    return True


def show_cv_mode_selection_ui():
    win_name = "Visual Selector"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_name, 880, 420)
    selected_mode = {"value": None}
    mouse = {"x": -1, "y": -1}

    buttons = [
        {"mode": "ascii", "rect": (80, 130, 400, 300), "hotkey": "1", "label": "ASCII Person"},
        {"mode": "landmarks68", "rect": (480, 130, 800, 300), "hotkey": "2", "label": "Face 68 + Zoom"},
    ]

    def on_mouse(event, x, y, flags, param):
        del flags, param
        mouse["x"], mouse["y"] = x, y
        if event != cv2.EVENT_LBUTTONUP:
            return
        for btn in buttons:
            x1, y1, x2, y2 = btn["rect"]
            if x1 <= x <= x2 and y1 <= y <= y2:
                selected_mode["value"] = btn["mode"]
                return

    cv2.setMouseCallback(win_name, on_mouse)

    while True:
        canvas = np.zeros((420, 880, 3), dtype=np.uint8)
        cv2.rectangle(canvas, (0, 0), (879, 419), (40, 40, 40), 2)

        cv2.putText(canvas, "Choose Visual Mode", (220, 85), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (240, 240, 240), 2)

        for btn in buttons:
            x1, y1, x2, y2 = btn["rect"]
            hovering = x1 <= mouse["x"] <= x2 and y1 <= mouse["y"] <= y2
            border_color = (120, 255, 120) if hovering else (200, 200, 200)
            fill_color = (50, 50, 50) if hovering else (20, 20, 20)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), fill_color, -1)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), border_color, 2)

            center_x = (x1 + x2) // 2
            cv2.putText(canvas, btn["hotkey"], (center_x - 12, 215), cv2.FONT_HERSHEY_SIMPLEX, 2.2, (255, 255, 255), 3)
            cv2.putText(canvas, btn["label"], (x1 + 52, 265), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (220, 220, 220), 2)

        cv2.putText(
            canvas,
            "Click an option or press 1/2. Press q to quit.",
            (180, 365),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (180, 180, 180),
            2,
        )
        cv2.imshow(win_name, canvas)

        if selected_mode["value"] is not None:
            chosen = selected_mode["value"]
            cv2.destroyWindow(win_name)
            return chosen

        key = cv2.waitKey(30) & 0xFF
        if key == ord("1"):
            cv2.destroyWindow(win_name)
            return "ascii"
        if key == ord("2"):
            cv2.destroyWindow(win_name)
            return "landmarks68"
        if key == ord("q"):
            cv2.destroyWindow(win_name)
            return None


def show_mode_selection_ui():
    if QApplication is not None:
        run_qt_visual_app()
        return None
    return show_cv_mode_selection_ui()


def run_ascii_person(cap):
    segmenter = create_segmenter()
    cv2.namedWindow("ASCII Person", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("ASCII Person", 900, 700)
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            ascii_img = process_ascii_frame(frame, segmenter, int(time.time() * 1000))
            cv2.imshow("ASCII Person", ascii_img)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
    finally:
        segmenter.close()
    cv2.destroyWindow("ASCII Person")


def run_landmarks68_zoom(cap):
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


def parse_args():
    parser = argparse.ArgumentParser(description="ASCII Art and 68-landmarks face zoom visualizer")
    parser.add_argument(
        "--mode",
        choices=["ascii", "landmarks68"],
        default=None,
        help="Run a mode directly. If omitted, a simple UI selector is shown.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if QApplication is not None:
        run_qt_visual_app(initial_mode=args.mode)
        return

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Can't open camera")

    try:
        mode = args.mode or show_cv_mode_selection_ui()
        if mode is None:
            return
        if mode == "ascii":
            run_ascii_person(cap)
        elif mode == "landmarks68":
            run_landmarks68_zoom(cap)
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
