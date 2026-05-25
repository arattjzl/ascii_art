import argparse
import sys
import time

import cv2
import numpy as np

from modes.ascii_mode import create_segmenter as create_ascii_segmenter
from modes.ascii_mode import mode_status as ascii_mode_status
from modes.ascii_mode import process_ascii_frame, run_ascii_mode
from modes.landmarks68_mode import create_landmark_tools
from modes.landmarks68_mode import mode_status as landmarks68_mode_status
from modes.landmarks68_mode import process_landmarks68_frame, run_landmarks68_mode
from modes.mandala_body_mode import create_segmenter as create_mandala_segmenter
from modes.mandala_body_mode import mode_status as mandala_mode_status
from modes.mandala_body_mode import process_mandala_body_frame, run_mandala_body_mode

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


def mode_statuses():
    return [ascii_mode_status(), landmarks68_mode_status(), mandala_mode_status()]


if QApplication is not None:
    class ModeSelectorDialog(QDialog):
        def __init__(self, initial_mode=None):
            super().__init__()
            self.selected_mode = initial_mode
            self.cap = None
            self.ascii_segmenter = None
            self.mandala_segmenter = None
            self.detector = None
            self.predictor = None
            self.mode_buttons = {}
            self.zoom_strength = 1.0
            self.intensity = 1.0
            self.spread = 1.0
            self.speed = 1.0
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

            self.zoom_controls = self._build_slider_group(right, "Zoom", 0, 200, 100, "1.0x", self._update_zoom_strength)
            self.mandala_intensity_controls = self._build_slider_group(right, "Intensity", 30, 220, 100, "1.0x", self._update_intensity)
            self.mandala_spread_controls = self._build_slider_group(right, "Spread", 40, 220, 100, "1.0x", self._update_spread)
            self.mandala_speed_controls = self._build_slider_group(right, "Speed", 0, 250, 100, "1.0x", self._update_speed)

            right.addStretch()

            close_button = QPushButton("Close")
            close_button.setObjectName("close")
            close_button.clicked.connect(self.close)
            right.addWidget(close_button)

            self.timer = self.startTimer(33)
            self._open_preview_camera()
            self._update_control_visibility()

        def _build_slider_group(self, parent_layout, title, minimum, maximum, value, value_text, callback):
            controls = QWidget()
            control_layout = QVBoxLayout(controls)
            control_layout.setContentsMargins(0, 4, 0, 0)
            control_layout.setSpacing(6)

            hint = QLabel(title)
            hint.setObjectName("hint")
            control_layout.addWidget(hint)

            slider = QSlider(Qt.Horizontal)
            slider.setMinimum(minimum)
            slider.setMaximum(maximum)
            slider.setValue(value)
            slider.valueChanged.connect(callback)
            control_layout.addWidget(slider)

            label = QLabel(value_text)
            label.setObjectName("hint")
            control_layout.addWidget(label)

            parent_layout.addWidget(controls)
            controls.slider = slider
            controls.value_label = label
            return controls

        def _select_mode(self, mode):
            self.selected_mode = mode
            for button_mode, button in self.mode_buttons.items():
                button.setChecked(button_mode == mode)
                button.setProperty("active", "true" if button_mode == mode else "false")
                button.style().unpolish(button)
                button.style().polish(button)
            self._update_control_visibility()
            self._ensure_mode_resources()

        def _update_control_visibility(self):
            self.zoom_controls.setVisible(self.selected_mode == "landmarks68")
            mandala_visible = self.selected_mode == "mandala_body"
            self.mandala_intensity_controls.setVisible(mandala_visible)
            self.mandala_spread_controls.setVisible(mandala_visible)
            self.mandala_speed_controls.setVisible(mandala_visible)

        def _update_zoom_strength(self, value):
            self.zoom_strength = value / 100.0
            self.zoom_controls.value_label.setText(f"{self.zoom_strength:.1f}x")

        def _update_intensity(self, value):
            self.intensity = value / 100.0
            self.mandala_intensity_controls.value_label.setText(f"{self.intensity:.1f}x")

        def _update_spread(self, value):
            self.spread = value / 100.0
            self.mandala_spread_controls.value_label.setText(f"{self.spread:.1f}x")

        def _update_speed(self, value):
            self.speed = value / 100.0
            self.mandala_speed_controls.value_label.setText(f"{self.speed:.1f}x")

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
            if self.selected_mode == "ascii" and self.ascii_segmenter is None:
                self.ascii_segmenter = create_ascii_segmenter()
            if self.selected_mode == "landmarks68" and self.predictor is None:
                self.detector, self.predictor = create_landmark_tools()
            if self.selected_mode == "mandala_body" and self.mandala_segmenter is None:
                self.mandala_segmenter = create_mandala_segmenter()

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
                    frame = process_ascii_frame(frame, self.ascii_segmenter, timestamp_ms)
                elif self.selected_mode == "landmarks68":
                    frame = process_landmarks68_frame(frame, self.detector, self.predictor, self.zoom_strength)
                elif self.selected_mode == "mandala_body":
                    frame = process_mandala_body_frame(
                        frame,
                        self.mandala_segmenter,
                        timestamp_ms,
                        intensity=self.intensity,
                        spread=self.spread,
                        speed=self.speed,
                    )

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
            if self.ascii_segmenter is not None:
                self.ascii_segmenter.close()
            if self.mandala_segmenter is not None:
                self.mandala_segmenter.close()
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
        {"mode": "ascii", "rect": (20, 130, 280, 300), "hotkey": "1", "label": "ASCII Person"},
        {"mode": "landmarks68", "rect": (310, 130, 570, 300), "hotkey": "2", "label": "Face 68 + Zoom"},
        {"mode": "mandala_body", "rect": (600, 130, 860, 300), "hotkey": "3", "label": "Body Mandala"},
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
            cv2.putText(canvas, btn["label"], (x1 + 22, 265), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (220, 220, 220), 2)

        cv2.putText(
            canvas,
            "Click an option or press 1/2/3. Press q to quit.",
            (150, 365),
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
        if key == ord("3"):
            cv2.destroyWindow(win_name)
            return "mandala_body"
        if key == ord("q"):
            cv2.destroyWindow(win_name)
            return None


def parse_args():
    parser = argparse.ArgumentParser(description="ASCII Art, face landmarks zoom, and body mandala visualizer")
    parser.add_argument(
        "--mode",
        choices=["ascii", "landmarks68", "mandala_body"],
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
            run_ascii_mode(cap)
        elif mode == "landmarks68":
            run_landmarks68_mode(cap)
        elif mode == "mandala_body":
            run_mandala_body_mode(cap)
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
