import os
import time

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

ASCII = " .:-=+*#%@"
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEGMENTER_MODEL_PATH = os.path.join(BASE_DIR, "selfie_segmenter.tflite")


def mode_status():
    missing = []
    if not os.path.exists(SEGMENTER_MODEL_PATH):
        missing.append("Missing selfie segmentation model")
    return {
        "mode": "ascii",
        "title": "ASCII Person",
        "subtitle": "MediaPipe segmentation",
        "description": "Live grayscale person isolation rendered as an animated ASCII display.",
        "ready": not missing,
        "missing": missing,
    }


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


def run_ascii_mode(cap):
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
