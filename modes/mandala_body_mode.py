import math
import os
import time

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEGMENTER_MODEL_PATH = os.path.join(BASE_DIR, "selfie_segmenter.tflite")


def mode_status():
    missing = []
    if not os.path.exists(SEGMENTER_MODEL_PATH):
        missing.append("Missing selfie segmentation model")
    return {
        "mode": "mandala_body",
        "title": "Body Mandala",
        "subtitle": "Segmentation outline",
        "description": "Detects your body outline and emits neon mandala lines with filled flower petals.",
        "ready": not missing,
        "missing": missing,
    }


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


def _body_mask(frame, segmenter, timestamp_ms, threshold=0.58):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = segmenter.segment_for_video(mp_image, timestamp_ms)
    prob = np.squeeze(result.confidence_masks[0].numpy_view())
    mask = (prob > threshold).astype(np.uint8) * 255
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


def _largest_contour(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    return max(contours, key=cv2.contourArea)


def _contour_center(contour):
    m = cv2.moments(contour)
    if m["m00"] <= 1e-6:
        x, y, w, h = cv2.boundingRect(contour)
        return x + w // 2, y + h // 2
    return int(m["m10"] / m["m00"]), int(m["m01"] / m["m00"])


def _sample_outline_points(contour, count):
    pts = contour[:, 0, :]
    if len(pts) < 4:
        return pts
    stride = max(1, len(pts) // max(1, count))
    return pts[::stride][:count]


def _blend_ring(base, center, radius, color, thickness, alpha):
    mask = np.zeros(base.shape[:2], dtype=np.uint8)
    cv2.circle(mask, center, max(1, int(radius)), 255, max(1, int(thickness)), cv2.LINE_AA)
    m = (mask.astype(np.float32) / 255.0) * float(alpha)
    if np.max(m) <= 0:
        return base
    color_arr = np.array(color, dtype=np.float32).reshape((1, 1, 3))
    out = base.astype(np.float32)
    out = out * (1.0 - m[..., None]) + color_arr * m[..., None]
    return np.clip(out, 0, 255).astype(np.uint8)


def _phase_color(age):
    age = max(0.0, min(1.0, age))
    black = np.array((0.0, 0.0, 0.0), dtype=np.float32)
    brown = np.array((60.0, 95.0, 145.0), dtype=np.float32)
    white = np.array((255.0, 255.0, 255.0), dtype=np.float32)
    if age < 0.34:
        t = age / 0.34
        color = black * (1.0 - t) + brown * t
    elif age < 0.74:
        t = (age - 0.34) / 0.40
        color = brown * (1.0 - t) + white * t
    else:
        color = white
    fade = max(0.0, (1.0 - age) / 0.26) if age > 0.74 else 1.0
    return tuple(int(x) for x in color), fade


def _phase_color_field(phase):
    phase = np.clip(phase, 0.0, 1.0).astype(np.float32)
    b = np.zeros_like(phase, dtype=np.float32)
    g = np.zeros_like(phase, dtype=np.float32)
    r = np.zeros_like(phase, dtype=np.float32)
    alpha = np.ones_like(phase, dtype=np.float32)

    early = phase < 0.34
    mid = (phase >= 0.34) & (phase < 0.74)
    late = phase >= 0.74

    if np.any(early):
        t = phase[early] / 0.34
        b[early] = 0.0 * (1.0 - t) + 60.0 * t
        g[early] = 0.0 * (1.0 - t) + 95.0 * t
        r[early] = 0.0 * (1.0 - t) + 145.0 * t

    if np.any(mid):
        t = (phase[mid] - 0.34) / 0.40
        b[mid] = 60.0 * (1.0 - t) + 255.0 * t
        g[mid] = 95.0 * (1.0 - t) + 255.0 * t
        r[mid] = 145.0 * (1.0 - t) + 255.0 * t

    if np.any(late):
        b[late] = 255.0
        g[late] = 255.0
        r[late] = 255.0
        alpha[late] = np.maximum(0.0, (1.0 - phase[late]) / 0.26)

    return np.stack([b, g, r], axis=-1), alpha


def _solid_phase_color_field(phase):
    phase = np.clip(phase, 0.0, 1.0).astype(np.float32)
    colors = np.zeros((phase.shape[0], phase.shape[1], 3), dtype=np.float32)
    alpha = np.ones_like(phase, dtype=np.float32)

    black_band = phase < 0.34
    brown_band = (phase >= 0.34) & (phase < 0.74)
    white_band = phase >= 0.74

    colors[black_band] = (0.0, 0.0, 0.0)
    colors[brown_band] = (60.0, 95.0, 145.0)
    colors[white_band] = (255.0, 255.0, 255.0)
    alpha[white_band] = np.maximum(0.0, (1.0 - phase[white_band]) / 0.26)
    return colors, alpha


def process_mandala_body_frame(frame, segmenter, timestamp_ms, intensity=1.0, spread=1.0, speed=1.0):
    h, w = frame.shape[:2]
    mask = _body_mask(frame, segmenter, timestamp_ms)
    contour = _largest_contour(mask)
    if contour is None or cv2.contourArea(contour) < 1200:
        return frame

    cx, cy = _contour_center(contour)
    output = frame.copy()

    area = cv2.contourArea(contour)
    body_radius = math.sqrt(max(area, 1.0) / math.pi)
    aura_width = int(max(24, body_radius * (0.22 + 0.42 * spread)))

    mask_inv = cv2.bitwise_not(mask)
    dist_out = cv2.distanceTransform(mask_inv, cv2.DIST_L2, 5)
    aura_region = (dist_out > 0.5) & (dist_out <= aura_width)

    now = time.time()
    pulse_cycles = 0.16 + (0.42 * speed)
    cycle = (now * pulse_cycles) % 1.0
    phase = ((dist_out / max(1.0, float(aura_width))) - cycle) % 1.0

    colors, fade_alpha = _solid_phase_color_field(phase)
    strength = (1.0 - (dist_out / max(1.0, float(aura_width))))
    strength = np.clip(strength, 0.0, 1.0)
    strength = strength * (0.62 + 0.30 * intensity)
    alpha = strength * fade_alpha * aura_region.astype(np.float32)

    if np.any(aura_region):
        solid_region = aura_region & (alpha >= 0.55)
        fade_region = aura_region & (alpha < 0.55)
        output[solid_region] = colors[solid_region].astype(np.uint8)
        if np.any(fade_region):
            a = alpha[fade_region].reshape(-1, 1).astype(np.float32)
            src = output[fade_region].astype(np.float32)
            col = colors[fade_region].astype(np.float32)
            output[fade_region] = np.clip(src * (1.0 - a) + col * a, 0, 255).astype(np.uint8)

    cv2.drawContours(output, [contour], -1, (255, 255, 255), 1, cv2.LINE_AA)
    return output


def run_mandala_body_mode(cap):
    segmenter = create_segmenter()
    cv2.namedWindow("Body Mandala", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Body Mandala", 1200, 720)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            out = process_mandala_body_frame(frame, segmenter, int(time.time() * 1000), 1.0, 1.0, 1.0)
            cv2.imshow("Body Mandala", out)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
    finally:
        segmenter.close()

    cv2.destroyWindow("Body Mandala")
