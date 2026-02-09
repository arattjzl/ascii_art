import os, time
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

ASCII = " .:-=+*#%@"

def frame_to_ascii_lines_block(gray, mask, block_x=8, block_y=12):
    mask = np.squeeze(mask).astype(bool)          # (H,W)
    gray = np.where(mask, gray, 0).astype(np.uint8)  # persona se queda, fondo negro

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

    canvas = np.zeros((img_h, img_w, 3), dtype=np.uint8)  # fondo negro

    y = pad + ch
    for line in lines:
        cv2.putText(canvas, line, (pad, y), font, font_scale, (255, 255, 255),
                    thickness, cv2.LINE_8)
        y += line_h
    return canvas

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "selfie_segmenter.tflite")

base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.ImageSegmenterOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    output_confidence_masks=True,   # <-- IMPORTANTE
)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Can't open camera")

cv2.namedWindow("ASCII Person", cv2.WINDOW_NORMAL)
cv2.resizeWindow("ASCII Person", 900, 700)

block_x, block_y = 8, 12
threshold = 0.6  # baja a 0.5 si te recorta, sube a 0.7 si entra fondo

with vision.ImageSegmenter.create_from_options(options) as segmenter:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)  # mejora contraste para ASCII

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        timestamp_ms = int(time.time() * 1000)
        result = segmenter.segment_for_video(mp_image, timestamp_ms)

        # confidence_masks: lista de masks (una por clase)
        # En Selfie Segmentation normalmente:
        #  - 0 = background
        #  - 1 = person
        conf = result.confidence_masks

        # toma máscara de PERSONA (índice 1)
        person_prob = conf[0].numpy_view()
        person_prob = np.squeeze(person_prob)  # (H,W)

        mask_person = person_prob > threshold

        lines = frame_to_ascii_lines_block(gray, mask_person, block_x=block_x, block_y=block_y)
        ascii_img = ascii_lines_to_image(lines, font_scale=0.7, thickness=1)

        cv2.imshow("ASCII Person", ascii_img)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()
