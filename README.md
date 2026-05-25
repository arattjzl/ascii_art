# ascii_art

Real-time webcam visuals with a live preview launcher and three modes:

- `ascii`: person-only ASCII render using MediaPipe selfie segmentation.
- `landmarks68`: dlib 68-point face landmarks with in-place enlargement for the eyes and mouth.
- `mandala_body`: body outline detection with neon mandala rays and filled flower-like petals.

## Requirements

- Python 3.10+
- Webcam
- Packages:
  - `opencv-python`
  - `numpy`
  - `mediapipe`
  - `dlib` (only needed for `landmarks68` mode)
  - `PySide6_Essentials` for the launcher UI (`PySide6` import namespace)

## Model Files

This project expects these files in the project root:

- `selfie_segmenter.tflite` (required for `ascii` and `mandala_body`)
- `shape_predictor_68_face_landmarks.dat` (required for `landmarks68` mode)

## Run

Use the project venv:

```bash
source venv312/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Open the launcher:

```bash
python3 main.py
```

Run a mode directly:

```bash
python3 main.py --mode ascii
python3 main.py --mode landmarks68
python3 main.py --mode mandala_body
```

Press `q` to quit the active OpenCV visual. In the Qt launcher, switch modes with the buttons on the right and use the zoom control when `landmarks68` is active.

## Notes

- The Qt launcher keeps the live camera preview visible while you switch modes.
- The zoom control appears only for `landmarks68`.
- `mandala_body` shows dedicated controls for intensity, spread, and speed.
- `landmarks68` requires `shape_predictor_68_face_landmarks.dat` in the project root.
