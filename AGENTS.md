# AGENTS.md

## Repo at a glance
- Single-script app: `main.py` contains all runtime logic (CLI, Qt launcher, OpenCV fallback UI, and both visual modes).
- No test/lint/typecheck/CI config is present in this repo; do not assume standard project automation commands exist.

## Setup and run (verified)
- Use the local venv used by this repo: `source venv312/bin/activate`.
- Install deps with `pip install -r requirements.txt`.
- Start launcher (or direct mode via Qt if available): `python3 main.py`.
- Run a mode directly: `python3 main.py --mode ascii` or `python3 main.py --mode landmarks68`.

## Required local assets and quirks
- `selfie_segmenter.tflite` must exist at repo root for `ascii` mode.
- `shape_predictor_68_face_landmarks.dat` must exist at repo root for `landmarks68` mode.
- `shape_predictor_68_face_landmarks.dat` is gitignored (`.gitignore`) even though runtime requires it.
- `dlib` import is optional at startup, but `landmarks68` will fail at runtime without it.

## Runtime behavior agents should know
- If `PySide6` is installed, `main.py` always runs the Qt dialog path (`run_qt_visual_app`) and does not use the OpenCV selector flow.
- OpenCV window loops exit on `q`; Qt dialog closes with Escape or Close button.
