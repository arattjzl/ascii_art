import unittest

import numpy as np

from main import parse_args
from modes import ascii_mode, landmarks68_mode, mandala_body_mode


class DummyRect:
    def __init__(self, w, h):
        self._w = w
        self._h = h

    def width(self):
        return self._w

    def height(self):
        return self._h


class ModeTests(unittest.TestCase):
    def test_ascii_status_shape(self):
        status = ascii_mode.mode_status()
        self.assertEqual(status["mode"], "ascii")
        self.assertIn("ready", status)
        self.assertIn("missing", status)

    def test_landmarks_status_shape(self):
        status = landmarks68_mode.mode_status()
        self.assertEqual(status["mode"], "landmarks68")
        self.assertIn("ready", status)
        self.assertIn("missing", status)

    def test_mandala_status_shape(self):
        status = mandala_body_mode.mode_status()
        self.assertEqual(status["mode"], "mandala_body")
        self.assertIn("ready", status)
        self.assertIn("missing", status)

    def test_ascii_block_output(self):
        gray = np.full((24, 24), 255, dtype=np.uint8)
        mask = np.ones((24, 24), dtype=np.uint8)
        lines = ascii_mode.frame_to_ascii_lines_block(gray, mask, block_x=8, block_y=12)
        self.assertEqual(len(lines), 2)
        self.assertTrue(all(len(line) == 3 for line in lines))

    def test_select_largest_face(self):
        rects = [DummyRect(10, 10), DummyRect(8, 40), DummyRect(30, 11)]
        largest = landmarks68_mode.select_largest_face(rects)
        self.assertEqual(largest.width() * largest.height(), 330)

    def test_parse_args_modes(self):
        import sys

        old_argv = sys.argv
        try:
            sys.argv = ["main.py", "--mode", "mandala_body"]
            args = parse_args()
            self.assertEqual(args.mode, "mandala_body")
        finally:
            sys.argv = old_argv


if __name__ == "__main__":
    unittest.main()
