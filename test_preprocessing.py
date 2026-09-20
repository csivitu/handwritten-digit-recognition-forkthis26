"""Regression tests for grayscale information preservation."""

import io
import unittest

import numpy as np
from PIL import Image

from backend.preprocessing import preprocess_image


def preprocess_array(pixels):
    buffer = io.BytesIO()
    Image.fromarray(pixels).save(buffer, format="PNG")
    return preprocess_image(buffer.getvalue())[0]


class PreprocessingTests(unittest.TestCase):
    def test_preserves_grayscale_including_faint_and_bright_pixels(self):
        pixels = np.zeros((40, 40), dtype=np.uint8)
        # A 20x20 crop avoids resampling so intensity loss is measurable.
        levels = np.array([81, 95, 96, 127, 128, 160, 200, 201, 220, 255], dtype=np.uint8)
        pixels[10:30, 10:30] = np.tile(levels, (20, 2))
        pixels[15:25, 15] = 40
        tensor = preprocess_array(pixels)
        for value in np.append(levels, 40):
            self.assertTrue(np.any(np.isclose(tensor, 1 - value / 255.0)), value)

    def test_nearby_intensities_do_not_collapse_to_same_tensor(self):
        tensors = []
        for intensity in (96, 110, 201, 220):
            pixels = np.zeros((40, 40), dtype=np.uint8)
            pixels[10:30, 10:30] = intensity
            tensors.append(preprocess_array(pixels))
        self.assertFalse(np.array_equal(tensors[0], tensors[1]))
        self.assertFalse(np.array_equal(tensors[2], tensors[3]))

    def test_resampling_keeps_normalized_values_in_range(self):
        pixels = np.zeros((60, 60), dtype=np.uint8)
        pixels[10:50, 20:30:2] = 255
        tensor = preprocess_array(pixels)
        self.assertEqual(tensor.dtype, np.float32)
        self.assertEqual(tensor.shape, (1, 28, 28))
        self.assertTrue(np.isfinite(tensor).all())
        self.assertGreaterEqual(tensor.min(), 0)
        self.assertLessEqual(tensor.max(), 1)


if __name__ == "__main__":
    unittest.main()
