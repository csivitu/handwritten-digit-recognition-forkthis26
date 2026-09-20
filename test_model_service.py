"""Request independence tests without loading TensorFlow or model weights."""

import unittest
from unittest.mock import patch

import numpy as np

from backend.model_service import ModelService


class StatefulModel:
    """A test model whose training path retains information from prior calls."""

    def __init__(self):
        self.state = 0

    def __call__(self, tensor, training):
        digit = int(tensor[0, 0, 0])
        if training:
            self.state = (self.state + digit) % 10
            digit = self.state
        probabilities = np.zeros((1, 10), dtype=np.float32)
        probabilities[0, digit] = 1

        class Output:
            def numpy(self):
                return probabilities

        return Output()


class PredictionIndependenceTests(unittest.TestCase):
    def test_intervening_prediction_does_not_change_result_or_model_state(self):
        with patch.object(ModelService, "_load_model"):
            service = ModelService()
        service.model = StatefulModel()
        first = np.full((1, 28, 28), 3, dtype=np.float32)
        second = np.full((1, 28, 28), 7, dtype=np.float32)

        original = service.predict(first)
        different = service.predict(second)
        repeated = service.predict(first)

        self.assertEqual(original["prediction"], 3)
        self.assertEqual(different["prediction"], 7)
        self.assertEqual(original, repeated)
        self.assertEqual(service.model.state, 0)
        # Response data also belongs to each individual call.
        different["probabilities"][3] = -1
        self.assertEqual(original["probabilities"][3], 1)
        np.testing.assert_array_equal(first, np.full((1, 28, 28), 3))


if __name__ == "__main__":
    unittest.main()
