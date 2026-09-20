"""
backend/model_service.py
Loads the trained MNIST Keras model once at server startup and provides
fast, real-time inference returning prediction, confidence, and class probabilities.
"""
import os
import numpy as np
import tensorflow as tf

class ModelService:
    def __init__(self, model_path: str = "model/mnist_model.h5"):
        self.model_path = model_path
        self.model = None
        self.load_model()

    def load_model(self):
        """Loads the trained model once from disk."""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"Model file not found at {self.model_path}. "
                "Please run `python train_model.py` first."
            )
        print(f"Loading trained MNIST model from {self.model_path}...")
        self.model = tf.keras.models.load_model(self.model_path)
        print("MNIST model loaded successfully!")

    def predict(self, input_tensor: np.ndarray) -> dict:
        """
        Runs real inference on a preprocessed (1, 28, 28) image tensor.

        Returns:
            dict containing:
            - prediction: int (0-9)
            - digit: int (0-9)
            - confidence: float (0.0-1.0)
            - probabilities: list[float] (length 10, sum to 1.0)
        """
        if self.model is None:
            raise RuntimeError("Model is not loaded.")

        # Real model prediction - inference mode
        raw_probs = self.model(input_tensor, training=False).numpy()[0]
        predicted_digit = int(np.argmax(raw_probs))
        confidence = float(np.max(raw_probs))
        probabilities = [round(float(p), 2) for p in raw_probs]

        return {
            "prediction": predicted_digit,
            "digit": predicted_digit,
            "confidence": round(confidence, 4),
            "probabilities": probabilities,
        }

# Global singleton instance
_model_service = None

def get_model_service() -> ModelService:
    global _model_service
    _model_service = ModelService()
    return _model_service
