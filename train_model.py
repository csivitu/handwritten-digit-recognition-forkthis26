"""
train_model.py
Trains the MNIST digit recognition model using the exact architecture,
hyperparameters, and preprocessing from Handwritten_Digit_Recognition_MNIST.ipynb,
and saves the trained model weights to model/mnist_model.h5.
"""

import os
import ssl
import numpy as np

# Handle macOS SSL certificate verification for dataset download
try:
    ssl._create_default_https_context = ssl._create_unverified_context
except AttributeError:
    pass

def train_and_save_model(model_save_path="model/mnist_model.h5"):
    from tensorflow.keras.datasets import mnist
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Flatten
    from tensorflow.keras.utils import to_categorical

    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

    print("1. Loading MNIST dataset...")
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    # Normalize image pixel values (0–255 -> 0–1)
    X_train = X_train / 255.0
    X_test = X_test / 255.0

    # One-hot encode labels (0–9)
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)

    print(f"   Training data shape: {X_train.shape}")
    print(f"   Test data shape:     {X_test.shape}")

    # Model architecture exactly matching notebook
    print("2. Building neural network architecture...")
    model = Sequential([
        Flatten(input_shape=(28, 28)),
        Dense(128, activation='relu'),
        Dense(10, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    print("3. Training model (5 epochs)...")
    model.fit(
        X_train, y_train,
        epochs=5,
        batch_size=32,
        validation_split=0.1,
        verbose=1
    )

    print("4. Evaluating model on test set...")
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    acc_percent = round(test_accuracy * 100, 2)
    print(f"   Test Loss:     {round(test_loss, 4)}")
    print(f"   Test Accuracy: {acc_percent}%")

    print(f"5. Saving model weights to {model_save_path}...")
    model.save(model_save_path)
    print("   Model saved successfully!")

    return model, acc_percent

if __name__ == "__main__":
    train_and_save_model()
