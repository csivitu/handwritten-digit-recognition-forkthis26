# Handwritten Digit Recognition using MNIST

This project implements a neural network to classify handwritten digits (0–9) using the MNIST dataset, connected to an interactive frontend UI (**DigitNet**) through a high-performance **FastAPI** backend.

---

## Objective
Build and deploy a system that can correctly identify handwritten digits from both interactive canvas drawings and uploaded images.

---

## Model Details
- **Architecture**: Feedforward Neural Network (Sequential)
  - **Input Layer**: 28×28 grayscale images flattened to 784-dimensional vector (`Flatten(input_shape=(28, 28))`)
  - **Hidden Layer**: Dense layer with 128 neurons and ReLU activation (`Dense(128, activation='relu')`)
  - **Output Layer**: Dense layer with 10 neurons and Softmax activation (`Dense(10, activation='softmax')`)
- **Optimizer**: Adam
- **Loss Function**: Categorical Crossentropy
- **Dataset**: MNIST (60,000 training images, 10,000 test images)
- **Trained Model Weights**: `model/mnist_model.h5`
- **Test Accuracy**: **97.83%** (matches the project notebook benchmark ~97.66% - 97.79%)

---

## System Architecture

```text
                    USER
                     │
              ┌──────┴──────┐
              │             │
              ▼             ▼
         DRAW DIGIT    UPLOAD IMAGE
          ON CANVAS     (PNG / JPG)
              │             │
              └──────┬──────┘
                     ↓
               FRONTEND (DigitNet)
                     ↓
             POST /predict API
                     ↓
              FASTAPI BACKEND
                     ↓
               PREPROCESSING
      (Auto-polarity inversion, crop,
       20x20 scale, center-of-mass)
                     ↓
             EXISTING ML MODEL
           (model/mnist_model.h5)
                     ↓
                PREDICTION
                     ↓
             JSON RESPONSE
  { prediction: 7, confidence: 0.9991,
    probabilities: [0.0, ..., 0.9991, ...] }
                     ↓
                  FRONTEND
                     ↓
            Predicted Digit: 7
```

Both canvas drawings and uploaded images use the exact same prediction endpoint (`POST /predict`) and the exact same trained neural network model.

---

## Quick Start & Running Instructions

### 1. Prerequisites & Environment Setup

Ensure Python 3.9+ is installed. Clone the repository and install dependencies:

```bash
# Optional: Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

*(Optional)* If you wish to re-train and export the model file from scratch:
```bash
python train_model.py
```

---

### 2. Start Backend Server (FastAPI)

Run the backend server using Uvicorn:

```bash
uvicorn backend.app:app --host 127.0.0.1 --port 8000 --reload
```

- **Backend API URL**: `http://localhost:8000`
- **Interactive Swagger Docs**: `http://localhost:8000/docs`
- **Health Check**: `http://localhost:8000/health`

---

### 3. Start Frontend Server

In a separate terminal window:

```bash
python3 -m http.server 3000 --directory frontend
```

---

### 4. Open Application in Browser

Open your browser and navigate to:

```text
http://localhost:3000
```

---

## API Specification

### `POST /predict`
Accepts multipart form-data with an image representation (either PNG export from canvas or uploaded PNG/JPG/JPEG file).

#### Request
- **Headers**: `Content-Type: multipart/form-data`
- **Body**: `file`: Binary image file (`image/png`, `image/jpeg`, `image/jpg`)

#### Response (`200 OK`)
```json
{
  "prediction": 7,
  "digit": 7,
  "confidence": 0.9991,
  "probabilities": [
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.9991,
    0.0008,
    0.0001
  ],
  "preview": "data:image/png;base64,..."
}
```

### `GET /health`
Returns backend service and ML model readiness:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "message": "DigitNet ML Backend is running"
}
```

---

## Automated Tests

To run the automated verification suite (covering health checks, digits 0–9 predictions, and error handling):

```bash
python test_system.py
```

---

## Technologies Used
- **Frontend**: Vanilla HTML5, CSS3, JavaScript (Canvas API, Fetch API, Flexbox/Grid)
- **Backend API**: FastAPI, Uvicorn, Python-Multipart, Starlette
- **Machine Learning**: TensorFlow / Keras, NumPy, Pillow
