"""
backend/app.py
FastAPI backend service for MNIST Handwritten Digit Recognition.
Handles both canvas drawings and uploaded images via POST /predict.
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from backend.preprocessing import preprocess_image
from backend.model_service import get_model_service

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Preload the ML model on server startup
    try:
        get_model_service()
        print("Backend startup complete. Model ready for inference.")
    except Exception as e:
        print(f"Warning during model initialization: {e}")
    yield

app = FastAPI(
    title="DigitNet — Handwritten Digit Recognition API",
    description="FastAPI backend connecting frontend canvas & image upload to MNIST model.",
    version="1.0.0",
    lifespan=lifespan,
)

# Configure CORS to allow frontend communication across localhost ports
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health_check():
    """Health check endpoint to verify backend and model readiness."""
    try:
        service = get_model_service()
        model_loaded = service.model is not None
    except Exception:
        model_loaded = False

    return {
        "status": "healthy" if model_loaded else "degraded",
        "model_loaded": model_loaded,
        "message": "DigitNet ML Backend is running",
    }

@app.post("/predict")
async def predict_digit(file: UploadFile = File(...)):
    """
    Accepts an uploaded image or canvas drawing (PNG, JPG, JPEG)
    and returns the model prediction, confidence, and probabilities.
    """
    # 1. Validate file format
    allowed_types = ["image/png", "image/jpeg", "image/jpg", "application/octet-stream"]
    if file.content_type and file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail="Unsupported file format. Please upload a PNG, JPG, or JPEG image.",
        )

    # 2. Read image bytes
    try:
        image_bytes = await file.read()
        if not image_bytes:
            raise HTTPException(
                status_code=400,
                detail="Please draw a digit or upload an image first.",
            )
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Failed to read uploaded file: {str(e)}",
        )

    # 3. Preprocess image
    try:
        tensor, preview_base64 = preprocess_image(image_bytes)
    except ValueError as ve:
        raise HTTPException(
            status_code=400,
            detail=str(ve),
        )
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Error processing image: {str(e)}",
        )

    # 4. Run model inference
    try:
        service = get_model_service()
        result = service.predict(tensor)
        result["preview"] = preview_base64
        return result
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}",
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.app:app", host="0.0.0.0", port=8000, reload=True)
