"""
test_system.py
End-to-end automated testing for DigitNet backend API.
Tests /health, /predict with canvas drawing, /predict with uploaded images (PNG/JPG/JPEG),
and error handling (empty image, unsupported format, invalid data).
"""

import io
import time
import requests
import numpy as np
from PIL import Image, ImageDraw, ImageFont

BASE_URL = "http://localhost:8000"

def create_sample_digit_image(digit: int, format: str = "PNG", bg_color="white", fg_color="black", size=(300, 300)):
    """Creates a synthetic handwritten-style digit image."""
    img = Image.new("RGB", size, bg_color)
    draw = ImageDraw.Draw(img)

    cx, cy = size[0] // 2, size[1] // 2
    w, h = size[0], size[1]

    # Draw strokes matching digit
    stroke_width = 20
    if digit == 0:
        draw.ellipse([cx - 50, cy - 80, cx + 50, cy + 80], outline=fg_color, width=stroke_width)
    elif digit == 1:
        draw.line([cx, cy - 90, cx, cy + 90], fill=fg_color, width=stroke_width)
    elif digit == 3:
        draw.arc([cx - 50, cy - 80, cx + 40, cy], start=270, end=90, fill=fg_color, width=stroke_width)
        draw.arc([cx - 50, cy, cx + 40, cy + 80], start=270, end=90, fill=fg_color, width=stroke_width)
    elif digit == 7:
        draw.line([cx - 60, cy - 80, cx + 60, cy - 80], fill=fg_color, width=stroke_width)
        draw.line([cx + 60, cy - 80, cx - 30, cy + 90], fill=fg_color, width=stroke_width)
    elif digit == 8:
        draw.ellipse([cx - 40, cy - 75, cx + 40, cy - 5], outline=fg_color, width=stroke_width)
        draw.ellipse([cx - 50, cy - 5, cx + 50, cy + 80], outline=fg_color, width=stroke_width)
    else:
        # Fallback to font or line
        draw.line([cx, cy - 80, cx, cy + 80], fill=fg_color, width=stroke_width)

    buf = io.BytesIO()
    img.save(buf, format=format)
    return buf.getvalue()

def test_health():
    print("\n--- Test: GET /health ---")
    res = requests.get(f"{BASE_URL}/health")
    print(f"Status Code: {res.status_code}")
    print(f"Response: {res.json()}")
    assert res.status_code == 200
    assert res.json().get("status") == "healthy"
    assert res.json().get("model_loaded") is True
    print("PASS: Health check succeeded.")

def test_predict_digits():
    print("\n--- Test: POST /predict with Synthetic Handwritten Digits ---")
    test_cases = [
        (7, "PNG", "white", "black"),
        (1, "PNG", "white", "black"),
        (0, "JPEG", "white", "black"),
        (8, "PNG", "black", "white"), # Inverted MNIST style
        (3, "JPEG", "white", "black"),
    ]

    for digit, fmt, bg, fg in test_cases:
        img_bytes = create_sample_digit_image(digit, format=fmt, bg_color=bg, fg_color=fg)
        files = {"file": (f"digit_{digit}.{fmt.lower()}", img_bytes, f"image/{fmt.lower()}")}
        res = requests.post(f"{BASE_URL}/predict", files=files)
        assert res.status_code == 200, f"Failed for digit {digit}: {res.text}"
        data = res.json()
        pred = data.get("prediction")
        conf = data.get("confidence")
        probs = data.get("probabilities")
        print(f"Input: {digit} ({fmt}, bg={bg}) -> Predicted: {pred} | Confidence: {conf*100:.2f}%")
        assert "prediction" in data
        assert "confidence" in data
        assert len(probs) == 10
        assert 0.0 <= conf <= 1.0
        assert abs(sum(probs) - 1.0) < 0.01

    print("PASS: Predictions succeeded with valid schema.")

def test_error_handling():
    print("\n--- Test: Error Handling ---")
    
    # 1. Empty image
    empty_img = Image.new("RGB", (300, 300), "white")
    buf = io.BytesIO()
    empty_img.save(buf, format="PNG")
    res = requests.post(f"{BASE_URL}/predict", files={"file": ("empty.png", buf.getvalue(), "image/png")})
    print(f"Empty Canvas: Status {res.status_code}, Detail: {res.json().get('detail')}")
    assert res.status_code == 400

    # 2. Unsupported file type
    res = requests.post(f"{BASE_URL}/predict", files={"file": ("test.txt", b"not an image", "text/plain")})
    print(f"Unsupported Type: Status {res.status_code}, Detail: {res.json().get('detail')}")
    assert res.status_code == 400

    # 3. Corrupted image
    res = requests.post(f"{BASE_URL}/predict", files={"file": ("corrupt.png", b"corrupted bytes", "image/png")})
    print(f"Corrupted File: Status {res.status_code}, Detail: {res.json().get('detail')}")
    assert res.status_code == 400

    print("PASS: Error handling tests succeeded.")

if __name__ == "__main__":
    test_health()
    test_predict_digits()
    test_error_handling()
    print("\nALL SYSTEM TESTS PASSED SUCCESSFULLY!")
