"""
backend/preprocessing.py
Robust image preprocessing pipeline for handwritten digits.
Converts arbitrary canvas drawings and uploaded images (PNG/JPG/JPEG)
into the exact 28x28 normalized tensor expected by the MNIST model.
"""

import io
import base64
import numpy as np
from PIL import Image, ImageOps

def preprocess_image(image_bytes: bytes) -> tuple[np.ndarray, str]:
    """
    Preprocesses raw image bytes into an MNIST-compatible tensor.

    Steps:
    1. Read image and handle transparency (composite over white background).
    2. Convert to Grayscale ('L').
    3. Auto-detect background polarity (invert if light background).
    4. Validate that the image contains handwritten strokes (not blank).
    5. Crop tightly to digit bounding box.
    6. Scale bounding box to fit inside a 20x20 box preserving aspect ratio.
    7. Center the digit inside a 28x28 black canvas using center-of-mass.
    8. Normalize pixel values to [0.0, 1.0].
    9. Reshape to (1, 28, 28) float32 tensor.

    Returns:
        tuple of:
        - tensor: np.ndarray of shape (1, 28, 28) with dtype float32
        - preview_base64: data URL string of the 28x28 preprocessed image
    """
    try:
        pil_img = Image.open(io.BytesIO(image_bytes))
    except Exception as e:
        raise ValueError("Invalid or corrupted image file.") from e

    # 1. Handle transparency if RGBA / LA / P
    if pil_img.mode in ("RGBA", "LA") or (pil_img.mode == "P" and "transparency" in pil_img.info):
        pil_img = pil_img.convert("RGBA")
        background = Image.new("RGBA", pil_img.size, (255, 255, 255, 255))
        background.paste(pil_img, (0, 0), pil_img)
        pil_img = background.convert("L")
    else:
        pil_img = pil_img.convert("L")

    img_arr = np.array(pil_img, dtype=np.float32)

    # 2. Auto-detect background polarity
    # Sample border pixels (5px margin) to determine whether background is light or dark
    h, w = img_arr.shape
    border_pixels = []
    margin = min(5, h // 4, w // 4)
    if margin > 0:
        border_pixels.extend(img_arr[:margin, :].ravel())
        border_pixels.extend(img_arr[-margin:, :].ravel())
        border_pixels.extend(img_arr[:, :margin].ravel())
        border_pixels.extend(img_arr[:, -margin:].ravel())
        mean_border = np.mean(border_pixels)
    else:
        mean_border = np.mean(img_arr)

    # If border is bright (> 128), invert so background is 0 (black) and digit is bright (white)
    if mean_border > 128:
        img_arr = 255.0 - img_arr

    # Noise reduction: zero out low values
    threshold = 30.0
    img_arr[img_arr < threshold] = 0.0
    img_arr[img_arr > 200.0] = 255.0

    # 3. Validate non-empty image
    active_indices = np.argwhere(img_arr > threshold)
    validation_pixels = np.argwhere(img_arr > 0)
    if len(active_indices) < 15 or len(validation_pixels) > 500:
        raise ValueError("Please draw a digit or upload an image first.")

    # 4. Crop tightly to digit bounding box
    y_min, x_min = active_indices.min(axis=0)
    y_max, x_max = active_indices.max(axis=0)

    cropped = img_arr[y_min : y_max + 1, x_min : x_max + 1]
    crop_h, crop_w = cropped.shape

    if crop_h == 0 or crop_w == 0:
        raise ValueError("Could not extract digit from image.")

    # 5. Resize to fit within a 20x20 box preserving aspect ratio
    crop_img = Image.fromarray((cropped // 32) * 32)
    if crop_w > crop_h:
        new_w = 20
        crop_w = new_w
        new_h = max(1, int(round((crop_h / crop_w) * 20.0)))
    else:
        new_h = 20
        new_w = max(1, int(round((crop_w / crop_h) * 20.0)))

    resized = crop_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
    resized_arr = np.array(resized, dtype=np.float32)

    # 6. Place on a 28x28 black canvas
    canvas = np.zeros((28, 28), dtype=np.float32)

    # Center placement
    start_y = (28 - new_h) // 2
    start_x = (28 - new_w) // 2
    canvas[start_y : start_y + new_h, start_x : start_x + new_w] = resized_arr

    # Fine-tune centering using center of mass (MNIST standard)
    total_mass = np.sum(canvas)
    if total_mass > 0:
        cy = np.sum(np.arange(28)[:, None] * canvas) / total_mass
        cx = np.sum(np.arange(28)[None, :] * canvas) / total_mass
        shift_y = int(round((new_h / 2) - cy))
        shift_x = int(round((new_w / 2) - cx))

        # Clamp shift to avoid cropping
        shift_y = max(-4, min(4, shift_y))
        shift_x = max(-4, min(4, shift_x))

        if shift_y != 0 or shift_x != 0:
            shifted = np.zeros_like(canvas)
            # Source bounds
            src_y_start = max(0, -shift_y)
            src_y_end = min(28, 28 - shift_y)
            src_x_start = max(0, -shift_x)
            src_x_end = min(28, 28 - shift_x)

            # Dest bounds
            dst_y_start = max(0, shift_y)
            dst_y_end = min(28, 28 + shift_y)
            dst_x_start = max(0, shift_x)
            dst_x_end = min(28, 28 + shift_x)

            shifted[dst_y_start:dst_y_end, dst_x_start:dst_x_end] = canvas[
                src_y_start:src_y_end, src_x_start:src_x_end
            ]
            canvas = shifted

    # 7. Normalize pixel values (0–255 -> 0–1)
    normalized = canvas / 255.0
    tensor = normalized.reshape(1, 28, 28)

    # Generate 28x28 base64 preview for debug / frontend verification
    preview_img = Image.fromarray(np.clip(canvas, 0, 255).astype(np.uint8))
    preview_buf = io.BytesIO()
    preview_img.save(preview_buf, format="PNG")
    preview_base64 = "data:image/png;base64," + base64.b64encode(preview_buf.getvalue()).decode("utf-8")

    return tensor, preview_base64
