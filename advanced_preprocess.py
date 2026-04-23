"""
Advanced image preprocessing for maximum OCR accuracy.
Applies multiple enhancement techniques tuned for Arabic/English mixed documents,
both handwritten and printed, on scanned government documents.

Pipeline:
1. Load & convert to grayscale
2. Upscale low-resolution images
3. Deskew (straighten rotated scans)
4. Denoise (remove scan artifacts)
5. CLAHE contrast enhancement (handles uneven lighting)
6. Sharpen text edges
7. Remove dark borders
8. Resize to optimal dimensions
9. Convert to model-specific format (RGB for Qwen, grayscale for Gemma)
"""
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from pathlib import Path
from typing import Union, Optional
from loguru import logger

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False
    logger.warning("opencv not installed. Install: pip install opencv-python-headless")


def adaptive_threshold(image: Image.Image) -> Image.Image:
    """
    Apply adaptive thresholding for better text/background separation.
    Critical for scanned documents with uneven lighting.
    """
    if not HAS_CV2:
        return image

    img_array = np.array(image.convert('L'))

    # Adaptive threshold - handles uneven lighting
    binary = cv2.adaptiveThreshold(
        img_array, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=15,    # Neighborhood size
        C=8,             # Constant subtracted from mean
    )

    return Image.fromarray(binary)


def deskew_image(image: Image.Image) -> Image.Image:
    """
    Detect and correct document skew (rotation).
    Even 1-2 degrees of skew significantly hurts OCR.
    """
    if not HAS_CV2:
        return image

    img_array = np.array(image.convert('L'))

    # Edge detection
    edges = cv2.Canny(img_array, 50, 150, apertureSize=3)

    # Detect lines
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180,
        threshold=100,
        minLineLength=100,
        maxLineGap=10,
    )

    if lines is None or len(lines) == 0:
        return image

    # Calculate average angle
    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        # Only consider near-horizontal lines
        if abs(angle) < 10:
            angles.append(angle)

    if not angles:
        return image

    median_angle = np.median(angles)

    if abs(median_angle) < 0.3:
        return image  # Already straight enough

    logger.info(f"📐 Deskewing by {median_angle:.2f}°")
    return image.rotate(median_angle, fillcolor=255, expand=True)


def denoise_image(image: Image.Image, strength: int = 10) -> Image.Image:
    """
    Remove noise from scanned documents.
    Uses non-local means denoising.
    """
    if not HAS_CV2:
        return image

    img_array = np.array(image.convert('L'))

    # Non-local means denoising
    denoised = cv2.fastNlMeansDenoising(
        img_array,
        h=strength,               # Filter strength (higher = more denoising)
        templateWindowSize=7,     # Template patch size
        searchWindowSize=21,      # Search area size
    )

    return Image.fromarray(denoised)


def sharpen_image(image: Image.Image, factor: float = 1.5) -> Image.Image:
    """Sharpen text edges for better recognition."""
    enhancer = ImageEnhance.Sharpness(image)
    return enhancer.enhance(factor)


def enhance_contrast_clahe(image: Image.Image, clip_limit: float = 2.0) -> Image.Image:
    """
    CLAHE (Contrast Limited Adaptive Histogram Equalization).
    Much better than simple contrast enhancement.
    Handles documents with varying brightness across the page.
    """
    if not HAS_CV2:
        enhancer = ImageEnhance.Contrast(image)
        return enhancer.enhance(1.5)

    img_array = np.array(image.convert('L'))

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    enhanced = clahe.apply(img_array)

    return Image.fromarray(enhanced)


def remove_borders(image: Image.Image, border_pct: float = 0.02) -> Image.Image:
    """
    Remove dark borders/edges from scanned documents.
    These confuse OCR models.
    """
    w, h = image.size
    crop_x = int(w * border_pct)
    crop_y = int(h * border_pct)

    cropped = image.crop((crop_x, crop_y, w - crop_x, h - crop_y))
    return cropped


def morphological_clean(image: Image.Image) -> Image.Image:
    """
    Morphological operations to clean up text:
    - Close small gaps in characters
    - Remove tiny noise dots
    Particularly helpful for handwritten text.
    """
    if not HAS_CV2:
        return image

    img_array = np.array(image.convert('L'))

    # Ensure binary
    _, binary = cv2.threshold(img_array, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Small kernel to close gaps in characters without merging lines
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))

    # Close: fills small gaps in character strokes
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    # Open: removes small noise dots
    kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel_open)

    return Image.fromarray(cleaned)


def enhance_for_handwriting(image: Image.Image) -> Image.Image:
    """
    Special enhancement pass for handwritten text regions.
    Increases contrast and sharpness more aggressively.
    """
    if not HAS_CV2:
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(2.0)
        enhancer = ImageEnhance.Sharpness(image)
        return enhancer.enhance(2.0)

    img_array = np.array(image.convert('L'))

    # Stronger CLAHE for handwriting
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(img_array)

    # Adaptive threshold to isolate handwriting
    binary = cv2.adaptiveThreshold(
        enhanced, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=11,
        C=5,
    )

    return Image.fromarray(binary)


def full_preprocess_pipeline(
    image_input: Union[str, Path, Image.Image],
    target_dpi: int = 300,
    max_width: int = 2048,
    for_model: str = "qwen",
    extra_contrast: bool = False,
    enhance_handwriting: bool = False,
) -> Image.Image:
    """
    Full preprocessing pipeline for maximum OCR accuracy.

    Pipeline:
    1. Load image
    2. Upscale if low resolution
    3. Deskew (straighten)
    4. Denoise
    5. CLAHE contrast enhancement
    6. Sharpen
    7. Remove borders
    8. Optional: extra contrast pass
    9. Optional: handwriting enhancement
    10. Resize to optimal size
    11. Convert to appropriate format (RGB for Qwen, Grayscale for Gemma)
    """
    # Load
    if isinstance(image_input, (str, Path)):
        image = Image.open(str(image_input))
    else:
        image = image_input

    original_size = image.size
    logger.info(f"🖼️ Original: {original_size}")

    # 1. Convert to grayscale for processing
    gray = image.convert('L')

    # 2. Upscale small images (critical for accuracy!)
    if max(gray.size) < 1500:
        scale = 2
        gray = gray.resize(
            (gray.width * scale, gray.height * scale),
            Image.LANCZOS,
        )
        logger.debug(f"Upscaled: {original_size} → {gray.size}")

    # 3. Deskew
    gray = deskew_image(gray)

    # 4. Denoise — lighter for printed text, heavier for handwritten
    denoise_strength = 15 if enhance_handwriting else 10
    gray = denoise_image(gray, strength=denoise_strength)

    # 5. CLAHE contrast
    clip_limit = 3.0 if extra_contrast else 2.0
    gray = enhance_contrast_clahe(gray, clip_limit=clip_limit)

    # 6. Sharpen
    sharpen_factor = 1.5 if enhance_handwriting else 1.3
    gray = sharpen_image(gray, factor=sharpen_factor)

    # 7. Remove borders
    gray = remove_borders(gray, border_pct=0.01)

    # 8. Extra contrast pass (for detail-focused OCR pass)
    if extra_contrast:
        gray = enhance_contrast_clahe(gray, clip_limit=3.0)
        gray = sharpen_image(gray, factor=1.5)

    # 9. Handwriting enhancement
    if enhance_handwriting:
        gray = enhance_for_handwriting(gray)

    # 10. Morphological cleanup
    gray = morphological_clean(gray)

    # 11. Resize to optimal size
    if gray.width > max_width:
        ratio = max_width / float(gray.width)
        new_h = int(gray.height * ratio)
        gray = gray.resize((max_width, new_h), Image.LANCZOS)

    # 12. Format for model
    if for_model == "qwen":
        # Qwen needs RGB
        result = Image.merge("RGB", (gray, gray, gray))
    else:
        # Gemma works with grayscale
        result = gray

    logger.info(f"✅ Preprocessed: {result.size} ({for_model} mode)")
    return result
