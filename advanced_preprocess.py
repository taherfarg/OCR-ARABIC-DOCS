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
8. Binarization (adaptive thresholding)
9. Morphological cleanup
10. Resize to optimal dimensions
11. Convert to model-specific format (RGB for Qwen, grayscale for Gemma)

Improvements over v1:
- Better deskew with multiple angle detection
- Selective binarization (only when it improves clarity)
- Dilation for thin/faded text
- Border detection with smart cropping
- Multiple preprocessing profiles for multi-pass diversity
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
    Improved: tries multiple methods and picks the best angle.
    """
    if not HAS_CV2:
        return image

    img_array = np.array(image.convert('L'))

    # Method 1: Hough line detection
    edges = cv2.Canny(img_array, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180,
        threshold=100,
        minLineLength=100,
        maxLineGap=10,
    )

    angles = []
    if lines is not None and len(lines) > 0:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            # Only consider near-horizontal lines
            if abs(angle) < 10:
                angles.append(angle)

    # Method 2: MinAreaRect on text contours (fallback)
    if not angles:
        # Find text-like contours
        binary = cv2.adaptiveThreshold(
            img_array, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            blockSize=15, C=10,
        )
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours by size (text-like)
        text_contours = [c for c in contours if 100 < cv2.contourArea(c) < 50000]
        
        if len(text_contours) >= 5:
            # Get minimum area bounding rect of all text contours
            all_points = np.vstack(text_contours)
            rect = cv2.minAreaRect(all_points)
            rect_angle = rect[-1]
            
            # Adjust angle
            if rect_angle > 45:
                rect_angle = rect_angle - 90
            elif rect_angle < -45:
                rect_angle = rect_angle + 90
            
            if abs(rect_angle) < 10:
                angles.append(rect_angle)

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
    Improved: uses contour detection for smart cropping.
    """
    if not HAS_CV2:
        # Fallback: simple percentage crop
        w, h = image.size
        crop_x = int(w * border_pct)
        crop_y = int(h * border_pct)
        return image.crop((crop_x, crop_y, w - crop_x, h - crop_y))

    img_array = np.array(image.convert('L'))
    
    # Detect content area using threshold
    _, binary = cv2.threshold(img_array, 240, 255, cv2.THRESH_BINARY_INV)
    
    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        # No content detected, use simple crop
        w, h = image.size
        crop_x = int(w * border_pct)
        crop_y = int(h * border_pct)
        return image.crop((crop_x, crop_y, w - crop_x, h - crop_y))
    
    # Find the bounding box of the largest content area
    # Combine all contours
    all_points = np.vstack(contours)
    x, y, w, h = cv2.boundingRect(all_points)
    
    # Add small margin
    margin = 10
    img_w, img_h = image.size
    x = max(0, x - margin)
    y = max(0, y - margin)
    w = min(img_w - x, w + 2 * margin)
    h = min(img_h - y, h + 2 * margin)
    
    return image.crop((x, y, x + w, y + h))


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


def dilate_text(image: Image.Image, iterations: int = 1) -> Image.Image:
    """
    Dilate (thicken) thin or faded text.
    Helps OCR models read faint text more accurately.
    """
    if not HAS_CV2:
        return image

    img_array = np.array(image.convert('L'))
    _, binary = cv2.threshold(img_array, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 1))
    dilated = cv2.dilate(binary, kernel, iterations=iterations)
    
    return Image.fromarray(dilated)


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


def selective_binarize(image: Image.Image) -> Image.Image:
    """
    Apply binarization only if it improves text clarity.
    Compares histogram spread before/after to decide.
    """
    if not HAS_CV2:
        return image

    img_array = np.array(image.convert('L'))
    
    # Calculate histogram spread (bimodal = good for binarization)
    hist = cv2.calcHist([img_array], [0], None, [256], [0, 256])
    hist_norm = hist.flatten() / hist.sum()
    
    # Check if histogram is bimodal (two peaks = text + background)
    # Simple check: if there's a significant valley between peaks
    from scipy.signal import find_peaks
    try:
        peaks, _ = find_peaks(hist_norm, height=0.01, distance=20)
        if len(peaks) >= 2:
            # Bimodal — binarization will help
            _, binary = cv2.threshold(img_array, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            return Image.fromarray(binary)
    except (ImportError, Exception):
        pass
    
    # Not clearly bimodal — try Sauvola binarization (adaptive)
    # This works better for documents with varying backgrounds
    mean = np.mean(img_array)
    std = np.std(img_array)
    
    if std > 40:  # High variance suggests mixed content
        binary = cv2.adaptiveThreshold(
            img_array, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            blockSize=25,
            C=10,
        )
        return Image.fromarray(binary)
    
    return image


def remove_noise_isolated(image: Image.Image) -> Image.Image:
    """
    Remove isolated noise pixels while preserving text.
    Uses connected component analysis.
    """
    if not HAS_CV2:
        return image

    img_array = np.array(image.convert('L'))
    _, binary = cv2.threshold(img_array, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Find connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    
    # Remove very small components (noise)
    min_size = 5  # Minimum component size in pixels
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] < min_size:
            binary[labels == i] = 0
    
    # Convert back to normal (white background, black text)
    result = cv2.bitwise_not(binary)
    return Image.fromarray(result)


def full_preprocess_pipeline(
    image_input: Union[str, Path, Image.Image],
    target_dpi: int = 300,
    max_width: int = 2048,
    for_model: str = "qwen",
    extra_contrast: bool = False,
    enhance_handwriting: bool = False,
    dilate: bool = False,
    binarize: bool = False,
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
    10. Optional: dilation for thin text
    11. Optional: selective binarization
    12. Morphological cleanup
    13. Remove isolated noise
    14. Resize to optimal size
    15. Convert to appropriate format (RGB for Qwen, Grayscale for Gemma)
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

    # 10. Dilation for thin/faded text
    if dilate:
        gray = dilate_text(gray, iterations=1)

    # 11. Selective binarization
    if binarize:
        gray = selective_binarize(gray)

    # 12-13: Morphological cleanup and noise removal only when binarization was applied,
    # otherwise they destroy thin Arabic text strokes
    if binarize or enhance_handwriting:
        gray = morphological_clean(gray)
        gray = remove_noise_isolated(gray)

    # 14. Resize to optimal size
    if gray.width > max_width:
        ratio = max_width / float(gray.width)
        new_h = int(gray.height * ratio)
        gray = gray.resize((max_width, new_h), Image.LANCZOS)

    # 15. Format for model
    if for_model == "qwen":
        # Qwen needs RGB
        result = Image.merge("RGB", (gray, gray, gray))
    else:
        # Gemma works with grayscale
        result = gray

    logger.info(f"✅ Preprocessed: {result.size} ({for_model} mode)")
    return result


def light_preprocess_vlm(
    image_input: Union[str, Path, Image.Image],
    max_width: int = 1600,
    min_width: int = 1000,
) -> Image.Image:
    """
    Light preprocessing for modern VLMs (Qwen2.5-VL, etc.).

    Modern VLMs work best with natural color images. This only does:
    1. Convert to RGB
    2. Upscale small images (critical for OCR accuracy)
    3. Downscale oversized images
    4. Light sharpening
    """
    if isinstance(image_input, (str, Path)):
        image = Image.open(str(image_input))
    else:
        image = image_input

    original_size = image.size

    # 1. Ensure RGB
    image = image.convert("RGB")

    # 2. Upscale small images — VLMs need enough pixels to read text
    if image.width < min_width:
        scale = min_width / image.width
        image = image.resize(
            (int(image.width * scale), int(image.height * scale)),
            Image.LANCZOS,
        )
        logger.info(f"Upscaled: {original_size} -> {image.size}")

    # 3. Downscale if too large
    if image.width > max_width:
        ratio = max_width / float(image.width)
        new_h = int(image.height * ratio)
        image = image.resize((max_width, new_h), Image.LANCZOS)

    # 4. Light sharpening for scanned documents
    image = sharpen_image(image, factor=1.2)

    logger.info(f"Light preprocessed: {image.size} (VLM mode)")
    return image
