"""
Preprocessing - uses PyMuPDF instead of poppler.
"""
from pathlib import Path
from typing import Union

from PIL import Image, ImageEnhance
from loguru import logger

import config


def preprocess_for_gemma(image_input: Union[str, Path, Image.Image]) -> Image.Image:
    if isinstance(image_input, (str, Path)):
        image = Image.open(str(image_input))
    else:
        image = image_input

    gray = image.convert('L')

    if gray.width > config.GEMMA_MAX_WIDTH:
        ratio = config.GEMMA_MAX_WIDTH / float(gray.width)
        new_h = int(gray.height * ratio)
        gray = gray.resize((config.GEMMA_MAX_WIDTH, new_h), Image.LANCZOS)

    enhancer = ImageEnhance.Contrast(gray)
    gray = enhancer.enhance(config.GEMMA_CONTRAST)

    return gray


def preprocess_for_qwen(image_input: Union[str, Path, Image.Image]) -> Image.Image:
    if isinstance(image_input, (str, Path)):
        image = Image.open(str(image_input)).convert("RGB")
    elif isinstance(image_input, Image.Image):
        image = image_input.convert("RGB")
    else:
        raise ValueError(f"Unsupported: {type(image_input)}")

    w, h = image.size

    if max(w, h) < config.QWEN_MIN_SIZE:
        image = image.resize((w * 2, h * 2), Image.LANCZOS)

    if max(image.size) > config.QWEN_MAX_SIZE:
        ratio = config.QWEN_MAX_SIZE / max(image.size)
        image = image.resize(
            (int(image.size[0] * ratio), int(image.size[1] * ratio)),
            Image.LANCZOS,
        )

    return image


def get_first_page_image(file_path: Union[str, Path]) -> Image.Image:
    """
    Extract first page from PDF using PyMuPDF (no poppler needed!).
    Or load image directly.
    """
    file_path = Path(file_path)
    suffix = file_path.suffix.lower()

    if suffix in config.SUPPORTED_IMAGE_FORMATS:
        logger.info(f"Image: {file_path.name}")
        return Image.open(str(file_path))

    elif suffix == config.SUPPORTED_PDF_FORMAT:
        try:
            import fitz  # PyMuPDF

            doc = fitz.open(str(file_path))
            page = doc[0]  # First page only

            # Render at high DPI
            zoom = config.PDF_DPI / 72  # 72 is default DPI
            matrix = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=matrix)

            # Convert to PIL Image
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

            doc.close()

            logger.info(f"PDF first page: {file_path.name} | Size: {img.size}")
            return img

        except ImportError:
            logger.error("PyMuPDF not installed!")
            logger.error("Fix: pip install PyMuPDF")
            raise

    else:
        raise ValueError(f"Unsupported: {suffix}")