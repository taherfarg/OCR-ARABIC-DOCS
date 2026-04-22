"""
Image preprocessing.
"""
import base64
from io import BytesIO
from pathlib import Path
from typing import List, Union

from PIL import Image, ImageEnhance
from loguru import logger

import config


def preprocess_image(
    image_input: Union[str, Path, Image.Image],
    max_width: int = config.MAX_IMAGE_WIDTH,
    do_enhance: bool = config.ENHANCE_CONTRAST,
    contrast_factor: float = config.CONTRAST_FACTOR,
    return_base64: bool = False
) -> Union[Image.Image, str]:

    if isinstance(image_input, (str, Path)):
        image = Image.open(str(image_input))
        logger.info(f"Loaded: {image_input} | Size: {image.size}")
    elif isinstance(image_input, Image.Image):
        image = image_input
    else:
        raise ValueError(f"Unsupported type: {type(image_input)}")

    gray_image = image.convert('L')

    if gray_image.width > max_width:
        ratio = max_width / float(gray_image.width)
        new_height = int(gray_image.height * ratio)
        gray_image = gray_image.resize((max_width, new_height), Image.LANCZOS)

    if do_enhance:
        enhancer = ImageEnhance.Contrast(gray_image)
        gray_image = enhancer.enhance(contrast_factor)

    if return_base64:
        buffered = BytesIO()
        gray_image.save(buffered, format="JPEG", optimize=True, quality=config.JPEG_QUALITY)
        img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
        return f"data:image/jpeg;base64,{img_str}"

    return gray_image


def pdf_to_images(pdf_path: Union[str, Path], dpi: int = config.PDF_DPI) -> List[Image.Image]:
    try:
        from pdf2image import convert_from_path
        images = convert_from_path(str(pdf_path), dpi=dpi)
        logger.info(f"PDF -> {len(images)} page(s)")
        return images
    except ImportError:
        logger.error("Install: pip install pdf2image + poppler-utils")
        raise


def load_document_images(file_path: Union[str, Path]) -> List[Image.Image]:
    file_path = Path(file_path)
    suffix = file_path.suffix.lower()

    if suffix in config.SUPPORTED_IMAGE_FORMATS:
        return [Image.open(str(file_path))]
    elif suffix == config.SUPPORTED_PDF_FORMAT:
        return pdf_to_images(file_path)
    else:
        raise ValueError(f"Unsupported: {suffix}")
