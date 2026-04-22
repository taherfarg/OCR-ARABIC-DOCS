"""
Config for pure OCR - text extraction only.
"""
import os
from pathlib import Path

# ========================
# PATHS
# ========================
BASE_DIR = Path(__file__).parent
INPUT_DIR = BASE_DIR / "input_documents"
OUTPUT_DIR = BASE_DIR / "output_results"
LOG_DIR = BASE_DIR / "logs"

INPUT_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)
LOG_DIR.mkdir(exist_ok=True)

# ========================
# MODEL
# ========================
MODEL_ID = "bakrianoo/arabic-legal-documents-ocr-1.0"
DEVICE_MAP = "auto"

# ========================
# IMAGE PREPROCESSING
# ========================
MAX_IMAGE_WIDTH = 1024
ENHANCE_CONTRAST = True
CONTRAST_FACTOR = 1.5
JPEG_QUALITY = 95

# ========================
# GENERATION
# ========================
MAX_NEW_TOKENS = 4096
DO_SAMPLE = False

# ========================
# PURE OCR PROMPT
# ========================
OCR_PROMPT = """اقرأ هذه الصورة واستخرج النص الموجود فيها كما هو بالضبط.
- حافظ على نفس ترتيب الكلمات والأسطر كما تظهر في الصورة.
- لا تضف أي تصنيف أو تحليل أو JSON.
- لا تغير أي كلمة.
- فقط انسخ النص كما هو.
- إذا كان هناك جدول، حافظ على شكله.
- أعد النص فقط بدون أي إضافات."""

OCR_PROMPT_EN = """Read this image and extract the text exactly as it appears.
- Keep the same word order and line breaks as shown in the image.
- Do NOT add any categorization, analysis, or JSON formatting.
- Do NOT change any word.
- Just copy the text as-is.
- If there is a table, preserve its structure.
- Return only the raw text with no additions."""

# ========================
# FILE FORMATS
# ========================
SUPPORTED_IMAGE_FORMATS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
SUPPORTED_PDF_FORMAT = ".pdf"
PDF_DPI = 300

# ========================
# API
# ========================
API_HOST = "0.0.0.0"
API_PORT = 8080
