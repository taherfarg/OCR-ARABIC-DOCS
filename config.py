"""
Dual model config.
"""
from pathlib import Path

# PATHS
BASE_DIR = Path(__file__).parent
INPUT_DIR = BASE_DIR / "input_documents"
OUTPUT_DIR = BASE_DIR / "output_results"
LOG_DIR = BASE_DIR / "logs"

INPUT_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)
LOG_DIR.mkdir(exist_ok=True)

# MODELS
CLASSIFIER_MODEL_ID = "bakrianoo/arabic-legal-documents-ocr-1.0"
CLASSIFIER_PROMPT = "Extract details to JSON."
OCR_MODEL_ID = "sherif1313/Arabic-English-handwritten-OCR-v3"
OCR_PROMPT = """ارجو استخراج النص العربي والانجليزي كاملاً من هذه الصورة من البداية الى النهاية بدون اي اختصار ودون زيادة او حذف. اقرأ كل المحتوى النصي الموجود في الصورة بما في ذلك الأرقام والتواريخ والجداول:"""

# MODE
PROCESSING_MODE = "full"

# IMAGE PREPROCESSING ← THIS IS WHAT WAS MISSING!
GEMMA_MAX_WIDTH = 1024
GEMMA_CONTRAST = 1.5
QWEN_MAX_SIZE = 1200
QWEN_MIN_SIZE = 800
PDF_DPI = 400

# GENERATION
GEMMA_MAX_TOKENS = 1024
GEMMA_REPETITION_PENALTY = 1.5
QWEN_MAX_TOKENS = 1024
QWEN_MIN_TOKENS = 50
QWEN_REPETITION_PENALTY = 1.1

# CLASSIFICATION
DOCUMENT_TYPES = {
    "payment_voucher": {
        "name_ar": "مستند صرف / سند دفع",
        "name_en": "Payment Voucher",
        "keywords": [
            "payment voucher", "ap - payment voucher", "ap-payment voucher",
            "pv no", "bc no", "pv_number",
            "net payment", "net_payment",
            "total debits", "total_debits",
            "credit amount", "debit amount",
            "supplier details", "supplier no", "supplier name", "supplier site",
            "payment details", "payment method",
            "for use of finance department",
            "for completion by submitting department",
            "invoice currency", "chart of account",
            "bank acct", "bank a/c", "bank name",
            "cheque", "check", "wire transfer", "clearance",
            "payee code", "payee",
            "authorised for payment", "authorized for payment",
            "govt accounts", "government accounts",
            "emirate of abu dhabi",
            "مستند الصرف", "مستند صرف", "سند دفع",
            "المبالغ المدينة", "المبالغ الدائنة",
            "صافي الدفعة", "بيانات المورد",
            "رقم المورد", "اسم المورد",
            "تفاصيل الدفعة", "طريقة الدفع",
            "رقم الفاتورة", "تاريخ الفاتورة",
            "اسم البنك", "رقم الحساب",
            "التوزيع الحسابي", "رقم المستفيد",
            "تاريخ الاستلام",
            "document_type", "payment",
        ],
        "weight": 1.0,
    },
    "invoice": {
        "name_ar": "فاتورة",
        "name_en": "Invoice",
        "keywords": [
            "invoice", "فاتورة", "tax invoice", "فاتورة ضريبية",
            "bill to", "ship to", "sold to",
            "unit price", "quantity", "subtotal",
            "vat", "total amount due",
            "invoice number", "invoice date",
            "due date", "payment terms",
        ],
        "weight": 1.0,
    },
    "purchase_order": {
        "name_ar": "أمر شراء",
        "name_en": "Purchase Order",
        "keywords": [
            "purchase order", "أمر شراء",
            "po number", "delivery date",
            "vendor", "order date",
            "requested by", "approved by",
        ],
        "weight": 1.0,
    },
    "contract": {
        "name_ar": "عقد",
        "name_en": "Contract / Agreement",
        "keywords": [
            "contract", "عقد", "agreement", "اتفاقية",
            "الطرف الأول", "الطرف الثاني",
            "terms and conditions",
            "effective date", "termination",
            "scope of work",
        ],
        "weight": 1.0,
    },
    "letter": {
        "name_ar": "خطاب / رسالة",
        "name_en": "Official Letter",
        "keywords": [
            "letter", "خطاب", "رسالة",
            "dear sir", "السيد المحترم",
            "to whom it may concern",
            "re:", "الموضوع:",
            "sincerely", "المكرم",
        ],
        "weight": 1.0,
    },
    "receipt": {
        "name_ar": "إيصال",
        "name_en": "Receipt",
        "keywords": [
            "receipt", "إيصال", "سند قبض",
            "received from", "استلمنا من",
            "amount received",
        ],
        "weight": 1.0,
    },
    "legal_document": {
        "name_ar": "وثيقة قانونية",
        "name_en": "Legal Document",
        "keywords": [
            "نظام", "قانون", "لائحة", "مرسوم",
            "المادة الأولى", "المادة",
            "هيئة الخبراء", "مجلس الوزراء",
            "bureau of experts",
        ],
        "weight": 1.0,
    },
    "bank_statement": {
        "name_ar": "كشف حساب بنكي",
        "name_en": "Bank Statement",
        "keywords": [
            "bank statement", "كشف حساب",
            "opening balance", "closing balance",
            "iban", "statement period",
        ],
        "weight": 1.0,
    },
    "unknown": {
        "name_ar": "غير مصنف",
        "name_en": "Unknown",
        "keywords": [],
        "weight": 0.0,
    },
}

MIN_CONFIDENCE = 15.0

# FILES
SUPPORTED_IMAGE_FORMATS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
SUPPORTED_PDF_FORMAT = ".pdf"