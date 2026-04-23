"""
Dual model config — improved for maximum OCR + classification accuracy.
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

# Detailed classification prompt — tells the model exactly what to extract
CLASSIFIER_PROMPT = """Analyze this document image carefully and extract the following information in JSON format:
1. "document_type": Identify the type (payment_voucher, invoice, purchase_order, contract, letter, receipt, legal_document, bank_statement, other)
2. "document_type_ar": Same type name in Arabic
3. "title": The document title or heading
4. "language": Primary language(s) detected (arabic, english, mixed)
5. "is_handwritten": Whether the document contains handwritten text (true/false)
6. "key_fields": List of important field names found in the document
7. "entities": Any named entities (people, organizations, departments)
8. "amounts": Any monetary amounts mentioned
9. "dates": Any dates mentioned
10. "reference_numbers": Any reference, invoice, or document numbers
11. "summary": Brief one-line summary of the document content

Respond ONLY with valid JSON, no other text."""

OCR_MODEL_ID = "sherif1313/Arabic-English-handwritten-OCR-v3"

# Primary OCR prompt — explicit instructions for complete extraction
OCR_PROMPT = """قم باستخراج النص الكامل من هذه الصورة بدقة تامة. اتبع التعليمات التالية:
1. اقرأ كل النص الموجود في الصورة من البداية إلى النهاية بدون أي اختصار أو حذف
2. حافظ على ترتيب الأسطر والفقرات كما تظهر في الصورة
3. استخرج جميع الأرقام والتواريخ والمبالغ بدقة
4. حافظ على البيانات الجدولية مع المحاذاة الصحيحة
5. اكتب النص العربي كما هو والنص الإنجليزي كما هو
6. لا تضف أي تعليقات أو تفسيرات من عندك
7. إذا كان هناك نص مكتوب بخط اليد، حاول قراءته بعناية فائقة
ابدأ بكتابة النص المستخرج:"""

# Secondary OCR prompt — English-focused for mixed documents
OCR_PROMPT_EN = """Extract ALL text from this document image with maximum accuracy. Follow these rules:
1. Read every word from top to bottom, right to left for Arabic, left to right for English
2. Preserve the exact line order and paragraph structure
3. Extract all numbers, dates, and monetary amounts precisely
4. Keep table data with proper column alignment
5. Write Arabic text exactly as written and English text exactly as written
6. Pay special attention to handwritten text — read it carefully
7. Do NOT add any comments, explanations, or text not in the image
8. Do NOT skip any text even if partially visible
Begin the extracted text:"""

# Detail-focused prompt — for numbers, names, and fine details
OCR_PROMPT_DETAIL = """اقرأ هذا المستند بعناية شديدة واستخرج:
- جميع الأسماء (أشخاص، شركات، مؤسسات، بنوك) كما هي مكتوبة بالضبط
- جميع الأرقام (أرقام الحسابات، أرقام المراجع، المبالغ، أرقام الفواتير)
- جميع التواريخ بالتنسيق الأصلي
- جميع البيانات الجدولية مع الحفاظ على الأعمدة
- النص المكتوب بخط اليد إن وجد
اكتب النص الكامل:"""

# MODE
PROCESSING_MODE = "full"

# IMAGE PREPROCESSING
GEMMA_MAX_WIDTH = 1024
GEMMA_CONTRAST = 1.5
QWEN_MAX_SIZE = 1600
QWEN_MIN_SIZE = 800
PDF_DPI = 400

# GENERATION — increased token limits for full page extraction
GEMMA_MAX_TOKENS = 2048
GEMMA_TEMPERATURE = 0.1
GEMMA_REPETITION_PENALTY = 1.3

QWEN_MAX_TOKENS = 4096
QWEN_MIN_TOKENS = 100
QWEN_TEMPERATURE = 0.1
QWEN_TOP_P = 0.3
QWEN_REPETITION_PENALTY = 1.15
QWEN_TOP_K = 50

# MULTI-PASS SETTINGS
MULTIPASS_ENABLED = True
MULTIPASS_COUNT = 3
MULTIPASS_MERGE_STRATEGY = "best_line"  # "best_line" or "longest"

# CLASSIFICATION
DOCUMENT_TYPES = {
    "payment_voucher": {
        "name_ar": "مستند صرف / سند دفع",
        "name_en": "Payment Voucher",
        "keywords": [
            "payment voucher", "ap - payment voucher", "ap-payment voucher",
            "pv no", "bc no", "pv_number", "pv number",
            "net payment", "net_payment",
            "total debits", "total_debits", "total credits",
            "credit amount", "debit amount",
            "supplier details", "supplier no", "supplier name", "supplier site",
            "supplier code", "supplier number",
            "payment details", "payment method",
            "for use of finance department",
            "for completion by submitting department",
            "invoice currency", "chart of account",
            "bank acct", "bank a/c", "bank name", "bank account",
            "cheque", "check", "wire transfer", "clearance",
            "payee code", "payee",
            "authorised for payment", "authorized for payment",
            "govt accounts", "government accounts",
            "emirate of abu dhabi",
            "مستند الصرف", "مستند صرف", "سند دفع", "سند الصرف",
            "المبالغ المدينة", "المبالغ الدائنة",
            "صافي الدفعة", "بيانات المورد",
            "رقم المورد", "اسم المورد",
            "تفاصيل الدفعة", "طريقة الدفع",
            "رقم الفاتورة", "تاريخ الفاتورة",
            "اسم البنك", "رقم الحساب",
            "التوزيع الحسابي", "رقم المستفيد",
            "تاريخ الاستلام", "اعتماد الدفع",
            "document_type", "payment",
            "debit", "credit",
        ],
        "weight": 1.0,
    },
    "invoice": {
        "name_ar": "فاتورة",
        "name_en": "Invoice",
        "keywords": [
            "invoice", "فاتورة", "فاتوره", "tax invoice", "فاتورة ضريبية",
            "bill to", "ship to", "sold to",
            "unit price", "quantity", "subtotal",
            "vat", "total amount due", "amount due",
            "invoice number", "invoice date", "invoice no",
            "due date", "payment terms",
            "item description", "line total",
            "tax registration", "trn",
            "رقم الفاتورة", "تاريخ الفاتورة",
            "المبلغ الإجمالي", "الضريبة",
            "المستلم", "البائع",
        ],
        "weight": 1.0,
    },
    "purchase_order": {
        "name_ar": "أمر شراء",
        "name_en": "Purchase Order",
        "keywords": [
            "purchase order", "أمر شراء", "امر شراء",
            "po number", "po no", "delivery date",
            "vendor", "order date", "order no",
            "requested by", "approved by",
            "ship to", "bill to",
            "طلب شراء", "رقم الأمر",
        ],
        "weight": 1.0,
    },
    "contract": {
        "name_ar": "عقد",
        "name_en": "Contract / Agreement",
        "keywords": [
            "contract", "عقد", "agreement", "اتفاقية", "اتفاقيه",
            "الطرف الأول", "الطرف الثاني", "الطرف الأول", "الطرف الثاني",
            "terms and conditions", "الشروط والأحكام",
            "effective date", "termination", "تاريخ السريان", "الإنهاء",
            "scope of work", "نطاق العمل",
            "مدة العقد", "قيمة العقد",
            "signatures", "التوقيعات",
        ],
        "weight": 1.0,
    },
    "letter": {
        "name_ar": "خطاب / رسالة",
        "name_en": "Official Letter",
        "keywords": [
            "letter", "خطاب", "رسالة", "رساله",
            "dear sir", "السيد المحترم",
            "to whom it may concern",
            "re:", "الموضوع:", "subject:",
            "sincerely", "المكرم", "وتفضلوا بقبول فائق الاحترام",
            "ref no", "reference no", "المرجع",
        ],
        "weight": 1.0,
    },
    "receipt": {
        "name_ar": "إيصال",
        "name_en": "Receipt",
        "keywords": [
            "receipt", "إيصال", "ايصال", "سند قبض",
            "received from", "استلمنا من", "تم استلام",
            "amount received", "المبلغ المستلم",
            "cash receipt", "سند قبض نقدي",
        ],
        "weight": 1.0,
    },
    "legal_document": {
        "name_ar": "وثيقة قانونية",
        "name_en": "Legal Document",
        "keywords": [
            "نظام", "قانون", "لائحة", "مرسوم",
            "المادة الأولى", "المادة", "مادة",
            "هيئة الخبراء", "مجلس الوزراء",
            "bureau of experts", "council of ministers",
            "الملكة العربية السعودية", "المملكة العربية السعودية",
            "مرسوم ملكي", "قرار مجلس",
            "باب تمهيدي", "أحكام عامة",
            "disposition", "provision",
        ],
        "weight": 1.0,
    },
    "bank_statement": {
        "name_ar": "كشف حساب بنكي",
        "name_en": "Bank Statement",
        "keywords": [
            "bank statement", "كشف حساب", "كشف حساب بنكي",
            "opening balance", "closing balance",
            "iban", "statement period", "فترة البيان",
            "account statement", "حركة الحساب",
            "debit", "credit", "balance",
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

# API SERVER
API_HOST = "0.0.0.0"
API_PORT = 8000

# QUANTIZATION — use 4-bit to save VRAM for larger models
USE_4BIT_QUANTIZATION = False  # Set True if VRAM < 16GB
