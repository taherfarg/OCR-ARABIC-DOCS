"""
Dual model config — maximized for highest OCR + classification accuracy.

Key accuracy improvements:
- 5-pass OCR with diverse prompts and preprocessing
- Lower temperature for more deterministic output
- Higher max tokens to never truncate
- Verification/correction pass
- Quality-gated retry
- Expanded keyword dictionaries
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
1. "document_type": Identify the type (payment_voucher, invoice, purchase_order, contract, letter, receipt, legal_document, bank_statement, memorandum, report, other)
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

IMPORTANT: Respond ONLY with valid JSON. Do not include any text before or after the JSON."""

OCR_MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"

# ===== OCR PROMPTS — Short, focused, anti-hallucination =====

# Pass 1: Arabic OCR — short prompt, explicit anti-hallucination
OCR_PROMPT = """اقرأ كل النص الموجود في هذه الصورة من الأعلى إلى الأسفل. اكتب النص بالضبط كما يظهر بدون أي إضافة أو تعديل أو شرح. لا تضف أي نص ليس في الصورة."""

# Pass 2: English OCR — short prompt, explicit anti-hallucination
OCR_PROMPT_EN = """Read all text in this image from top to bottom. Write the text exactly as it appears. Do not add, explain, or modify anything. Do not offer help or suggestions. Only output what is written in the image."""

# Structured extraction — for forms/vouchers/invoices
OCR_PROMPT_STRUCTURED = """Extract all fields and values from this document. Output each field on a new line as: Field: Value
Include all names, numbers, amounts, dates, and account numbers exactly as written. Do not add anything not in the image."""

# Legacy prompts (kept for reference, not used in default pipeline)
OCR_PROMPT_DETAIL = OCR_PROMPT
OCR_PROMPT_TABLE = OCR_PROMPT_STRUCTURED
OCR_PROMPT_VERIFY = OCR_PROMPT

# MODE
PROCESSING_MODE = "full"

# IMAGE PREPROCESSING
GEMMA_MAX_WIDTH = 1024
GEMMA_CONTRAST = 1.5
QWEN_MAX_SIZE = 2000
QWEN_MIN_SIZE = 800
PDF_DPI = 400

# GENERATION — tuned for maximum accuracy
GEMMA_MAX_TOKENS = 2048
GEMMA_TEMPERATURE = 0.05
GEMMA_REPETITION_PENALTY = 1.2
GEMMA_TOP_P = 0.85
GEMMA_TOP_K = 40

QWEN_MAX_TOKENS = 1024            # Single page rarely needs more than 800 tokens
QWEN_MIN_TOKENS = 0                # Don't force generation past content end
QWEN_TEMPERATURE = 1.0             # Unused with do_sample=False (greedy)
QWEN_TOP_P = 1.0                   # Unused with do_sample=False (greedy)
QWEN_REPETITION_PENALTY = 1.1      # Mild — too high causes word avoidance
QWEN_TOP_K = None                  # Unused with do_sample=False (greedy)
QWEN_DO_SAMPLE = False             # Greedy decoding for deterministic OCR

# GENERATION SAFETY — prevent hangs and infinite loops
GENERATION_TIMEOUT_SEC = 300       # Max seconds per generation call (5 min)
NO_REPEAT_NGRAM_SIZE = 6           # Prevent repetition loops in generation
QWEN_MAX_PIXELS = 2007040          # 1600*28*28 — more visual tokens for small text

# MULTI-PASS SETTINGS — 2 passes (Arabic + English) + optional structured
MULTIPASS_ENABLED = True
MULTIPASS_COUNT = 2
MULTIPASS_MERGE_STRATEGY = "best_line"  # "best_line" or "longest"

# QUALITY GATE — retry if quality is poor
QUALITY_RETRY_ENABLED = True
QUALITY_RETRY_MAX_ATTEMPTS = 2  # Max retry attempts per document
QUALITY_MIN_CHARS = 50  # Minimum characters for acceptable output
QUALITY_MIN_LINES = 3   # Minimum non-empty lines

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
            "pv bc", "voucher no", "voucher number",
            "payment advice", "سند صرف نقدي",
            "الصرف", "مستندات الصرف",
            "حساب المورد", "إذن دفع",
            "أمر دفع", "إشعار دفع",
            "payment order", "payment advice note",
            "debit advice", "credit advice",
            "إشعار مدين", "إشعار دائن",
        ],
        "weight": 1.0,
    },
    "invoice": {
        "name_ar": "فاتورة",
        "name_en": "Invoice",
        "keywords": [
            "invoice", "فاتورة", "فاتوره", "tax invoice", "فاتورة ضريبية",
            "فاتوره ضريبيه", "فاتورة ضريبة", "فاتوره ضريبه",
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
            "المجموع الفرعي", "سعر الوحدة",
            "الكمية", "وصف البند",
            "إجمالي الفاتورة", "المبلغ المستحق",
            "رقم التسجيل الضريبي", "الرقم الضريبي",
            "فاتورة مبدئية", "فاتورة نهائية",
            "proforma invoice", "commercial invoice",
            "فاتورة تجارية", "فاتورة أولية",
            "tax id", "tax number", "الرقم الضريبي",
            "seller", "buyer", "البائع", "المشتري",
            "invoice total", "grand total",
            "الإجمالي الكلي", "صافي الفاتورة",
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
            "أمر توريد", "طلب توريد",
            "المورد", "تاريخ التسليم",
            "اعتماد", "طلب شراء", "أمر شراء رسمي",
            "purchase requisition", "طلب احتياج",
            "procurement", "المشتريات", "التوريدات",
            "specifications", "المواصفات",
            "quantity ordered", "الكمية المطلوبة",
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
            "مادة", "بنود", "شرط جزائي",
            "penalty clause", "force majeure", "قوة قاهرة",
            "confidentiality", "سرية",
            "arbitration", "تحكيم",
            "liability", "مسؤولية",
            "indemnification", "تعويض",
            "representations and warranties", "الإقرارات والضمانات",
            "governing law", "القانون الحاكم",
            "dispute resolution", "حل النزاعات",
            "عقد خدمات", "عقد توريد", "عقد تشغيل وصيانة",
            "service agreement", "service level agreement",
            "اتفاقية مستوى الخدمة",
            "memorandum of understanding", "مذكرة تفاهم",
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
            "circular", "تعميم", "إخطار", "notification",
            "correspondence", "مراسلة", "مراسلات",
            "official letter", "خطاب رسمي",
            "معالي", "سعادة", "سعادة الوزير",
            "your excellency", "your highness",
            "صاحب السمو", "صاحب المعالي",
            "تحية طيبة وبعد", "والسلام عليكم ورحمة الله",
            "cc:", "نسخة إلى", "توزيع",
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
            "receipt no", "رقم الإيصال",
            "received by", "استلم بواسطة",
            "cash payment", "دفعة نقدية",
            "إيصال استلام", "إيصال قبض",
            "acknowledgment", "إقرار", "إقرار استلام",
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
            "المادة الثانية", "المادة الثالثة",
            "الفصل الأول", "الفصل الثاني",
            "القسم الأول", "القسم الثاني",
            "أحكام ختامية", "أحكام انتقالية",
            "تنفيذية", "لائحة تنفيذية",
            "royal decree", "مرسوم ملكي",
            "cabinet decision", "قرار وزاري",
            "ministerial resolution", "قرار وزاري",
            "regulation", "decree", "law", "statute",
            "تشريع", "نظام قانوني",
            "حكم", "أحكام", "نص قانوني",
            "jurisdiction", "اختصاص",
            "penalty", "عقوبة", "جزاء",
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
            "رصيد افتتاحي", "رصيد ختامي",
            "إيداع", "سحب", "تحويل",
            "deposit", "withdrawal", "transfer",
            "transaction", "عملية", "حركة",
            "account number", "رقم الحساب",
            "branch", "الفرع",
            "statement date", "تاريخ البيان",
            "reference", "المرجع",
            "description", "البيان", "الوصف",
            "value date", "تاريخ القيمة",
        ],
        "weight": 1.0,
    },
    "memorandum": {
        "name_ar": "مذكرة",
        "name_en": "Memorandum",
        "keywords": [
            "memorandum", "مذكرة", "مذكره",
            "memo", "internal memo", "مذكرة داخلية",
            "from:", "to:", "date:", "من:", "إلى:", "تاريخ:",
            "background", "الخلفية",
            "recommendation", "التوصية", "توصيات",
            "objective", "الهدف", "الأهداف",
            "analysis", "التحليل", "دراسة",
            "findings", "النتائج", "ملاحظات",
            "conclusion", "الخلاصة",
            "executive summary", "ملخص تنفيذي",
        ],
        "weight": 1.0,
    },
    "report": {
        "name_ar": "تقرير",
        "name_en": "Report",
        "keywords": [
            "report", "تقرير", "تقرير سنوي",
            "annual report", "monthly report", "تقرير شهري",
            "progress report", "تقرير إنجاز",
            "financial report", "تقرير مالي",
            "audit report", "تقرير تدقيق",
            "technical report", "تقرير فني",
            "statistics", "إحصائيات",
            "performance", "الأداء",
            "achievements", "الإنجازات",
            "challenges", "التحديات",
            "recommendations", "التوصيات",
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
USE_4BIT_QUANTIZATION = True   # Enabled for RTX 4060 8GB VRAM
