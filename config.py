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

OCR_MODEL_ID = "sherif1313/Arabic-English-handwritten-OCR-v3"

# ===== OCR PROMPTS — 5 diverse prompts for maximum coverage =====

# Pass 1: Primary Arabic prompt — comprehensive extraction
OCR_PROMPT = """أنت خبير في قراءة المستندات العربية والإنجليزية. قم باستخراج النص الكامل من هذه الصورة بدقة تامة 100%.

التعليمات الصارمة:
1. اقرأ كل حرف وكل كلمة وكل سطر من البداية إلى النهاية بدون أي اختصار أو حذف أو إضافة
2. حافظ على ترتيب الأسطر والفقرات بالضبط كما تظهر في الصورة من الأعلى إلى الأسفل
3. استخرج جميع الأرقام والتواريخ والمبالغ بدقة متناهية — لا تغير أي رقم
4. حافظ على البيانات الجدولية مع المحاذاة الصحيحة والأعمدة
5. اكتب النص العربي كما هو بالضبط والنص الإنجليزي كما هو بالضبط
6. لا تضف أي تعليقات أو تفسيرات أو تصحيحات من عندك
7. إذا كان هناك نص مكتوب بخط اليد، اقرأه بعناية فائقة كلمة بكلمة
8. لا تتجاهل أي نص حتى لو كان غير واضح تماماً — حاول قراءته
9. حافظ على المسافات والتنسيق الأصلي قدر الإمكان
10. إذا كان هناك ختم أو توقيع، اكتب [ختم] أو [توقيع]

ابدأ بكتابة النص المستخرج بالضبط كما يظهر:"""

# Pass 2: English prompt — for mixed/English documents
OCR_PROMPT_EN = """You are an expert OCR reader for Arabic and English documents. Extract ALL text from this document image with 100% accuracy.

Strict rules:
1. Read every single character, word, and line from top to bottom without any omission or addition
2. For Arabic text: read right to left. For English text: read left to right
3. Preserve the exact line order and paragraph structure as they appear
4. Extract all numbers, dates, and monetary amounts with perfect precision — never change any digit
5. Keep table data with proper column alignment using spaces
6. Write Arabic text exactly as written and English text exactly as written
7. Pay extreme attention to handwritten text — read it word by word carefully
8. Do NOT add any comments, explanations, or text not in the image
9. Do NOT skip any text even if partially visible — try your best to read it
10. Preserve original spacing and formatting as much as possible
11. If there is a stamp or signature, write [stamp] or [signature]

Begin the extracted text exactly as it appears:"""

# Pass 3: Detail-focused Arabic prompt — for numbers, names, fine details
OCR_PROMPT_DETAIL = """اقرأ هذا المستند بعناية شديدة جداً واستخرج كل التفاصيل:
- جميع الأسماء (أشخاص، شركات، مؤسسات، بنوك، دوائر حكومية) كما هي مكتوبة بالضبط حرفياً
- جميع الأرقام (أرقام الحسابات، أرقام المراجع، المبالغ، أرقام الفواتير، أرقام الهواتف) بدون أي تغيير
- جميع التواريخ بالتنسيق الأصلي بالضبط
- جميع البيانات الجدولية مع الحفاظ على الأعمدة والمحاذاة
- النص المكتوب بخط اليد إن وجد — اقرأه حرفاً حرفاً
- جميع الأختمة والتوقيعات — اكتب [ختم] أو [توقيع]
- لا تختصر ولا تحذف ولا تضيف أي شيء

اكتب النص الكامل بالضبط:"""

# Pass 4: Table/structure-focused prompt — for structured documents
OCR_PROMPT_TABLE = """Extract all content from this document with special focus on structure and tables.

Instructions:
1. Read every field label and its value — do not miss any field
2. For tables: extract each row completely with all column values
3. For forms: extract each field name followed by its value
4. Preserve the exact numeric values — amounts, account numbers, reference numbers
5. Read both Arabic and English text exactly as written
6. For any checkboxes or marked fields, indicate [checked] or [unchecked]
7. Maintain the visual structure using line breaks and spacing

Write the complete extracted text:"""

# Pass 5: Verification/correction prompt — asks model to be extra careful
OCR_PROMPT_VERIFY = """هذه صورة مستند رسمي. اقرأها بعناية مضاعفة وتأكد من كل كلمة:

- اقرأ كل سطر مرتين في ذهنك قبل كتابته
- تأكد من كل رقم وعلامة ترقيم
- إذا كانت هناك كلمة غير واضحة، اكتب أقرب قراءة صحيحة لها
- لا تترك أي سطر بدون قراءة
- انتبه بشكل خاص للأسماء والأرقام والمبالغ
- حافظ على ترتيب النص بالضبط كما في الصورة

اكتب النص الكامل والمؤكد:"""

# MODE
PROCESSING_MODE = "full"

# IMAGE PREPROCESSING
GEMMA_MAX_WIDTH = 1024
GEMMA_CONTRAST = 1.5
QWEN_MAX_SIZE = 1600
QWEN_MIN_SIZE = 800
PDF_DPI = 400

# GENERATION — tuned for maximum accuracy
GEMMA_MAX_TOKENS = 4096
GEMMA_TEMPERATURE = 0.05
GEMMA_REPETITION_PENALTY = 1.2
GEMMA_TOP_P = 0.85
GEMMA_TOP_K = 40

QWEN_MAX_TOKENS = 8192
QWEN_MIN_TOKENS = 200
QWEN_TEMPERATURE = 0.05
QWEN_TOP_P = 0.85
QWEN_REPETITION_PENALTY = 1.1
QWEN_TOP_K = 40

# MULTI-PASS SETTINGS — 5 passes for maximum accuracy
MULTIPASS_ENABLED = True
MULTIPASS_COUNT = 5
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
