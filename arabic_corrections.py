"""
Arabic/English dictionary-based corrections.
Fixes common OCR mistakes specific to Abu Dhabi/UAE government documents.

Expanded significantly with:
- 100+ Arabic OCR error corrections
- 50+ English OCR error corrections
- Context-aware corrections (field-specific)
- Common government term corrections
- Number/amount pattern fixes
- Date format normalization
"""
import re
from loguru import logger


# ===== KNOWN WORD CORRECTIONS =====
# Maps common OCR errors to correct text
ARABIC_CORRECTIONS = {
    # === تفاصيل (details) — very commonly misread ===
    'تغاضيل': 'تفاصيل',
    'تغاصيل': 'تفاصيل',
    'تقاصيل': 'تفاصيل',
    'تفاضيل': 'تفاصيل',
    'تفاصل': 'تفاصيل',
    'تفاصي': 'تفاصيل',

    # === فاتورة (invoice) ===
    'الكاتورة': 'الفاتورة',
    'الكاتوره': 'الفاتورة',
    'الغاتورة': 'الفاتورة',
    'الفاتوره': 'الفاتورة',
    'فاتوره': 'فاتورة',
    'الفاثورة': 'الفاتورة',
    'الفاتوررة': 'الفاتورة',
    'فاثورة': 'فاتورة',
    'فاتورةضريبية': 'فاتورة ضريبية',

    # === مورد (supplier) ===
    'الوورد': 'المورد',
    'الوورّد': 'المورّد',
    'الموررد': 'المورد',
    'المورردة': 'الموردة',
    'موررد': 'مورد',

    # === مبالغ (amounts) ===
    'العماله': 'المبالغ',
    'المبالع': 'المبالغ',
    'المبالغا': 'المبالغ',
    'مبالغا': 'مبالغ',

    # === داخلي (internal) ===
    'اليوناني': 'الداخلي',
    'الداخلى': 'الداخلي',

    # === مستند صرف (payment voucher) ===
    'مستندصرف': 'مستند صرف',
    'مستندالصرف': 'مستند الصرف',
    'مستندصرفنقدي': 'مستند صرف نقدي',
    'مستنداتالصرف': 'مستندات الصرف',
    'مستنداتصرف': 'مستندات صرف',

    # === توزيع (distribution) ===
    'تنزيع': 'توزيع',
    'توزيغ': 'توزيع',
    'التوزيعالحسابي': 'التوزيع الحسابي',

    # === حسابي (accounting) ===
    'الحسبي': 'الحسابي',
    'الحسباي': 'الحسابي',
    'حسابي': 'حسابي',

    # === دائنة (credit) ===
    'الدائمة': 'الدائنة',
    'الدائنه': 'الدائنة',
    'الدائتة': 'الدائنة',

    # === مدينة (debit) ===
    'المدينه': 'المدينة',
    'المديتة': 'المدينة',

    # === معيار (standard) ===
    'العيار': 'المعيار',

    # === أخرى (other) ===
    'الاخرى': 'الأخرى',
    'الاخري': 'الأخرى',

    # === لوائح (regulations) ===
    'الائتمانات': 'اللوائح',
    'اللوايح': 'اللوائح',
    'لائحه': 'لائحة',

    # === Government terms ===
    'الشؤون الادارية': 'الشؤون الإدارية',
    'الشؤون الاداريه': 'الشؤون الإدارية',
    'الشؤونالإدارية': 'الشؤون الإدارية',
    'المالية': 'المالية',
    'الماليه': 'المالية',
    'الدائرة': 'الدائرة',
    'الدائره': 'الدائرة',
    'دائرة المالية': 'دائرة المالية',
    'دائرةالمالية': 'دائرة المالية',

    # === Abu Dhabi specific ===
    'امارة ابوظبي': 'إمارة أبوظبي',
    'امارة أبوظبي': 'إمارة أبوظبي',
    'إمارة ابوظبي': 'إمارة أبوظبي',
    'ابو ظبي': 'أبوظبي',
    'ابوظبي': 'أبوظبي',
    'أبو ظبي': 'أبوظبي',
    'حكومة ابوظبي': 'حكومة أبوظبي',
    'حكومة أبو ظبي': 'حكومة أبوظبي',

    # === UAE terms ===
    'الإمارات العربية المتحدة': 'الإمارات العربية المتحدة',
    'الامارات العربية المتحدة': 'الإمارات العربية المتحدة',
    'الامارات العربيه المتحده': 'الإمارات العربية المتحدة',
    'دولةالإمارات': 'دولة الإمارات',
    'دولة الامارات': 'دولة الإمارات',

    # === Payment/financial terms ===
    'صافيالدفعة': 'صافي الدفعة',
    'صافى الدفعة': 'صافي الدفعة',
    'المبلغالإجمالي': 'المبلغ الإجمالي',
    'المبلغ الاجمالي': 'المبلغ الإجمالي',
    'الضريبة': 'الضريبة',
    'الضريبه': 'الضريبة',
    'القيمةالمضافة': 'القيمة المضافة',
    'القيمه المضافه': 'القيمة المضافة',

    # === Bank terms ===
    'رقمالحساب': 'رقم الحساب',
    'اسمالبنك': 'اسم البنك',
    'الحسابالبنكي': 'الحساب البنكي',
    'رقمالحسابالبنكي': 'رقم الحساب البنكي',
    'البنكالمركزي': 'البنك المركزي',

    # === Document fields ===
    'رقمالفاتورة': 'رقم الفاتورة',
    'تاريخالفاتورة': 'تاريخ الفاتورة',
    'رقمالمورد': 'رقم المورد',
    'اسمالمورد': 'اسم المورد',
    'رقمالمستفيد': 'رقم المستفيد',
    'طريقةالدفع': 'طريقة الدفع',
    'تفاصيلالدفع': 'تفاصيل الدفع',
    'بياناتالمورد': 'بيانات المورد',
    'اعتمادالدفع': 'اعتماد الدفع',
    'تاريخالاستلام': 'تاريخ الاستلام',
    'التوزيعالحسابي': 'التوزيع الحسابي',

    # === Common Arabic OCR confusions ===
    'لاال': 'لالا',
    'اللغ': 'اللغة',
    'اللغه': 'اللغة',
    'النسخه': 'النسخة',
    'النسخةالأصلية': 'النسخة الأصلية',
    'الرقمالمرجعي': 'الرقم المرجعي',
    'الرقمالمرجعى': 'الرقم المرجعي',
    'الملاحظات': 'الملاحظات',
    'الملاحظات': 'الملاحظات',
    'التوقيع': 'التوقيع',
    'التوقيعات': 'التوقيعات',
    'الختم': 'الختم',
    'الاعتماد': 'الاعتماد',
    'الاعتماده': 'الاعتمادة',
    'الموافقة': 'الموافقة',
    'الموافقه': 'الموافقة',
    'المراجعة': 'المراجعة',
    'المراجعه': 'المراجعة',
    'التنفيذ': 'التنفيذ',
    'التنفيذه': 'التنفيذية',

    # === Legal terms ===
    'المادةالأولى': 'المادة الأولى',
    'المادهالأولى': 'المادة الأولى',
    'المادةالثانية': 'المادة الثانية',
    'المادةالثالثة': 'المادة الثالثة',
    'أحكامعامة': 'أحكام عامة',
    'أحكامختامية': 'أحكام ختامية',
    'أحكامانتقالية': 'أحكام انتقالية',
    'بابتمهيدي': 'باب تمهيدي',
    'مرسومملكي': 'مرسوم ملكي',
    'قرارمجلس': 'قرار مجلس',
    'مجلسالوزراء': 'مجلس الوزراء',
    'هيئةالخبراء': 'هيئة الخبراء',
    'المملكةالعربيةالسعودية': 'المملكة العربية السعودية',
    'الملكةالعربيةالسعودية': 'المملكة العربية السعودية',

    # === Common letter/word confusions ===
    'السلامعليكم': 'السلام عليكم',
    'ورحمةالله': 'ورحمة الله',
    'وبركاته': 'وبركاته',
    'بسماللهالرحمنالرحيم': 'بسم الله الرحمن الرحيم',
    'بسمالله': 'بسم الله',
    'تحيةطيبةوبعد': 'تحية طيبة وبعد',
    'والسلامعليكم': 'والسلام عليكم',
    'وتفضلوابقبول': 'وتفضلوا بقبول',
    'فائقالاحترام': 'فائق الاحترام',
    'فائقالاحتراموالتقدير': 'فائق الاحترام والتقدير',

    # === Department/organization names ===
    'دائرةالخدماتالمالية': 'دائرة الخدمات المالية',
    'دائرةالشؤونالقانونية': 'دائرة الشؤون القانونية',
    'وزارةالمالية': 'وزارة المالية',
    'ديوانوليالعهد': 'ديوان ولي العهد',
    'ديوانوليالعهد': 'ديوان ولي العهد',
    'مجلسأبوظبي': 'مجلس أبوظبي',
    'هيئةأبوظبي': 'هيئة أبوظبي',

    # === Amount/currency ===
    'درهمإماراتي': 'درهم إماراتي',
    'دراهمإماراتية': 'دراهم إماراتية',
    'د.إ': 'د.إ',
    'AED': 'AED',
    'ريالسعودي': 'ريال سعودي',
    'ر.س': 'ر.س',

    # === Misc common errors ===
    'الإجمالي': 'الإجمالي',
    'الإجمالى': 'الإجمالي',
    'المجموع': 'المجموع',
    'المجموعالفرعي': 'المجموع الفرعي',
    'الكمية': 'الكمية',
    'الكميه': 'الكمية',
    'السعر': 'السعر',
    'سعرالوحدة': 'سعر الوحدة',
    'سعرالوحده': 'سعر الوحدة',
    'الوصف': 'الوصف',
    'وصفالبند': 'وصف البند',
    'الإجماليالكلي': 'الإجمالي الكلي',
    'صافيالمبلغ': 'صافي المبلغ',
    'المبلغالمستحق': 'المبلغ المستحق',
    'المبلغالمستلم': 'المبلغ المستلم',
    'تاريخالاستحقاق': 'تاريخ الاستحقاق',
    'شروطالدفع': 'شروط الدفع',
    'طريقةالتوريد': 'طريقة التوريد',
}

ENGLISH_CORRECTIONS = {
    # Common OCR English errors — payment/financial
    'Clearace': 'Clearance',
    'Clearane': 'Clearance',
    'Clearnce': 'Clearance',
    'Suppliar': 'Supplier',
    'Suplier': 'Supplier',
    'Supplire': 'Supplier',
    'Suppllier': 'Supplier',
    'Ammount': 'Amount',
    'Amoutn': 'Amount',
    'Amont': 'Amount',
    'Amounnt': 'Amount',
    'Payament': 'Payment',
    'Paymant': 'Payment',
    'Paymnet': 'Payment',
    'Payemnt': 'Payment',
    'Invioce': 'Invoice',
    'Inovice': 'Invoice',
    'Invoce': 'Invoice',
    'Invoiec': 'Invoice',
    'Certiicate': 'Certificate',
    'Certficate': 'Certificate',
    'Certificaet': 'Certificate',
    'Authoised': 'Authorised',
    'Authroised': 'Authorised',
    'Authoried': 'Authorised',
    'Authorizd': 'Authorized',
    'Goverment': 'Government',
    'Governmnet': 'Government',
    'Govermnent': 'Government',
    'Departmant': 'Department',
    'Deparment': 'Department',
    'Departmetn': 'Department',
    'Acconut': 'Account',
    'Acocunt': 'Account',
    'Accoutn': 'Account',
    'Acccount': 'Account',
    'Distribtuion': 'Distribution',
    'Ditribution': 'Distribution',
    'Distirbution': 'Distribution',
    'Regulatons': 'Regulations',
    'Regulaitons': 'Regulations',
    'Regultaions': 'Regulations',
    'oF': 'of',
    'lnvoice': 'Invoice',
    'Emiratos': 'Emirates',
    'Arnount': 'Amount',
    'Nurnber': 'Number',
    'Numbr': 'Number',
    'Numbner': 'Number',
    'Refrence': 'Reference',
    'Refernce': 'Reference',
    'Referecne': 'Reference',

    # Bank names
    'National Bank oF': 'National Bank of',
    'Sree code': 'Swift Code',
    'Swft Code': 'Swift Code',
    'Swif Code': 'Swift Code',

    # Common typos
    'G.OVT': 'GOVT',
    'G O V T': 'GOVT',
    'GOVERMENT': 'GOVERNMENT',
    'FINACIAL': 'FINANCIAL',
    'FINANCILA': 'FINANCIAL',
    'DEPARMENT': 'DEPARTMENT',
    'DEPARTMNET': 'DEPARTMENT',

    # Payment terms
    'Cheuqe': 'Cheque',
    'Chque': 'Cheque',
    'Cheqe': 'Cheque',
    'Tranfer': 'Transfer',
    'Transfe': 'Transfer',
    'Transfr': 'Transfer',
    'Wire Trnasfer': 'Wire Transfer',
    'Wire Tansfer': 'Wire Transfer',

    # Document fields
    'Desciption': 'Description',
    'Descritpion': 'Description',
    'Descripion': 'Description',
    'Quantitiy': 'Quantity',
    'Qauntity': 'Quantity',
    'Quntity': 'Quantity',
    'Subotal': 'Subtotal',
    'Subtoal': 'Subtotal',
    'Ttoal': 'Total',
    'Toatl': 'Total',
    'Tota': 'Total',
    'Reciept': 'Receipt',
    'Receipet': 'Receipt',
    'Recepit': 'Receipt',
    'Vocher': 'Voucher',
    'Voucer': 'Voucher',
    'Vouchre': 'Voucher',
    'Purcahse': 'Purchase',
    'Purchse': 'Purchase',
    'Purhase': 'Purchase',

    # Arabic-English mixed errors
    'Emirati': 'Emirati',
    'Abu Dhab': 'Abu Dhabi',
    'AbuDhabi': 'Abu Dhabi',
    'U A E': 'UAE',
    'U.A.E.': 'UAE',
}

# Number pattern corrections
NUMBER_PATTERNS = {
    # Arabic to Western numeral mapping for specific fields
    '٠': '0', '١': '1', '٢': '2', '٣': '3', '٤': '4',
    '٥': '5', '٦': '6', '٧': '7', '٨': '8', '٩': '9',
    '۰': '0', '۱': '1', '۲': '2', '۳': '3', '۴': '4',
    '۵': '5', '۶': '6', '۷': '7', '۸': '8', '۹': '9',
}


def apply_arabic_corrections(text: str) -> str:
    """Apply Arabic word corrections."""
    count = 0
    for wrong, correct in ARABIC_CORRECTIONS.items():
        if wrong in text:
            text = text.replace(wrong, correct)
            count += 1

    if count > 0:
        logger.debug(f"📝 Applied {count} Arabic corrections")
    return text


def apply_english_corrections(text: str) -> str:
    """Apply English word corrections (case-insensitive where needed)."""
    count = 0
    for wrong, correct in ENGLISH_CORRECTIONS.items():
        if wrong in text:
            text = text.replace(wrong, correct)
            count += 1

    # Also try case-insensitive for some
    for wrong, correct in ENGLISH_CORRECTIONS.items():
        pattern = re.compile(re.escape(wrong), re.IGNORECASE)
        if pattern.search(text):
            text = pattern.sub(correct, text)

    if count > 0:
        logger.debug(f"📝 Applied {count} English corrections")
    return text


def normalize_numbers_in_fields(text: str) -> str:
    """
    Convert Arabic/Persian numerals to Western in specific contexts.
    Only for fields where Western numerals are expected
    (account numbers, reference numbers, etc.)
    """
    # Convert in specific field contexts
    field_patterns = [
        r'(No\.?\s*:?\s*)([٠-٩۰-۹\s\-]+)',           # No. : numbers
        r'(Bank Acct\.?\s*(?:No\.?)?\s*:?\s*)([٠-٩۰-۹\s\-]+)',  # Bank Acct
        r'(PV No\.?\s*:?\s*)([٠-٩۰-۹\s\-]+)',         # PV No
        r'(BC NO\.?\s*:?\s*)([٠-٩۰-۹\s\-]+)',         # BC NO
        r'(Supplier No\.?\s*:?\s*)([٠-٩۰-۹\s\-]+)',   # Supplier No
        r'(Invoice No\.?\s*:?\s*)([٠-٩۰-۹\s\-]+)',    # Invoice No
        r'(رقم\s*:?\s*)([٠-٩۰-۹\s\-]+)',              # رقم : numbers
        r'(رقم الحساب\s*:?\s*)([٠-٩۰-۹\s\-]+)',       # رقم الحساب
        r'(المبلغ\s*:?\s*)([٠-٩۰-۹\s\-.,]+)',          # المبلغ
    ]

    for pattern in field_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            prefix = match.group(1)
            number_part = match.group(2)

            # Convert Arabic/Persian numerals
            converted = number_part
            for ar, en in NUMBER_PATTERNS.items():
                converted = converted.replace(ar, en)

            text = text.replace(
                prefix + number_part,
                prefix + converted,
            )

    return text


def fix_table_alignment(text: str) -> str:
    """
    Fix table data that got jumbled.
    Normalize tab/space alignment.
    """
    lines = text.split('\n')
    fixed = []

    for line in lines:
        # Replace multiple tabs with single tab
        line = re.sub(r'\t{2,}', '\t', line)
        # Replace 3+ spaces with tab (likely table column)
        line = re.sub(r' {3,}', '\t', line)
        fixed.append(line)

    return '\n'.join(fixed)


def fix_common_arabic_patterns(text: str) -> str:
    """
    Fix common Arabic text patterns that OCR often gets wrong.
    Context-aware corrections based on surrounding text.
    """
    # Fix disconnected Arabic words (common OCR error)
    # Example: "ال مبلغ" → "المبلغ"
    text = re.sub(r'ال\s+([أ-ي])', r'ال\1', text)
    
    # Fix "ال ال" → "ال" (double definite article)
    text = re.sub(r'ال\s*ال', 'ال', text)
    
    # Fix disconnected "و" (and) prefix
    text = re.sub(r'و\s{2,}([أ-ي])', r'و \1', text)
    
    # Fix "ب" prefix disconnected
    text = re.sub(r'ب\s{2,}([أ-ي])', r'ب\1', text)
    
    # Fix "لل" prefix disconnected
    text = re.sub(r'لل\s+([أ-ي])', r'لل\1', text)
    
    # Fix common date format issues
    # Arabic dates: DD/MM/YYYY or YYYY/MM/DD
    text = re.sub(r'(\d{1,2})\s*/\s*(\d{1,2})\s*/\s*(\d{4})', r'\1/\2/\3', text)
    text = re.sub(r'(\d{4})\s*/\s*(\d{1,2})\s*/\s*(\d{1,2})', r'\1/\2/\3', text)
    
    # Fix amount format: remove spaces in numbers
    # Example: "1 000.00" → "1,000.00"
    text = re.sub(r'(\d)\s{1}(\d{3})', r'\1,\2', text)
    
    # Fix disconnected punctuation
    text = re.sub(r'\s+:\s+', ': ', text)
    text = re.sub(r'\s+;\s+', '; ', text)
    
    return text


def fix_ocr_artifacts(text: str) -> str:
    """
    Fix common OCR artifacts that aren't language-specific.
    """
    # Remove stray single characters that are likely noise
    # But keep them if they're part of known abbreviations
    lines = text.split('\n')
    fixed = []
    
    for line in lines:
        # Remove lines that are just dots or dashes
        if re.match(r'^[\s.\-_—]+$', line):
            continue
        
        # Fix broken numbers: "O" instead of "0" in numeric contexts
        line = re.sub(r'(\d)O(\d)', r'\g<1>0\2', line)
        line = re.sub(r'O(\d{2,})', r'0\1', line)
        
        # Fix "l" instead of "1" in numeric contexts
        line = re.sub(r'(\d)l(\d)', r'\g<1>1\2', line)
        
        # Fix "S" instead of "5" in numeric contexts
        line = re.sub(r'(\d)S(\d)', r'\g<1>5\2', line)
        
        fixed.append(line)
    
    return '\n'.join(fixed)


def apply_all_corrections(text: str) -> str:
    """
    Apply all corrections in order.
    Order matters — some corrections depend on others being applied first.
    """
    original = text

    # 1. Fix OCR artifacts first (before language-specific corrections)
    text = fix_ocr_artifacts(text)

    # 2. Arabic corrections
    text = apply_arabic_corrections(text)

    # 3. English corrections
    text = apply_english_corrections(text)

    # 4. Common Arabic pattern fixes
    text = fix_common_arabic_patterns(text)

    # 5. Number normalization
    text = normalize_numbers_in_fields(text)

    # 6. Table alignment
    text = fix_table_alignment(text)

    if text != original:
        diff = sum(1 for a, b in zip(text, original) if a != b)
        logger.info(f"📝 Dictionary corrections: {diff} chars changed")

    return text
