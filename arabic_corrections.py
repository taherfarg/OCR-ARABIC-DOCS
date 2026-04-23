"""
Arabic/English dictionary-based corrections.
Fixes common OCR mistakes specific to Abu Dhabi government documents.
"""
import re
from loguru import logger


# ===== KNOWN WORD CORRECTIONS =====
# Maps common OCR errors to correct text
ARABIC_CORRECTIONS = {
    # Common OCR Arabic errors
    'تغاضيل': 'تفاصيل',
    'تغاصيل': 'تفاصيل',
    'تقاصيل': 'تفاصيل',
    'الكاتورة': 'الفاتورة',
    'الكاتوره': 'الفاتورة',
    'الغاتورة': 'الفاتورة',
    'الوورد': 'المورد',
    'الوورّد': 'المورّد',
    'العماله': 'المبالغ',
    'اليوناني': 'الداخلي',
    'مستندصرف': 'مستند صرف',
    'مستندالصرف': 'مستند الصرف',
    'تنزيع': 'توزيع',
    'الحسبي': 'الحسابي',
    'الدائمة': 'الدائنة',
    'العيار': 'المعيار',
    'الاخرى': 'الأخرى',
    'الائتمانات': 'اللوائح',
    'تغاضيل': 'تفاصيل',

    # Government terms
    'الشؤون الادارية': 'الشؤون الإدارية',
    'المالية': 'المالية',
    'الدائرة': 'الدائرة',
    'دائرة المالية': 'دائرة المالية',

    # Abu Dhabi specific
    'امارة ابوظبي': 'إمارة أبوظبي',
    'ابو ظبي': 'أبوظبي',
}

ENGLISH_CORRECTIONS = {
    # Common OCR English errors
    'Clearace': 'Clearance',
    'Clearane': 'Clearance',
    'Suppliar': 'Supplier',
    'Suplier': 'Supplier',
    'Ammount': 'Amount',
    'Amoutn': 'Amount',
    'Payament': 'Payment',
    'Paymant': 'Payment',
    'Invioce': 'Invoice',
    'Inovice': 'Invoice',
    'Certiicate': 'Certificate',
    'Certficate': 'Certificate',
    'Authoised': 'Authorised',
    'Authroised': 'Authorised',
    'Goverment': 'Government',
    'Governmnet': 'Government',
    'Departmant': 'Department',
    'Deparment': 'Department',
    'Acconut': 'Account',
    'Acocunt': 'Account',
    'Distribtuion': 'Distribution',
    'Ditribution': 'Distribution',
    'Regulatons': 'Regulations',
    'Regulaitons': 'Regulations',
    'oF': 'of',
    'lnvoice': 'Invoice',
    'Emiratos': 'Emirates',
    'Arnount': 'Amount',
    'Nurnber': 'Number',

    # Bank names
    'National Bank oF': 'National Bank of',
    'Sree code': 'Swift Code',

    # Common typos
    'G.OVT': 'GOVT',
    'G O V T': 'GOVT',
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


def apply_all_corrections(text: str) -> str:
    """
    Apply all corrections in order.
    """
    original = text

    # 1. Arabic corrections
    text = apply_arabic_corrections(text)

    # 2. English corrections
    text = apply_english_corrections(text)

    # 3. Number normalization
    text = normalize_numbers_in_fields(text)

    # 4. Table alignment
    text = fix_table_alignment(text)

    if text != original:
        diff = sum(1 for a, b in zip(text, original) if a != b)
        logger.info(f"📝 Dictionary corrections: {diff} chars changed")

    return text