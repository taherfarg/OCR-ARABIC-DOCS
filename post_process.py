"""
Post-processing pipeline for OCR accuracy.

Cleans OCR output by removing hallucinations, repetitions, and artifacts
while preserving legitimate document content.
"""
import re
from collections import Counter
from loguru import logger

from arabic_corrections import apply_all_corrections


def remove_repetitions(text: str, max_repeats: int = 2) -> str:
    """
    Remove consecutive repeated lines.
    Preserves legitimate repeated content (like table headers) that
    appear with other content between them.
    """
    lines = text.split('\n')
    cleaned = []
    prev = None
    count = 0

    for line in lines:
        s = line.strip()
        if s == prev and s:
            count += 1
            if count <= max_repeats:
                cleaned.append(line)
        else:
            count = 0
            cleaned.append(line)
            prev = s

    removed = len(lines) - len(cleaned)
    if removed > 0:
        logger.warning(f"Removed {removed} repeated lines")
    return '\n'.join(cleaned)


def remove_prefix_loops(text: str, prefix_len: int = 15, max_consecutive: int = 8) -> str:
    """
    Detect and remove hallucination loops where the model generates many
    consecutive lines that share the same long prefix.

    Uses prefix_len=15 and max_consecutive=8 to avoid false positives
    on legal article sequences (which share short prefixes like "المادة").
    """
    lines = text.split('\n')
    if len(lines) < max_consecutive:
        return text

    cleaned = []
    i = 0
    total_removed = 0

    while i < len(lines):
        if i + max_consecutive <= len(lines):
            current_stripped = lines[i].strip()

            # Numbered list hallucination (e.g., "No. 31", "No. 32", ...)
            if re.match(r'^No\.?\s*\d+\s*$', current_stripped, re.IGNORECASE):
                loop_count = 0
                j = i
                while j < len(lines):
                    if re.match(r'^No\.?\s*\d+\s*$', lines[j].strip(), re.IGNORECASE):
                        loop_count += 1
                        j += 1
                    else:
                        break

                if loop_count >= max_consecutive:
                    logger.warning(
                        f"Removed numbered list loop: 'No. XX' x{loop_count}"
                    )
                    total_removed += loop_count
                    i = j
                    continue

            # Prefix-based loop detection (long prefix only)
            if len(current_stripped) >= prefix_len:
                current_prefix = _normalize_prefix(current_stripped[:prefix_len])

                loop_count = 0
                j = i
                while j < len(lines):
                    line_stripped = lines[j].strip()
                    if len(line_stripped) >= prefix_len:
                        line_prefix = _normalize_prefix(line_stripped[:prefix_len])
                        if line_prefix == current_prefix:
                            loop_count += 1
                            j += 1
                            continue
                    break

                if loop_count >= max_consecutive:
                    logger.warning(
                        f"Removed prefix loop: '{current_stripped[:30]}...' "
                        f"x{loop_count} consecutive lines"
                    )
                    total_removed += loop_count
                    i = j
                    continue

        cleaned.append(lines[i])
        i += 1

    if total_removed > 0:
        logger.info(f"Total hallucination-loop lines removed: {total_removed}")

    return '\n'.join(cleaned)


def _normalize_prefix(s: str) -> str:
    """Normalize a string prefix for comparison."""
    s = s.strip().lower()
    # Remove Arabic diacritics
    s = re.sub(r'[ؗ-ًؚ-ْ]', '', s)
    # Normalize Arabic variants
    s = s.replace('أ', 'ا').replace('إ', 'ا').replace('آ', 'ا')
    s = s.replace('ة', 'ه').replace('ى', 'ي')
    # Collapse whitespace
    s = re.sub(r'\s+', ' ', s).strip()
    return s


def remove_near_duplicates(text: str, max_repeats: int = 3) -> str:
    """
    Remove near-duplicate lines (same content after normalization).
    Uses max_repeats=3 to avoid removing legitimate legal article patterns.
    Only normalizes diacritics and whitespace, keeps punctuation to reduce
    false matches.
    """
    lines = text.split('\n')
    cleaned = []
    seen = {}

    for line in lines:
        s = line.strip()
        if not s:
            cleaned.append(line)
            continue

        # Only normalize diacritics and whitespace for comparison,
        # keep punctuation and numbers to distinguish similar lines
        norm = s.lower()
        norm = re.sub(r'[ؗ-ًؚ-ْ]', '', norm)
        norm = norm.replace('أ', 'ا').replace('إ', 'ا').replace('آ', 'ا')
        norm = norm.replace('ة', 'ه').replace('ى', 'ي')
        norm = re.sub(r'\s+', ' ', norm).strip()

        c = seen.get(norm, 0)
        seen[norm] = c + 1
        if c < max_repeats:
            cleaned.append(line)

    removed = len(lines) - len(cleaned)
    if removed > 0:
        logger.warning(f"Removed {removed} near-duplicates")
    return '\n'.join(cleaned)


def remove_hallucinated_languages(text: str) -> str:
    """Remove non-Arabic/English text that the model hallucinated."""
    script_patterns = [
        ("CJK", r'[一-鿿぀-ゟ゠-ヿ]+'),
        ("Hindi", r'[ऀ-ॿ]+'),
        ("Cyrillic", r'[Ѐ-ӿ]+'),
        ("Thai", r'[฀-๿]+'),
        ("Korean", r'[가-힯]+'),
        ("Georgian", r'[Ⴀ-ჿ]+'),
        ("Armenian", r'[԰-֏]+'),
        ("Bengali", r'[ঀ-৿]+'),
        ("Tamil", r'[஀-௿]+'),
        ("Gujarati", r'[઀-૿]+'),
    ]

    for name, pattern in script_patterns:
        regex = re.compile(pattern)
        matches = regex.findall(text)
        if matches:
            logger.warning(f"Removed {name} script: {len(matches)} occurrences")
            text = regex.sub('', text)

    return text


def remove_code_garbage(text: str) -> str:
    """
    Remove code-like artifacts that VLMs sometimes generate.
    Uses conservative patterns to avoid removing legitimate document content.
    Words like "import", "export", "class" appear in legal/financial docs.
    """
    # Only match full code statements (require semicolons or braces)
    text = re.sub(r'(?:function|const|var|let)\s+\w+\s*\(.*?\)\s*\{.*?\}', '', text)

    # HTML tags
    text = re.sub(r'<(?:script|style|div|span|html|body|head|meta|link)[^>]*>.*?</(?:script|style|div|span|html|body|head|meta|link)>', '', text, flags=re.DOTALL)

    # Console/browser objects
    text = re.sub(r'(?:console|window|document)\.\w+\(.*?\)', '', text)

    # JSON-like artifacts
    text = re.sub(r'\b(?:undefined|NaN|Infinity)\b', '', text)

    # Markdown link syntax (preserve link text)
    text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)

    # URL artifacts
    text = re.sub(r'https?://\S+', '', text)

    return text


def remove_model_hallucinations(text: str) -> str:
    """
    Remove common VLM hallucination patterns where the model
    explains itself instead of just extracting text.
    Patterns require start-of-line to avoid false matches in document text.
    """
    hallucination_patterns = [
        # English hallucinations (start-of-line only)
        r'^I\s+(?:cannot|can\'t|am unable to)\s+(?:read|see|extract|identify|determine).*$',
        r'^(?:Please|Kindly)\s+(?:note|be advised).*$',
        r'^(?:Note|Disclaimer|Warning|Caution)\s*:\s*(?:This|The)\s+(?:image|document|text).*$',
        r'^I\s+(?:would|will)\s+(?:be happy|glad|pleased).*$',
        r'^(?:Let me|Allow me|Let\'s)\s+(?:know|help|explain).*$',
        r'^(?:Here|Below|Following)\s+(?:is|are)\s+(?:the|my)\s+(?:extracted|OCR|transcri).*$',
        r'^(?:Based on|From|According to)\s+(?:what|the)\s+I\s+(?:can|am).*$',
        r'^As an AI.*$',
        # Arabic hallucinations (start-of-line only)
        r'^لا\s*(?:أستطيع|يمكنني|أقدر)\s*(?:قراءة|رؤية|استخراج).*$',
        r'^سأقوم\s*(?:بـ|باستخراج).*$',
    ]

    for pattern in hallucination_patterns:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE | re.MULTILINE)

    return text


def normalize_arabic_text(text: str) -> str:
    """
    Normalize Arabic text for consistency without changing meaning.
    Only removes kashida and fixes doubled punctuation.
    Does NOT replace commas globally (that corrupts English text and numbers).
    """
    # Remove tatweel (kashida)
    text = text.replace('ـ', '')

    # Fix doubled Arabic punctuation
    text = text.replace('،،', '،')
    text = text.replace('؟؟', '؟')

    # Replace commas with Arabic commas ONLY between Arabic characters
    text = re.sub(r'([؀-ۿ]),(\s*[؀-ۿ])', r'\1،\2', text)

    return text


def fix_number_formats(text: str) -> str:
    """
    Fix broken number formats from OCR errors.
    Conservative: only fix clearly broken patterns.
    """
    # Fix broken decimals: "1. 00" -> "1.00"
    text = re.sub(r'(\d)\.\s+(\d{2})\b', r'\1.\2', text)

    # Normalize date separators with spaces
    text = re.sub(r'(\d{1,4})\s*[-–—]\s*(\d{1,2})\s*[-–—]\s*(\d{1,4})', r'\1-\2-\3', text)

    return text


def clean_ocr_text(text: str) -> str:
    """
    Full cleaning pipeline. Order matters.
    """
    original_len = len(text)

    # 1. Remove hallucinated scripts (CJK, Hindi, etc.)
    text = remove_hallucinated_languages(text)

    # 2. Remove model hallucinations (AI explanations)
    text = remove_model_hallucinations(text)

    # 3. Remove code artifacts
    text = remove_code_garbage(text)

    # 4. Remove repetitions
    text = remove_repetitions(text, max_repeats=2)
    text = remove_near_duplicates(text, max_repeats=3)
    text = remove_prefix_loops(text, prefix_len=15, max_consecutive=8)

    # 5. Apply dictionary corrections
    text = apply_all_corrections(text)

    # 6. Normalize Arabic text
    text = normalize_arabic_text(text)

    # 7. Fix number formats
    text = fix_number_formats(text)

    # 8. Clean formatting (conservative)
    text = re.sub(r'\*{5,}', '', text)
    text = re.sub(r'\.{5,}', '...', text)
    text = re.sub(r'-{5,}', '—', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'\t{2,}', '\t', text)

    # Remove empty lines but preserve paragraph breaks
    lines = [l for l in text.split('\n') if l.strip()]
    text = '\n'.join(lines)

    cleaned_len = len(text)
    if original_len != cleaned_len:
        pct = round((1 - cleaned_len / max(original_len, 1)) * 100, 1)
        logger.info(f"Cleaned: {original_len} -> {cleaned_len} ({pct}% removed)")

    return text.strip()


def validate_ocr_output(text: str, file_name: str = "") -> dict:
    """Validate OCR output quality with detailed scoring."""
    issues = []

    if len(text) < 30:
        issues.append("Very short output")

    lines = [l.strip() for l in text.split('\n') if l.strip()]

    # Check for repetition
    if lines:
        counts = Counter(lines)
        top, top_count = counts.most_common(1)[0]
        if top_count > 3:
            issues.append(f"Repetition: '{top[:40]}' x{top_count}")

    # Check for hallucination markers
    hallucination_markers = [
        "I cannot", "I am unable", "I can't",
        "As an AI", "I don't have", "I'm not able",
    ]
    text_lower = text.lower()
    for marker in hallucination_markers:
        if marker.lower() in text_lower:
            issues.append("Hallucination detected")
            break

    # Character analysis
    arabic = len(re.findall(r'[؀-ۿ]', text))
    english = len(re.findall(r'[a-zA-Z]', text))
    digits = len(re.findall(r'\d', text))
    total_alpha = arabic + english

    # Quality scoring
    quality_score = 0

    # Length score (0-25)
    if len(text) > 500:
        quality_score += 25
    elif len(text) > 200:
        quality_score += 20
    elif len(text) > 100:
        quality_score += 15
    elif len(text) > 50:
        quality_score += 10

    # Line count score (0-15)
    if len(lines) > 10:
        quality_score += 15
    elif len(lines) > 5:
        quality_score += 10
    elif len(lines) > 3:
        quality_score += 5

    # Content diversity score (0-20)
    if total_alpha > 0:
        unique_words = len(set(text.split()))
        total_words = len(text.split())
        if total_words > 0:
            diversity = unique_words / total_words
            quality_score += int(diversity * 20)

    # Number presence score (0-10)
    if digits > 0:
        quality_score += min(digits, 10)

    # Language detection score (0-15)
    if arabic > 0 and english > 0:
        quality_score += 15
    elif arabic > 0:
        quality_score += 10
    elif english > 0:
        quality_score += 10

    # Penalty for issues
    quality_score -= len(issues) * 5

    if quality_score >= 60 and not issues:
        quality = "excellent"
    elif quality_score >= 40 and len(issues) <= 1:
        quality = "good"
    elif quality_score >= 20:
        quality = "needs_review"
    else:
        quality = "poor"

    return {
        "file_name": file_name,
        "quality": quality,
        "quality_score": max(0, quality_score),
        "issues": issues,
        "char_count": len(text),
        "arabic_chars": arabic,
        "english_chars": english,
        "digit_chars": digits,
        "arabic_ratio": round(arabic / max(total_alpha, 1) * 100, 1),
        "line_count": len(lines),
    }
