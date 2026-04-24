"""
Enhanced post-processing pipeline for maximum OCR accuracy.

Improvements over v1:
- Better hallucination detection (more languages, more patterns)
- Smarter repetition removal (preserves legitimate repeated content)
- Context-aware cleaning
- Improved quality validation with detailed scoring
- Arabic-specific text normalization
- Number and date format normalization
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
        logger.warning(f"🔄 Removed {removed} repeated lines")
    return '\n'.join(cleaned)


def remove_prefix_loops(text: str, prefix_len: int = 10, max_consecutive: int = 5) -> str:
    """
    Detect and remove hallucination loops where the model generates many
    consecutive lines that share the same prefix (e.g., "الى جانب الشر...").

    This catches patterns like:
        الى جانب التحرير
        الى جانب التوفيق
        الى جانب التبرع
        الى جانب الامتياز
        ... (hundreds more)

    Also catches numbered list hallucinations:
        No. 31
        No. 32
        No. 33
        ... (dozens more)

    These are model hallucinations — real documents don't have 5+ consecutive
    lines all starting with the same phrase or pattern.
    """
    lines = text.split('\n')
    if len(lines) < max_consecutive:
        return text

    cleaned = []
    i = 0
    total_removed = 0

    while i < len(lines):
        # Check if we're entering a prefix loop
        if i + max_consecutive <= len(lines):
            current_stripped = lines[i].strip()

            # === Check 1: Numbered list pattern (e.g., "No. 31", "No. 32") ===
            numbered_match = re.match(r'^(No\.?\s*\d+)', current_stripped, re.IGNORECASE)
            if numbered_match:
                # Count consecutive numbered lines matching "No. XX" pattern
                loop_count = 0
                j = i
                while j < len(lines):
                    if re.match(r'^No\.?\s*\d+', lines[j].strip(), re.IGNORECASE):
                        loop_count += 1
                        j += 1
                        continue
                    break

                if loop_count >= max_consecutive:
                    logger.warning(
                        f"🔄 Removed numbered list loop: 'No. XX' "
                        f"×{loop_count} consecutive lines"
                    )
                    total_removed += loop_count
                    i = j
                    continue

            # === Check 2: Prefix-based loop (e.g., "الى جانب الشر...") ===
            if len(current_stripped) >= prefix_len:
                current_prefix = _normalize_prefix(current_stripped[:prefix_len])

                # Count how many consecutive lines share this prefix
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
                    # Prefix changed — check if it's a sub-prefix loop
                    # (e.g., "الى جانب الشر" → "الى جانب السع")
                    if len(line_stripped) >= prefix_len:
                        line_prefix = _normalize_prefix(line_stripped[:prefix_len])
                        # Check if still sharing a shorter prefix (6 chars)
                        short_current = _normalize_prefix(current_stripped[:6])
                        short_line = _normalize_prefix(line_stripped[:6])
                        if short_current == short_line and short_current:
                            loop_count += 1
                            j += 1
                            continue
                    break

                if loop_count >= max_consecutive:
                    # This is a hallucination loop — skip all of it
                    logger.warning(
                        f"🔄 Removed prefix loop: '{current_stripped[:30]}...' "
                        f"×{loop_count} consecutive lines"
                    )
                    total_removed += loop_count
                    i = j
                    continue

        cleaned.append(lines[i])
        i += 1

    if total_removed > 0:
        logger.info(f"🔄 Total hallucination-loop lines removed: {total_removed}")

    return '\n'.join(cleaned)


def _normalize_prefix(s: str) -> str:
    """Normalize a string prefix for comparison."""
    s = s.strip().lower()
    # Remove Arabic diacritics
    s = re.sub(r'[\u0617-\u061A\u064B-\u0652]', '', s)
    # Normalize Arabic variants
    s = s.replace('أ', 'ا').replace('إ', 'ا').replace('آ', 'ا')
    s = s.replace('ة', 'ه').replace('ى', 'ي')
    # Collapse whitespace
    s = re.sub(r'\s+', ' ', s).strip()
    return s


def remove_near_duplicates(text: str, max_repeats: int = 2) -> str:
    """
    Remove near-duplicate lines (same content after normalization).
    Uses Arabic-aware normalization for comparison.
    """
    lines = text.split('\n')
    cleaned = []
    seen = {}

    for line in lines:
        s = line.strip()
        if not s:
            cleaned.append(line)
            continue

        norm = s.lower()
        # Remove Arabic diacritics for comparison
        norm = re.sub(r'[\u0617-\u061A\u064B-\u0652]', '', norm)
        # Normalize Arabic variants
        norm = norm.replace('أ', 'ا').replace('إ', 'ا').replace('آ', 'ا')
        norm = norm.replace('ة', 'ه').replace('ى', 'ي')
        # Remove extra whitespace
        norm = re.sub(r'\s+', ' ', norm).strip()
        # Remove punctuation for comparison
        norm = re.sub(r'[.,;:\-–—/()؟!،]', '', norm)

        c = seen.get(norm, 0)
        seen[norm] = c + 1
        if c < max_repeats:
            cleaned.append(line)

    removed = len(lines) - len(cleaned)
    if removed > 0:
        logger.warning(f"🔄 Removed {removed} near-duplicates")
    return '\n'.join(cleaned)


def remove_hallucinated_languages(text: str) -> str:
    """
    Remove non-Arabic/English text that the model hallucinated.
    Expanded to cover more scripts and patterns.
    """
    # Script patterns to remove (not expected in Arabic/English docs)
    script_patterns = [
        ("CJK", r'[\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ff\uac00-\ud7af]+'),
        ("Hindi", r'[\u0900-\u097f]+'),
        ("Cyrillic", r'[\u0400-\u04ff]+'),
        ("Thai", r'[\u0e00-\u0e7f]+'),
        ("Korean", r'[\uac00-\ud7af]+'),
        ("Georgian", r'[\u10a0-\u10ff]+'),
        ("Armenian", r'[\u0530-\u058f]+'),
        ("Bengali", r'[\u0980-\u09ff]+'),
        ("Tamil", r'[\u0b80-\u0bff]+'),
        ("Gujarati", r'[\u0a80-\u0aff]+'),
    ]

    for name, pattern in script_patterns:
        regex = re.compile(pattern)
        matches = regex.findall(text)
        if matches:
            logger.warning(f"🚨 Removed {name}: {matches[:3]}")
            text = regex.sub('', text)

    return text


def remove_code_garbage(text: str) -> str:
    """
    Remove code-like artifacts that VLMs sometimes generate.
    Expanded patterns for better coverage.
    """
    # JavaScript-like code
    text = re.sub(r'(?:function|const|var|let|async|await|return|import|export|class)\s*[\(\{].*?[\)\}];?', '', text, flags=re.DOTALL)
    
    # HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    
    # Console/browser objects
    text = re.sub(r'(?:console|window|document|process)\.[a-zA-Z]+\(.*?\)', '', text)
    
    # JSON-like artifacts (but not if it looks like actual data)
    text = re.sub(r'(?:undefined|null|NaN|Infinity)\b', '', text)
    
    # Python-like code
    text = re.sub(r'(?:def|class|import|from|print)\s+[\w\.]+\s*[\(\:].*?[\)\:]', '', text, flags=re.DOTALL)
    
    # Markdown artifacts
    text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)  # Links → text
    
    # URL artifacts
    text = re.sub(r'https?://\S+', '', text)
    
    return text


def remove_model_hallucinations(text: str) -> str:
    """
    Remove common VLM hallucination patterns where the model
    explains itself instead of just extracting text.
    """
    hallucination_patterns = [
        # English hallucinations
        r'I\s+(?:cannot|can\'t|am unable to)\s+(?:read|see|extract|identify|determine).*?(?:\n|$)',
        r'(?:The|This)\s+(?:image|document|text)\s+(?:appears to|seems to|looks like).*?(?:\n|$)',
        r'(?:Please|Kindly)\s+(?:note|be advised).*?(?:\n|$)',
        r'(?:Note|Disclaimer|Warning|Caution)\s*:\s*(?:This|The).*?(?:\n|$)',
        r'I\s+(?:would|will)\s+(?:be happy|glad|pleased).*?(?:\n|$)',
        r'(?:Let me|Allow me|Let\'s)\s+(?:know|help|explain).*?(?:\n|$)',
        r'(?:Here|Below|Following)\s+(?:is|are)\s+(?:the|my).*?(?:extracted|OCR|result).*?(?:\n|$)',
        r'(?:Based on|From|According to)\s+(?:what|the)\s+I\s+(?:can|am).*?(?:\n|$)',
        
        # Arabic hallucinations
        r'لا\s*(?:أستطيع|يمكنني|أقدر)\s*(?:قراءة|رؤية|استخراج).*?(?:\n|$)',
        r'(?:هذه|هذا)\s*(?:الصورة|المستند|النص)\s*(?:يبدو|تبدو|يظهر).*?(?:\n|$)',
        r'(?:يرجى|رجاء)\s*(?:ملاحظة|الانتباه).*?(?:\n|$)',
        r'(?:ملاحظة|تنبيه|تحذير)\s*:\s*(?:هذا|هذه).*?(?:\n|$)',
        r'سأقوم\s*(?:بـ|باستخراج).*?(?:\n|$)',
    ]
    
    for pattern in hallucination_patterns:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE)
    
    return text


def normalize_arabic_text(text: str) -> str:
    """
    Normalize Arabic text for consistency without changing meaning.
    - Normalize alef variants
    - Normalize taa marbuta
    - Remove tatweel (kashida)
    - Normalize whitespace
    """
    # Remove tatweel (kashida) — elongation character
    text = text.replace('\u0640', '')
    
    # Normalize alef variants to bare alef (for consistency)
    # Only in specific contexts where it doesn't change meaning
    # We DON'T normalize أ/إ/آ → ا because they carry meaning
    
    # Fix common Arabic punctuation issues
    text = text.replace('،،', '،')
    text = text.replace('..', '…')
    text = text.replace('؟؟', '؟')
    
    # Normalize Arabic comma
    text = text.replace(',', '،')
    # But keep commas in numbers: 1,000.00
    text = re.sub(r'(\d)،(\d{3})', r'\1,\2', text)
    
    return text


def fix_number_formats(text: str) -> str:
    """
    Normalize number formats for consistency.
    - Fix broken decimal numbers
    - Normalize amount formats
    - Fix date formats
    """
    # Fix spaces in numbers: "1 000" → "1,000"
    text = re.sub(r'(\d)\s+(\d{3})', r'\1,\2', text)
    
    # Fix broken decimals: "1. 00" → "1.00"
    text = re.sub(r'(\d)\.\s+(\d{2})', r'\1.\2', text)
    
    # Fix comma-decimal confusion in amounts: "1,00" at end of line → "1.00"
    # Only when it looks like a decimal (2 digits after comma)
    text = re.sub(r'(\d+),(\d{2})\s*$', r'\1.\2', text, flags=re.MULTILINE)
    
    # Normalize date separators
    text = re.sub(r'(\d{1,4})\s*[-–—]\s*(\d{1,2})\s*[-–—]\s*(\d{1,4})', r'\1-\2-\3', text)
    
    return text


def clean_ocr_text(text: str) -> str:
    """
    Full enhanced cleaning pipeline:
    1. Remove hallucinated languages
    2. Remove model hallucinations (explanations, disclaimers)
    3. Remove code garbage
    4. Remove repetitions
    5. Apply dictionary corrections
    6. Normalize Arabic text
    7. Fix number formats
    8. Clean formatting
    """
    original_len = len(text)

    # 1. Remove hallucinations (non-Arabic/English scripts)
    text = remove_hallucinated_languages(text)

    # 2. Remove model hallucinations (explanations, disclaimers)
    text = remove_model_hallucinations(text)

    # 3. Remove code
    text = remove_code_garbage(text)

    # 4. Remove repetitions
    text = remove_repetitions(text, max_repeats=2)
    text = remove_near_duplicates(text, max_repeats=2)
    text = remove_prefix_loops(text, prefix_len=10, max_consecutive=5)

    # 5. Apply dictionary corrections (major accuracy boost!)
    text = apply_all_corrections(text)

    # 6. Normalize Arabic text
    text = normalize_arabic_text(text)

    # 7. Fix number formats
    text = fix_number_formats(text)

    # 8. Clean formatting
    text = re.sub(r'\*{5,}', '', text)
    text = re.sub(r'\.{5,}', '...', text)
    text = re.sub(r'\-{5,}', '—', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r' {3,}', ' ', text)
    text = re.sub(r'\t{2,}', '\t', text)

    # Remove empty lines but preserve paragraph breaks
    lines = [l for l in text.split('\n') if l.strip()]
    text = '\n'.join(lines)

    cleaned_len = len(text)
    if original_len != cleaned_len:
        pct = round((1 - cleaned_len / max(original_len, 1)) * 100, 1)
        logger.info(f"🧹 {original_len} → {cleaned_len} ({pct}% cleaned)")

    return text.strip()


def validate_ocr_output(text: str, file_name: str = "") -> dict:
    """
    Validate OCR output quality with detailed scoring.
    Returns quality level, issues, and statistics.
    """
    issues = []

    if len(text) < 50:
        issues.append("Very short output")

    lines = [l.strip() for l in text.split('\n') if l.strip()]
    
    # Check for repetition
    if lines:
        counts = Counter(lines)
        top, top_count = counts.most_common(1)[0]
        if top_count > 3:
            issues.append(f"Repetition: '{top[:40]}' x{top_count}")

    # Check for code remnants
    code_words = ["function", "console", "const ", "<script", "var ", "import "]
    if any(w in text.lower() for w in code_words):
        issues.append("Code remnants")

    # Check for hallucination markers
    hallucination_markers = [
        "I cannot", "I am unable", "I can't", "لا أستطيع",
        "As an AI", "I don't have", "I'm not able",
        "appears to be", "seems to be", "looks like",
    ]
    text_lower = text.lower()
    for marker in hallucination_markers:
        if marker.lower() in text_lower:
            issues.append("Hallucination detected")
            break

    # Character analysis
    arabic = len(re.findall(r'[\u0600-\u06ff]', text))
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
    
    # Number presence score (0-10) — documents usually have numbers
    if digits > 0:
        quality_score += min(digits, 10)
    
    # Language detection score (0-15)
    if arabic > 0 and english > 0:
        quality_score += 15  # Mixed document — good sign
    elif arabic > 0:
        quality_score += 10  # Arabic document
    elif english > 0:
        quality_score += 10  # English document
    
    # Penalty for issues (0-15)
    quality_score -= len(issues) * 5
    
    # Determine quality level
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
