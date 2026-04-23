"""
Enhanced post-processing pipeline.
"""
import re
from collections import Counter
from loguru import logger

from arabic_corrections import apply_all_corrections


def remove_repetitions(text: str, max_repeats: int = 2) -> str:
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


def remove_near_duplicates(text: str, max_repeats: int = 2) -> str:
    lines = text.split('\n')
    cleaned = []
    seen = {}

    for line in lines:
        s = line.strip()
        if not s:
            cleaned.append(line)
            continue

        norm = s.lower()
        norm = re.sub(r'[\u0617-\u061A\u064B-\u0652]', '', norm)
        norm = norm.replace('أ', 'ا').replace('إ', 'ا').replace('آ', 'ا')
        norm = norm.replace('ة', 'ه').replace('ى', 'ي')
        norm = re.sub(r'\s+', ' ', norm).strip()

        c = seen.get(norm, 0)
        seen[norm] = c + 1
        if c < max_repeats:
            cleaned.append(line)

    removed = len(lines) - len(cleaned)
    if removed > 0:
        logger.warning(f"🔄 Removed {removed} near-duplicates")
    return '\n'.join(cleaned)


def remove_hallucinated_languages(text: str) -> str:
    """Remove non-Arabic/English text."""
    for name, pattern in [
        ("CJK", r'[\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ff\uac00-\ud7af]+'),
        ("Hindi", r'[\u0900-\u097f]+'),
        ("Cyrillic", r'[\u0400-\u04ff]+'),
        ("Thai", r'[\u0e00-\u0e7f]+'),
    ]:
        regex = re.compile(pattern)
        matches = regex.findall(text)
        if matches:
            logger.warning(f"🚨 Removed {name}: {matches[:3]}")
            text = regex.sub('', text)
    return text


def remove_code_garbage(text: str) -> str:
    text = re.sub(r'(?:function|const|var|let|async|await|return|import)\s*[\(\{].*?[\)\}];?', '', text, flags=re.DOTALL)
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'(?:console|window|document)\.[a-zA-Z]+\(.*?\)', '', text)
    return text


def clean_ocr_text(text: str) -> str:
    """
    Full enhanced cleaning pipeline:
    1. Remove hallucinated languages
    2. Remove code garbage
    3. Remove repetitions
    4. Apply dictionary corrections ← NEW
    5. Clean formatting
    """
    original_len = len(text)

    # 1. Remove hallucinations
    text = remove_hallucinated_languages(text)

    # 2. Remove code
    text = remove_code_garbage(text)

    # 3. Remove repetitions
    text = remove_repetitions(text, max_repeats=2)
    text = remove_near_duplicates(text, max_repeats=2)

    # 4. Apply dictionary corrections (NEW - major accuracy boost!)
    text = apply_all_corrections(text)

    # 5. Clean formatting
    text = re.sub(r'\*{5,}', '', text)
    text = re.sub(r'\.{5,}', '...', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r' {3,}', ' ', text)

    lines = [l for l in text.split('\n') if l.strip()]
    text = '\n'.join(lines)

    cleaned_len = len(text)
    if original_len != cleaned_len:
        pct = round((1 - cleaned_len / max(original_len, 1)) * 100, 1)
        logger.info(f"🧹 {original_len} → {cleaned_len} ({pct}% cleaned)")

    return text.strip()


def validate_ocr_output(text: str, file_name: str = "") -> dict:
    issues = []

    if len(text) < 50:
        issues.append("Very short output")

    lines = [l.strip() for l in text.split('\n') if l.strip()]
    if lines:
        counts = Counter(lines)
        top, top_count = counts.most_common(1)[0]
        if top_count > 3:
            issues.append(f"Repetition: '{top[:40]}' x{top_count}")

    code_words = ["function", "console", "const ", "<script"]
    if any(w in text.lower() for w in code_words):
        issues.append("Code remnants")

    arabic = len(re.findall(r'[\u0600-\u06ff]', text))
    english = len(re.findall(r'[a-zA-Z]', text))
    total = arabic + english

    quality = "good"
    if issues:
        quality = "needs_review"
    if len(issues) > 2:
        quality = "poor"

    return {
        "file_name": file_name,
        "quality": quality,
        "issues": issues,
        "char_count": len(text),
        "arabic_chars": arabic,
        "english_chars": english,
        "arabic_ratio": round(arabic / max(total, 1) * 100, 1),
        "line_count": len(lines),
    }