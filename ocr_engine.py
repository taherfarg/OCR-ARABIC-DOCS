"""
OCR engine with maximum accuracy improvements.

Key features:
- Multi-pass OCR with diverse prompts (Arabic, English, Structured)
- Quality-gated retry: re-process if output is poor
- Consensus merge with monotonic alignment and best-line fallback
- Arabic-specific line scoring heuristics
"""
import gc
import json
import time
import re
from typing import Union, Optional, List, Tuple
from pathlib import Path
from collections import Counter

import torch
import json_repair
from PIL import Image
from loguru import logger
from transformers import StoppingCriteria, StoppingCriteriaList

import config
from model_loader import GemmaModelManager, QwenModelManager
from advanced_preprocess import full_preprocess_pipeline, light_preprocess_vlm
from preprocess import get_first_page_image
from post_process import clean_ocr_text, validate_ocr_output


class SafeStoppingCriteria(StoppingCriteria):
    """
    Combined timeout + progress logging stopping criteria.
    Prevents generation from hanging indefinitely.
    """
    def __init__(self, max_seconds: int = 300, log_interval: int = 30):
        self.start_time = time.time()
        self.max_seconds = max_seconds
        self.log_interval = log_interval
        self.last_log_time = self.start_time
        self._token_count = 0

    def __call__(self, input_ids, scores, **kwargs):
        self._token_count = input_ids.shape[-1]
        now = time.time()
        elapsed = now - self.start_time

        # Progress logging — so user knows it's not stuck
        if now - self.last_log_time >= self.log_interval:
            logger.info(f"  ⏳ Generating... {self._token_count} tokens, {elapsed:.0f}s elapsed")
            self.last_log_time = now

        # Timeout check
        if elapsed > self.max_seconds:
            logger.warning(f"⚠️ Generation timeout after {elapsed:.0f}s ({self._token_count} tokens), stopping")
            return True

        return False


def process_vision_info_qwen(messages):
    """Extract images from Qwen-format messages."""
    images = []
    for msg in messages:
        if isinstance(msg["content"], list):
            for item in msg["content"]:
                if item["type"] == "image":
                    img = item["image"]
                    if isinstance(img, str):
                        img = Image.open(img).convert("RGB")
                    elif isinstance(img, Image.Image):
                        img = img.convert("RGB")
                    images.append(img)
    return images if images else None, []


class OCREngine:

    def __init__(self):
        self.mode = config.PROCESSING_MODE
        logger.info(f"🔧 OCR Engine mode: {self.mode}")

    # ===== GEMMA-3: Classification =====

    def _run_gemma(self, image: Image.Image) -> dict:
        model, processor = GemmaModelManager.get_model()

        # Preprocess for Gemma
        processed = full_preprocess_pipeline(image, for_model="gemma", max_width=1024)

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": processed},
                    {"type": "text", "text": config.CLASSIFIER_PROMPT},
                ]
            }
        ]

        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(model.device)

        input_len = inputs["input_ids"].shape[-1]

        # Safety: timeout + progress logging
        stopping = StoppingCriteriaList([SafeStoppingCriteria(
            max_seconds=config.GENERATION_TIMEOUT_SEC,
        )])

        try:
            with torch.inference_mode():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=config.GEMMA_MAX_TOKENS,
                    do_sample=True,
                    temperature=config.GEMMA_TEMPERATURE,
                    top_p=config.GEMMA_TOP_P,
                    top_k=config.GEMMA_TOP_K,
                    repetition_penalty=config.GEMMA_REPETITION_PENALTY,
                    no_repeat_ngram_size=config.NO_REPEAT_NGRAM_SIZE,
                    stopping_criteria=stopping,
                )
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logger.error("⚠️ CUDA OOM during Gemma generation!")
                torch.cuda.empty_cache()
                gc.collect()
                return {"raw_text": "OOM_ERROR"}
            raise

        raw = processor.decode(outputs[0][input_len:], skip_special_tokens=True).strip()
        logger.info(f"  📝 Gemma generated {len(raw)} chars")

        del inputs, outputs
        gc.collect()
        torch.cuda.empty_cache()

        # Try to parse JSON from the response
        return self._parse_json_response(raw)

    def _parse_json_response(self, raw: str) -> dict:
        """Robustly parse JSON from model response, handling various formats."""
        # Try direct parse
        try:
            data = json.loads(raw)
            return data if isinstance(data, dict) else {"raw": str(data)}
        except json.JSONDecodeError:
            pass

        # Try to extract JSON from markdown code block
        json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', raw, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group(1))
                return data if isinstance(data, dict) else {"raw": str(data)}
            except json.JSONDecodeError:
                pass

        # Try to find first { ... } in text
        brace_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', raw, re.DOTALL)
        if brace_match:
            try:
                data = json.loads(brace_match.group(0))
                return data if isinstance(data, dict) else {"raw": str(data)}
            except json.JSONDecodeError:
                pass

        # Last resort: json_repair
        try:
            data = json_repair.loads(raw)
            return data if isinstance(data, dict) else {"raw_text": raw}
        except Exception:
            return {"raw_text": raw}

    # ===== QWEN: Single pass OCR =====

    def _run_qwen_single(self, image: Image.Image, prompt: str) -> str:
        model, processor = QwenModelManager.get_model()

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        text_input = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, _ = process_vision_info_qwen(messages)

        inputs = processor(
            text=[text_input],
            images=image_inputs,
            padding=True,
            return_tensors="pt",
        ).to(model.device)

        input_len = inputs.input_ids.shape[1]

        # Safety: timeout + progress logging
        stopping = StoppingCriteriaList([SafeStoppingCriteria(
            max_seconds=config.GENERATION_TIMEOUT_SEC,
        )])

        logger.info(f"  🚀 Starting generation (max {config.QWEN_MAX_TOKENS} tokens)...")
        try:
            with torch.inference_mode():
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=config.QWEN_MAX_TOKENS,
                    do_sample=config.QWEN_DO_SAMPLE,
                    repetition_penalty=config.QWEN_REPETITION_PENALTY,
                    no_repeat_ngram_size=config.NO_REPEAT_NGRAM_SIZE,
                    pad_token_id=processor.tokenizer.eos_token_id,
                    eos_token_id=processor.tokenizer.eos_token_id,
                    stopping_criteria=stopping,
                )
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logger.error("⚠️ CUDA OOM during Qwen generation!")
                torch.cuda.empty_cache()
                gc.collect()
                return "[OOM_ERROR - try reducing image size or disabling multi-pass]"
            raise

        gen_tokens = generated_ids.shape[-1] - input_len
        output = processor.batch_decode(
            generated_ids[:, input_len:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )[0].strip()

        logger.info(f"  ✅ Generated {gen_tokens} tokens → {len(output)} chars")

        del inputs, generated_ids
        gc.collect()
        torch.cuda.empty_cache()

        return output

    # ===== MULTI-PASS OCR =====

    def _run_qwen_multipass(self, original_image: Image.Image) -> str:
        """
        Run OCR with up to 3 passes using light preprocessing for VLMs.

        Pass 1: Arabic prompt
        Pass 2: English prompt
        Pass 3: Structured extraction prompt
        Then merge using consensus voting.
        """
        if not config.MULTIPASS_ENABLED:
            img = light_preprocess_vlm(original_image, max_width=2000)
            return self._run_qwen_single(img, config.OCR_PROMPT)

        num_passes = config.MULTIPASS_COUNT
        logger.info(f"🔄 Multi-pass OCR ({num_passes} passes)...")

        passes = []

        # ===== PASS 1: Arabic prompt =====
        logger.info(f"  Pass 1/{num_passes}: Arabic OCR...")
        img1 = light_preprocess_vlm(original_image, max_width=2000)
        text1 = self._run_qwen_single(img1, config.OCR_PROMPT)
        passes.append(("arabic", text1))

        # ===== PASS 2: English prompt =====
        if num_passes >= 2:
            logger.info(f"  Pass 2/{num_passes}: English OCR...")
            img2 = light_preprocess_vlm(original_image, max_width=2000)
            text2 = self._run_qwen_single(img2, config.OCR_PROMPT_EN)
            passes.append(("english", text2))

        # ===== PASS 3: Structured extraction =====
        if num_passes >= 3:
            logger.info(f"  Pass 3/{num_passes}: Structured extraction...")
            img3 = light_preprocess_vlm(original_image, max_width=2000)
            text3 = self._run_qwen_single(img3, config.OCR_PROMPT_STRUCTURED)
            passes.append(("structured", text3))

        # ===== MERGE: Consensus voting =====
        texts = [t for _, t in passes]
        merged = self._merge_ocr_results_consensus(*texts)

        for name, text in passes:
            logger.info(f"  {name}: {len(text)} chars")
        logger.info(f"  → Merged: {len(merged)} chars")

        return merged

    # ===== MERGE: Consensus Voting =====

    def _merge_ocr_results_consensus(self, *texts: str) -> str:
        """
        Merge multiple OCR passes using consensus voting.

        Strategy:
        1. Align lines across passes using monotonic fuzzy matching
        2. For each line position, if 2+ passes agree → use that
        3. If all disagree → pick the best scored line
        """
        if len(texts) == 1:
            return texts[0]

        all_lines = [t.split('\n') for t in texts]

        all_nonempty = []
        for lines in all_lines:
            nonempty = [(i, l) for i, l in enumerate(lines) if l.strip()]
            all_nonempty.append(nonempty)

        lengths = [len(ne) for ne in all_nonempty]
        base_idx = lengths.index(max(lengths))
        base_lines = [l for _, l in all_nonempty[base_idx]]

        if not base_lines:
            return ""

        aligned = [[] for _ in range(len(base_lines))]

        for pass_idx, nonempty in enumerate(all_nonempty):
            pass_lines = [l for _, l in nonempty]

            if pass_idx == base_idx:
                for i, line in enumerate(pass_lines):
                    aligned[i].append(line)
                continue

            # Monotonic alignment: enforce sequential order
            # pass line j matched to base line i means j+1.. can only match i+1..
            search_start = 0
            for i, base_line in enumerate(base_lines):
                best_match = None
                best_score = -1
                best_j = -1

                base_norm = self._normalize_for_match(base_line)

                for j in range(search_start, len(pass_lines)):
                    cand_norm = self._normalize_for_match(pass_lines[j])
                    score = self._line_similarity(base_norm, cand_norm)
                    if score > best_score:
                        best_score = score
                        best_match = pass_lines[j]
                        best_j = j

                if best_match and best_score > 0.3:
                    aligned[i].append(best_match)
                    search_start = best_j + 1

        merged_lines = []
        for i, candidates in enumerate(aligned):
            if not candidates:
                continue

            if len(candidates) == 1:
                merged_lines.append(candidates[0])
                continue

            consensus = self._find_consensus(candidates)
            if consensus:
                merged_lines.append(consensus)
            else:
                merged_lines.append(self._pick_best_line(candidates))

        return '\n'.join(merged_lines)

    def _normalize_for_match(self, line: str) -> str:
        """Normalize a line for fuzzy matching."""
        s = line.strip().lower()
        # Remove diacritics
        s = re.sub(r'[\u0617-\u061A\u064B-\u0652]', '', s)
        # Normalize Arabic variants
        s = s.replace('أ', 'ا').replace('إ', 'ا').replace('آ', 'ا')
        s = s.replace('ة', 'ه').replace('ى', 'ي')
        s = s.replace('ؤ', 'و').replace('ئ', 'ي')
        # Collapse whitespace
        s = re.sub(r'\s+', ' ', s).strip()
        return s

    def _line_similarity(self, a: str, b: str) -> float:
        """
        Compute similarity between two normalized lines.
        Uses bigram overlap (Dice coefficient) + word overlap + length similarity.
        """
        if not a or not b:
            return 0.0

        # Bigram (character pair) overlap — Dice coefficient
        def bigrams(s):
            return [s[i:i+2] for i in range(len(s)-1)] if len(s) >= 2 else [s]

        bg_a = bigrams(a)
        bg_b = bigrams(b)
        if bg_a and bg_b:
            set_a = set(bg_a)
            set_b = set(bg_b)
            intersection = len(set_a & set_b)
            bigram_sim = 2.0 * intersection / (len(set_a) + len(set_b))
        else:
            bigram_sim = 1.0 if a == b else 0.0

        # Word-level overlap (more weight)
        words_a = set(a.split())
        words_b = set(b.split())
        if words_a and words_b:
            w_intersection = len(words_a & words_b)
            w_union = len(words_a | words_b)
            word_sim = w_intersection / w_union if w_union else 0.0
        else:
            word_sim = 0.0

        # Length similarity
        len_sim = min(len(a), len(b)) / max(len(a), len(b), 1)

        return 0.35 * bigram_sim + 0.45 * word_sim + 0.20 * len_sim

    def _find_consensus(self, candidates: List[str]) -> Optional[str]:
        """
        Find if 2+ candidates agree (are very similar).
        If so, return the best version of the consensus text.
        """
        if len(candidates) < 2:
            return candidates[0] if candidates else None

        # Pairwise similarity
        groups = []
        assigned = [False] * len(candidates)

        for i in range(len(candidates)):
            if assigned[i]:
                continue
            group = [i]
            assigned[i] = True
            norm_i = self._normalize_for_match(candidates[i])

            for j in range(i + 1, len(candidates)):
                if assigned[j]:
                    continue
                norm_j = self._normalize_for_match(candidates[j])
                sim = self._line_similarity(norm_i, norm_j)
                if sim > 0.6:
                    group.append(j)
                    assigned[j] = True

            groups.append(group)

        # Find the largest group
        largest = max(groups, key=len)

        if len(largest) >= 2:
            # Consensus found — pick the best version from the group
            group_candidates = [candidates[i] for i in largest]
            return self._pick_best_line(group_candidates)

        # No consensus
        return None

    def _merge_ocr_results(self, *texts: str) -> str:
        """Legacy merge — delegates to consensus merge."""
        return self._merge_ocr_results_consensus(*texts)

    def _pick_best_line(self, candidates: List[str]) -> str:
        """
        Pick the best OCR line from multiple candidates.
        Improved scoring with Arabic-specific heuristics.
        """
        if not candidates:
            return ""

        if len(candidates) == 1:
            return candidates[0]

        scored = []
        for line in candidates:
            score = 0
            stripped = line.strip()

            # Length score — longer usually means more content captured
            score += min(len(stripped), 200) * 0.3

            # Penalize garbled text (mixed scripts in one word)
            garbled = len(re.findall(r'[\u0600-\u06ff][a-zA-Z]|[a-zA-Z][\u0600-\u06ff]', stripped))
            score -= garbled * 15

            # Reward proper numbers (digits sequences)
            proper_numbers = len(re.findall(r'\d{2,}', stripped))
            score += proper_numbers * 3

            # Reward proper Arabic words (3+ chars)
            arabic_words = len(re.findall(r'[\u0600-\u06ff]{3,}', stripped))
            score += arabic_words * 5  # Increased weight for Arabic

            # Reward proper English words (3+ chars)
            english_words = len(re.findall(r'[a-zA-Z]{3,}', stripped))
            score += english_words * 3

            # Penalize excessive special characters
            special = len(re.findall(r'[^\w\s\u0600-\u06ff.,;:\-/()٠-٩@#%&+=]', stripped))
            score -= special * 5

            # Penalize very short lines that should have content
            if len(stripped) < 3 and any(len(c.strip()) >= 10 for c in candidates):
                score -= 20

            # Reward lines with common document patterns
            # Dates, amounts, reference numbers
            if re.search(r'\d{1,4}[/-]\d{1,2}[/-]\d{1,4}', stripped):
                score += 5  # Date pattern
            if re.search(r'\d+[\.,]\d{2}', stripped):
                score += 3  # Decimal amount
            if re.search(r'(?:No|Number|رقم|عدد)\s*[:.]?\s*\d+', stripped, re.IGNORECASE):
                score += 3  # Reference number

            # Arabic-specific: reward lines with Arabic diacritics (more faithful to original)
            diacritics = len(re.findall(r'[\u0617-\u061A\u064B-\u0652]', stripped))
            score += diacritics * 0.5  # Small bonus — diacritics mean careful reading

            # Arabic-specific: reward lines with proper Arabic word patterns
            # (words connected with proper connectors)
            arabic_connectors = len(re.findall(r'و\s+[\u0600-\u06ff]|ب\s+[\u0600-\u06ff]|لل\w+', stripped))
            score += arabic_connectors * 2

            # Penalize lines that look like model hallucinations
            hallucination_patterns = [
                r'I\s+cannot', r'I\s+am\s+unable', r'sorry',
                r'لا\s+أستطيع', r'لا\s+يمكنني', r'عذراً',
                r'Note:', r'Disclaimer:', r'ملاحظة:',
            ]
            for pattern in hallucination_patterns:
                if re.search(pattern, stripped, re.IGNORECASE):
                    score -= 50

            # Reward lines with balanced Arabic/English (mixed documents)
            arabic_chars = len(re.findall(r'[\u0600-\u06ff]', stripped))
            english_chars = len(re.findall(r'[a-zA-Z]', stripped))
            if arabic_chars > 0 and english_chars > 0:
                ratio = min(arabic_chars, english_chars) / max(arabic_chars, english_chars, 1)
                if ratio > 0.1:  # Reasonable mix
                    score += 2

            scored.append((score, line))

        # Return highest scored
        scored.sort(key=lambda x: -x[0])
        return scored[0][1]

    # ===== QUALITY GATE =====

    def _check_quality(self, text: str) -> Tuple[str, List[str]]:
        """
        Check OCR output quality and return quality level + issues.
        Returns: (quality_level, issues_list)
        quality_level: "excellent", "good", "needs_review", "poor"
        """
        issues = []

        if not text or len(text) < config.QUALITY_MIN_CHARS:
            issues.append("Very short output")
            return "poor", issues

        lines = [l.strip() for l in text.split('\n') if l.strip()]
        if len(lines) < config.QUALITY_MIN_LINES:
            issues.append("Too few lines")

        # Check for repetition
        if lines:
            counts = Counter(lines)
            top, top_count = counts.most_common(1)[0]
            if top_count > 3:
                issues.append(f"Repetition: '{top[:40]}' x{top_count}")

        # Check for hallucination markers
        hallucination_markers = [
            "I cannot", "I am unable", "sorry", "لا أستطيع",
            "As an AI", "I don't have", "I'm not able",
        ]
        text_lower = text.lower()
        for marker in hallucination_markers:
            if marker.lower() in text_lower:
                issues.append(f"Hallucination: '{marker}'")
                break

        # Check for code remnants
        code_words = ["function", "console", "const ", "<script", "var ", "import "]
        for w in code_words:
            if w in text_lower:
                issues.append("Code remnants")
                break

        # Check for excessive non-Arabic/non-English text
        arabic = len(re.findall(r'[\u0600-\u06ff]', text))
        english = len(re.findall(r'[a-zA-Z]', text))
        total_alpha = arabic + english
        if total_alpha > 0:
            non_alpha = len(text) - total_alpha - len(re.findall(r'[\s\d.,;:\-/()@#%&+=]', text))
            if non_alpha > total_alpha * 0.3:
                issues.append("Excessive special characters")

        # Determine quality
        if not issues:
            if len(text) > 200 and len(lines) > 5:
                return "excellent", issues
            return "good", issues
        elif len(issues) <= 1:
            return "needs_review", issues
        else:
            return "poor", issues

    # ===== CROSS-VALIDATION =====

    def _cross_validate_json(self, json_data: dict, ocr_text: str) -> dict:
        """
        Cross-validate JSON extraction against OCR text.
        Fixes: null dates when OCR text has them, fabricated reference numbers,
        and date format issues.
        """
        if not isinstance(json_data, dict):
            return json_data

        # 1. Fill null dates from OCR text
        date_pattern = re.compile(
            r'(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})|'
            r'(\d{1,2}[/-][A-Z]{3}[/-]\d{2,4})|'
            r'(\d{1,2}\s+[A-Z]{3}\s+\d{4})',
            re.IGNORECASE,
        )
        ocr_dates = date_pattern.findall(ocr_text)
        ocr_dates_flat = [d for group in ocr_dates for d in group if d]

        for key in list(json_data.keys()):
            key_lower = key.lower()
            # Fill null date fields with dates found in OCR
            if ('date' in key_lower or 'تاريخ' in key) and json_data[key] is None:
                if ocr_dates_flat:
                    # Try to find a date near the field label in OCR text
                    found = self._find_date_near_label(key, ocr_text)
                    if found:
                        json_data[key] = found
                        logger.info(f"Cross-validation: filled null '{key}' with '{found}' from OCR text")

        # 2. Detect fabricated reference numbers (strings of X's or clearly fake patterns)
        for key in list(json_data.keys()):
            val = json_data[key]
            if not isinstance(val, str):
                continue
            # Flag reference numbers full of X's or not found anywhere in OCR
            if re.search(r'X{4,}', val):
                if val not in ocr_text:
                    logger.warning(f"Cross-validation: removing fabricated '{key}': '{val}'")
                    json_data[key] = None

        # 3. Validate dates against OCR text — catch day/month swaps
        for key in list(json_data.keys()):
            val = json_data[key]
            if not isinstance(val, str):
                continue
            key_lower = key.lower()
            if 'date' not in key_lower and 'تاريخ' not in key:
                continue

            # If JSON has a date that's not in OCR text, try to find the correct one
            if val and val not in ocr_text:
                # Check for day/month swap: JSON says 06/01/2004 but OCR has 06/10/2004
                swapped = self._try_swap_date(val)
                if swapped and swapped in ocr_text:
                    logger.warning(
                        f"Cross-validation: fixed date swap '{key}': '{val}' -> '{swapped}'"
                    )
                    json_data[key] = swapped

        return json_data

    def _find_date_near_label(self, label: str, ocr_text: str) -> Optional[str]:
        """Find a date value near a field label in OCR text."""
        label_variants = [label.lower().replace('_', ' ')]
        # Common label mappings
        if 'invoice' in label.lower():
            label_variants.extend(['invoice date', 'inv date', 'تاريخ الفاتورة'])
        elif 'issue' in label.lower():
            label_variants.extend(['date', 'تاريخ'])

        date_pat = re.compile(
            r'(\d{1,2}[/-]\w{2,3}[/-]\d{2,4}|\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{1,2}\s+[A-Z]{3}\s+\d{4})',
            re.IGNORECASE,
        )

        for variant in label_variants:
            idx = ocr_text.lower().find(variant)
            if idx >= 0:
                # Look within 80 chars after the label
                window = ocr_text[idx:idx + 80]
                match = date_pat.search(window)
                if match:
                    return match.group(1)

        return None

    def _try_swap_date(self, date_str: str) -> Optional[str]:
        """Try swapping day/month in a date string."""
        m = re.match(r'^(\d{1,2})([/-])(\d{1,2})([/-])(\d{2,4})$', date_str)
        if m:
            a, sep1, b, sep2, year = m.groups()
            swapped = f"{b}{sep1}{a}{sep2}{year}"
            if swapped != date_str:
                return swapped
        return None

    # ===== MAIN =====

    def process_first_page(self, file_path: Union[str, Path]) -> dict:
        file_path = Path(file_path)
        logger.info(f"📄 {file_path.name} (mode: {self.mode})")

        start = time.time()
        first_page = get_first_page_image(file_path)

        result = {
            "file_name": file_path.name,
            "json_data": None,
            "ocr_text": "",
            "clean_ocr_text": "",
        }

        # Step 1: Classification (Gemma-3)
        if self.mode in ("classify_only", "full"):
            logger.info("🏷️ Step 1: Classification...")
            t1 = time.time()
            result["json_data"] = self._run_gemma(first_page)
            result["classify_time"] = round(time.time() - t1, 2)
            logger.info(f"✅ Classification: {result['classify_time']}s")

        # Step 2: OCR (Qwen - Multi-pass)
        if self.mode in ("ocr_only", "full"):
            if self.mode == "full":
                logger.info("🔄 Swapping models...")

            logger.info("📝 Step 2: Multi-pass OCR...")
            t2 = time.time()

            # Use multi-pass for maximum accuracy
            raw_ocr = self._run_qwen_multipass(first_page)
            cleaned = clean_ocr_text(raw_ocr)

            # Quality gate: retry if output is poor
            if config.QUALITY_RETRY_ENABLED:
                attempt = 1
                quality, issues = self._check_quality(cleaned)
                
                while quality == "poor" and attempt <= config.QUALITY_RETRY_MAX_ATTEMPTS:
                    logger.warning(f"⚠️ Quality: {quality} | Issues: {issues} | Retry {attempt}/{config.QUALITY_RETRY_MAX_ATTEMPTS}")
                    
                    # Retry with light preprocessing (VLM-friendly)
                    retry_img = light_preprocess_vlm(
                        first_page, max_width=1800,
                    )
                    retry_text = self._run_qwen_single(retry_img, config.OCR_PROMPT_VERIFY)
                    retry_cleaned = clean_ocr_text(retry_text)
                    
                    retry_quality, retry_issues = self._check_quality(retry_cleaned)
                    logger.info(f"  Retry {attempt}: {retry_quality} ({len(retry_cleaned)} chars)")
                    
                    # Use retry if it's better
                    if retry_quality != "poor" or len(retry_cleaned) > len(cleaned) * 1.2:
                        raw_ocr = retry_text
                        cleaned = retry_cleaned
                        quality = retry_quality
                        issues = retry_issues
                    
                    attempt += 1

                logger.info(f"📊 Final quality: {quality} | Issues: {issues}")

            result["ocr_text"] = raw_ocr
            result["clean_ocr_text"] = cleaned
            result["ocr_time"] = round(time.time() - t2, 2)
            logger.info(f"✅ OCR: {result['ocr_time']}s")

        elapsed = time.time() - start

        # Cross-validate JSON against OCR text
        if result["json_data"] and result["clean_ocr_text"]:
            result["json_data"] = self._cross_validate_json(
                result["json_data"], result["clean_ocr_text"]
            )

        text_to_validate = result["clean_ocr_text"] or str(result.get("json_data", ""))
        validation = validate_ocr_output(text_to_validate, file_path.name)

        result["processing_time"] = round(elapsed, 2)
        result["char_count"] = len(result["clean_ocr_text"] or "")
        result["quality"] = validation["quality"]
        result["issues"] = validation["issues"]

        logger.info(f"✅ Total: {elapsed:.1f}s | Quality: {result['quality']}")
        return result

    def extract_from_document(self, file_path: Union[str, Path], custom_prompt: str = None) -> dict:
        """API-friendly method for extracting from a document."""
        file_path = Path(file_path)
        first_page = get_first_page_image(file_path)

        start = time.time()

        # Run OCR
        if custom_prompt:
            img = light_preprocess_vlm(first_page, max_width=1600)
            raw_ocr = self._run_qwen_single(img, custom_prompt)
        else:
            raw_ocr = self._run_qwen_multipass(first_page)

        cleaned = clean_ocr_text(raw_ocr)
        validation = validate_ocr_output(cleaned, file_path.name)

        return {
            "file_name": file_path.name,
            "ocr_text": cleaned,
            "raw_ocr_text": raw_ocr,
            "processing_time": round(time.time() - start, 2),
            "char_count": len(cleaned),
            "quality": validation["quality"],
            "issues": validation["issues"],
        }
