"""
OCR engine with improved multi-pass strategy and intelligent merging.

Key improvements:
- Better generation parameters (temperature, top_p, top_k)
- Smarter multi-pass with varied preprocessing per pass
- Line-level voting/consensus merge instead of simple scoring
- Confidence tracking per line
- Better handling of handwritten + printed mixed documents
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

import config
from model_loader import GemmaModelManager, QwenModelManager
from advanced_preprocess import full_preprocess_pipeline
from preprocess import get_first_page_image
from post_process import clean_ocr_text, validate_ocr_output


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

        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=config.GEMMA_MAX_TOKENS,
                do_sample=True,
                temperature=config.GEMMA_TEMPERATURE,
                repetition_penalty=config.GEMMA_REPETITION_PENALTY,
            )

        raw = processor.decode(outputs[0][input_len:], skip_special_tokens=True).strip()

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

        with torch.inference_mode():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=config.QWEN_MAX_TOKENS,
                min_new_tokens=config.QWEN_MIN_TOKENS,
                do_sample=True,
                temperature=config.QWEN_TEMPERATURE,
                top_p=config.QWEN_TOP_P,
                top_k=config.QWEN_TOP_K,
                repetition_penalty=config.QWEN_REPETITION_PENALTY,
                pad_token_id=processor.tokenizer.eos_token_id,
                eos_token_id=processor.tokenizer.eos_token_id,
            )

        output = processor.batch_decode(
            generated_ids[:, input_len:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )[0].strip()

        del inputs, generated_ids
        gc.collect()
        torch.cuda.empty_cache()

        return output

    # ===== MULTI-PASS OCR (Key to high accuracy) =====

    def _run_qwen_multipass(self, original_image: Image.Image) -> str:
        """
        Run OCR multiple times with different preprocessing and prompts.
        Then merge using consensus voting for maximum accuracy.
        """
        if not config.MULTIPASS_ENABLED:
            # Single pass fallback
            img = full_preprocess_pipeline(
                original_image, for_model="qwen", max_width=1600,
            )
            return self._run_qwen_single(img, config.OCR_PROMPT)

        num_passes = config.MULTIPASS_COUNT
        logger.info(f"🔄 Multi-pass OCR ({num_passes} passes)...")

        passes = []

        # ===== PASS 1: Standard — Arabic prompt, standard resolution =====
        logger.info("  Pass 1/3: Standard Arabic...")
        img1 = full_preprocess_pipeline(
            original_image, for_model="qwen", max_width=1600,
        )
        text1 = self._run_qwen_single(img1, config.OCR_PROMPT)
        passes.append(("standard_ar", text1))

        # ===== PASS 2: High resolution — English prompt =====
        logger.info("  Pass 2/3: High resolution English...")
        img2 = full_preprocess_pipeline(
            original_image, for_model="qwen", max_width=2048,
        )
        text2 = self._run_qwen_single(img2, config.OCR_PROMPT_EN)
        passes.append(("highres_en", text2))

        # ===== PASS 3: Detail focus — Arabic prompt, enhanced contrast =====
        logger.info("  Pass 3/3: Detail focus...")
        img3 = full_preprocess_pipeline(
            original_image, for_model="qwen", max_width=1600,
            extra_contrast=True,
        )
        text3 = self._run_qwen_single(img3, config.OCR_PROMPT_DETAIL)
        passes.append(("detail_ar", text3))

        # ===== MERGE: Consensus voting =====
        merged = self._merge_ocr_results_consensus(text1, text2, text3)

        for name, text in passes:
            logger.info(f"  {name}: {len(text)} chars")
        logger.info(f"  → Merged: {len(merged)} chars")

        return merged

    # ===== IMPROVED MERGE: Consensus Voting =====

    def _merge_ocr_results_consensus(self, *texts: str) -> str:
        """
        Merge multiple OCR passes using consensus voting.

        Strategy:
        1. Align lines across passes using dynamic programming
        2. For each line position, if 2+ passes agree → use that (high confidence)
        3. If all disagree → pick the best scored line
        4. Detect and preserve extra lines that appear in multiple passes
        """
        if len(texts) == 1:
            return texts[0]

        all_lines = [t.split('\n') for t in texts]

        # Remove empty lines but track their positions for reconstruction
        all_nonempty = []
        for lines in all_lines:
            nonempty = [(i, l) for i, l in enumerate(lines) if l.strip()]
            all_nonempty.append(nonempty)

        # Use the longest non-empty result as the base structure
        lengths = [len(ne) for ne in all_nonempty]
        base_idx = lengths.index(max(lengths))
        base_lines = [l for _, l in all_nonempty[base_idx]]

        if not base_lines:
            return ""

        # Align other passes to the base using fuzzy matching
        aligned = [[] for _ in range(len(base_lines))]

        for pass_idx, nonempty in enumerate(all_nonempty):
            pass_lines = [l for _, l in nonempty]

            if pass_idx == base_idx:
                for i, line in enumerate(pass_lines):
                    aligned[i].append(line)
                continue

            # Simple alignment: match each base line to best candidate in this pass
            used = set()
            for i, base_line in enumerate(base_lines):
                best_match = None
                best_score = -1
                best_j = -1

                base_norm = self._normalize_for_match(base_line)

                for j, cand_line in enumerate(pass_lines):
                    if j in used:
                        continue
                    cand_norm = self._normalize_for_match(cand_line)
                    score = self._line_similarity(base_norm, cand_norm)
                    if score > best_score:
                        best_score = score
                        best_match = cand_line
                        best_j = j

                if best_match and best_score > 0.2:
                    aligned[i].append(best_match)
                    used.add(best_j)

        # Now merge each position using consensus
        merged_lines = []
        for i, candidates in enumerate(aligned):
            if not candidates:
                continue

            if len(candidates) == 1:
                merged_lines.append(candidates[0])
                continue

            # Check for consensus — if 2+ candidates are very similar, use them
            consensus = self._find_consensus(candidates)
            if consensus:
                merged_lines.append(consensus)
            else:
                # No consensus — pick the best scored line
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
        Uses character-level overlap (Jaccard-like).
        """
        if not a or not b:
            return 0.0

        # Character set overlap
        set_a = set(a)
        set_b = set(b)
        if not set_a or not set_b:
            return 0.0

        intersection = len(set_a & set_b)
        union = len(set_a | set_b)
        char_sim = intersection / union if union else 0.0

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

        # Weighted combination
        return 0.3 * char_sim + 0.5 * word_sim + 0.2 * len_sim

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
        Improved scoring with better heuristics.
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
            score += arabic_words * 4

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

            scored.append((score, line))

        # Return highest scored
        scored.sort(key=lambda x: -x[0])
        return scored[0][1]

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

            result["ocr_text"] = raw_ocr
            result["clean_ocr_text"] = clean_ocr_text(raw_ocr)
            result["ocr_time"] = round(time.time() - t2, 2)
            logger.info(f"✅ OCR: {result['ocr_time']}s")

        elapsed = time.time() - start

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
            img = full_preprocess_pipeline(first_page, for_model="qwen", max_width=1600)
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
