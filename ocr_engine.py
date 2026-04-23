"""
OCR engine with multi-pass for maximum accuracy.
"""
import gc
import json
import time
import re
from typing import Union, Optional, List
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
                do_sample=False,
                repetition_penalty=config.GEMMA_REPETITION_PENALTY,
            )

        raw = processor.decode(outputs[0][input_len:], skip_special_tokens=True).strip()

        del inputs, outputs
        gc.collect()
        torch.cuda.empty_cache()

        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            try:
                data = json_repair.loads(raw)
                return data if isinstance(data, dict) else {"raw": str(data)}
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
                do_sample=False,
                temperature=0.1,
                top_p=0.1,
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

    # ===== MULTI-PASS OCR (Key to 99%) =====

    def _run_qwen_multipass(self, original_image: Image.Image) -> str:
        """
        Run OCR 3 times with different preprocessing and prompts.
        Then merge the best results line by line.
        """
        logger.info("🔄 Multi-pass OCR (3 passes)...")

        # ===== PASS 1: Standard preprocessing =====
        logger.info("  Pass 1/3: Standard...")
        img1 = full_preprocess_pipeline(
            original_image, for_model="qwen", max_width=1600,
        )
        text1 = self._run_qwen_single(img1, config.OCR_PROMPT)

        # ===== PASS 2: Higher resolution + different prompt =====
        logger.info("  Pass 2/3: High resolution...")
        img2 = full_preprocess_pipeline(
            original_image, for_model="qwen", max_width=2048,
        )
        prompt2 = """Read this document image carefully and extract ALL text exactly as written.
Include every word, number, date, and symbol. 
Maintain the exact order of lines.
For tables, preserve the column alignment.
Do not skip any text, even if partially visible.
Output the complete text:"""
        text2 = self._run_qwen_single(img2, prompt2)

        # ===== PASS 3: Focus on numbers and details =====
        logger.info("  Pass 3/3: Detail focus...")
        img3 = full_preprocess_pipeline(
            original_image, for_model="qwen", max_width=1600,
        )
        prompt3 = """Extract all text from this document with special attention to:
- All numbers (account numbers, invoice numbers, amounts, dates)
- All names (person names, company names, bank names)
- All Arabic text exactly as written
- Table data with correct alignment
Output the complete text:"""
        text3 = self._run_qwen_single(img3, prompt3)

        # ===== MERGE: Best of all passes =====
        merged = self._merge_ocr_results(text1, text2, text3)

        logger.info(
            f"  Merge: Pass1={len(text1)} | Pass2={len(text2)} | "
            f"Pass3={len(text3)} → Merged={len(merged)} chars"
        )

        return merged

    def _merge_ocr_results(self, *texts: str) -> str:
        """
        Intelligent merge of multiple OCR passes.
        For each line position, pick the best version.
        """
        # Split into lines
        all_lines = [t.split('\n') for t in texts]

        # Use the longest result as base
        max_len = max(len(lines) for lines in all_lines)
        base_idx = [len(lines) for lines in all_lines].index(max_len)
        base_lines = all_lines[base_idx]

        merged_lines = []

        for i, base_line in enumerate(base_lines):
            candidates = [base_line]

            # Collect same-position lines from other passes
            for j, lines in enumerate(all_lines):
                if j == base_idx:
                    continue
                if i < len(lines):
                    candidates.append(lines[i])

            # Pick the best line
            best = self._pick_best_line(candidates)
            merged_lines.append(best)

        return '\n'.join(merged_lines)

    def _pick_best_line(self, candidates: List[str]) -> str:
        """
        Pick the best OCR line from multiple candidates.
        Criteria:
        1. Longer is usually better (more content captured)
        2. Fewer garbled characters
        3. Consistent numbers
        """
        if not candidates:
            return ""

        if len(candidates) == 1:
            return candidates[0]

        scored = []
        for line in candidates:
            score = 0

            # Length score (longer = more content)
            score += len(line.strip()) * 0.5

            # Penalize garbled text (mixed scripts in one word)
            garbled = len(re.findall(r'[\u0600-\u06ff][a-zA-Z]|[a-zA-Z][\u0600-\u06ff]', line))
            score -= garbled * 10

            # Reward proper numbers
            proper_numbers = len(re.findall(r'\d+', line))
            score += proper_numbers * 2

            # Reward proper Arabic words (3+ chars)
            arabic_words = len(re.findall(r'[\u0600-\u06ff]{3,}', line))
            score += arabic_words * 3

            # Reward proper English words (3+ chars)
            english_words = len(re.findall(r'[a-zA-Z]{3,}', line))
            score += english_words * 2

            # Penalize excessive special characters
            special = len(re.findall(r'[^\w\s\u0600-\u06ff.,;:\-/()٠-٩]', line))
            score -= special * 3

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

        logger.info(f"✅ Total: {elapsed:.1f}s")
        return result