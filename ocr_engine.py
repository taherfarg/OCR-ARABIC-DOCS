"""
Pure OCR engine - extracts raw text only.
No JSON, no categorization, just text as-is.
"""
import gc
import time
from typing import Union, Optional
from pathlib import Path

import torch
from PIL import Image
from loguru import logger

import config
from model_loader import ModelManager
from preprocess import preprocess_image, load_document_images


class OCREngine:

    def __init__(self, prompt: Optional[str] = None):
        self.manager = ModelManager.get_instance()
        self.prompt = prompt or config.OCR_PROMPT
        logger.info("OCR Engine ready (pure text mode)")

    def _generate(self, image: Image.Image, prompt: str) -> str:
        """Run inference and return raw text."""
        model = self.manager.model
        processor = self.manager.processor

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt}
                ]
            }
        ]

        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt"
        ).to(model.device)

        input_len = inputs["input_ids"].shape[-1]

        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=config.MAX_NEW_TOKENS,
                do_sample=config.DO_SAMPLE,
            )

        generated_tokens = outputs[0][input_len:]
        raw_text = processor.decode(generated_tokens, skip_special_tokens=True)

        # Free memory
        del inputs, outputs
        gc.collect()
        torch.cuda.empty_cache()

        return raw_text.strip()

    def ocr_image(
        self,
        image_input: Union[str, Path, Image.Image],
        custom_prompt: Optional[str] = None
    ) -> str:
        """
        OCR a single image -> returns plain text.
        """
        start = time.time()

        processed_img = preprocess_image(image_input, return_base64=False)

        prompt = custom_prompt or self.prompt
        logger.info("Running OCR...")

        text = self._generate(processed_img, prompt)

        elapsed = time.time() - start
        logger.info(f"Done in {elapsed:.1f}s | {len(text)} chars extracted")

        return text

    def ocr_document(
        self,
        file_path: Union[str, Path],
        custom_prompt: Optional[str] = None
    ) -> dict:
        """
        OCR a full document (image or PDF).
        Returns dict with page texts.
        """
        file_path = Path(file_path)
        logger.info(f"Processing: {file_path.name}")

        images = load_document_images(file_path)
        logger.info(f"Pages: {len(images)}")

        result = {
            "file_name": file_path.name,
            "total_pages": len(images),
            "pages": []
        }

        for i, image in enumerate(images):
            logger.info(f"Page {i+1}/{len(images)}")
            start = time.time()
            text = self.ocr_image(image, custom_prompt)
            elapsed = time.time() - start

            result["pages"].append({
                "page_number": i + 1,
                "text": text,
                "processing_time": round(elapsed, 2)
            })

        return result
