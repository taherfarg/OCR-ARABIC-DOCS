"""
Batch processor - saves plain text output.
"""
import time
from pathlib import Path
from typing import Optional, List

from tqdm import tqdm
from loguru import logger

import config
from ocr_engine import OCREngine


class BatchProcessor:

    def __init__(self, custom_prompt: Optional[str] = None):
        self.engine = OCREngine(prompt=custom_prompt)

    def get_files(self, input_dir: Optional[Path] = None) -> List[Path]:
        input_dir = input_dir or config.INPUT_DIR
        all_formats = config.SUPPORTED_IMAGE_FORMATS | {config.SUPPORTED_PDF_FORMAT}
        files = sorted([
            f for f in input_dir.iterdir()
            if f.suffix.lower() in all_formats
        ])
        logger.info(f"Found {len(files)} document(s)")
        return files

    def process_batch(
        self,
        input_dir: Optional[Path] = None,
        output_dir: Optional[Path] = None
    ) -> dict:
        input_dir = input_dir or config.INPUT_DIR
        output_dir = output_dir or config.OUTPUT_DIR

        files = self.get_files(input_dir)
        if not files:
            logger.warning("No documents found!")
            return {"total": 0}

        start = time.time()
        success = failed = 0

        for f in tqdm(files, desc="OCR Processing"):
            try:
                result = self.engine.ocr_document(f)

                # ===== SAVE AS PLAIN TEXT FILE =====
                out_file = output_dir / f"{f.stem}.txt"
                with open(out_file, "w", encoding="utf-8") as fp:
                    for page in result["pages"]:
                        if result["total_pages"] > 1:
                            fp.write(f"--- page {page['page_number']} ---\n\n")
                        fp.write(page["text"])
                        fp.write("\n\n")

                logger.info(f"Saved: {out_file}")
                success += 1

            except Exception as e:
                logger.error(f"Failed: {f.name}: {e}")
                failed += 1

        elapsed = time.time() - start

        logger.info(f"\n{'='*50}")
        logger.info(f"DONE | {success} ok | {failed} failed")
        logger.info(f"Total: {elapsed:.1f}s | Avg: {elapsed/max(len(files),1):.1f}s per doc")
        logger.info(f"Output: {output_dir}")
        logger.info(f"{'='*50}")

        return {
            "total": len(files),
            "success": success,
            "failed": failed,
            "time_seconds": round(elapsed, 2)
        }
