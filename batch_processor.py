"""
Batch processor for dual-model system.
"""
import csv
import json
import time
from pathlib import Path
from typing import Optional, List

from tqdm import tqdm
from loguru import logger

import config
from ocr_engine import OCREngine
from classifier import DocumentClassifier


class BatchProcessor:

    def __init__(self):
        self.engine = OCREngine()
        self.classifier = DocumentClassifier()

    def get_files(self, input_dir: Optional[Path] = None) -> List[Path]:
        input_dir = input_dir or config.INPUT_DIR
        all_fmts = config.SUPPORTED_IMAGE_FORMATS | {config.SUPPORTED_PDF_FORMAT}
        return sorted([f for f in input_dir.iterdir() if f.suffix.lower() in all_fmts])

    def process_single(self, file_path: Path, output_dir: Path) -> dict:
        try:
            # Process
            ocr = self.engine.process_first_page(file_path)

            # Classify using BOTH json_data and ocr text
            classify_text = ocr.get("clean_ocr_text") or str(ocr.get("json_data", ""))
            cls = self.classifier.classify(
                ocr_text=classify_text,
                json_data=ocr.get("json_data"),
            )

            # Save .txt
            with open(output_dir / f"{file_path.stem}.txt", "w", encoding="utf-8") as f:
                f.write(f"File: {file_path.name}\n")
                f.write(f"Type: {cls['name_en']} / {cls['name_ar']}\n")
                f.write(f"Confidence: {cls['confidence']}%\n")
                f.write(f"Method: {cls.get('method', '')}\n")
                if cls.get("model_said"):
                    f.write(f"Model said: {cls['model_said']}\n")
                f.write(f"{'='*60}\n\n")

                # Write OCR text if available
                if ocr.get("clean_ocr_text"):
                    f.write("📜 OCR TEXT:\n")
                    f.write(ocr["clean_ocr_text"])
                    f.write("\n\n")

                # Write JSON data if available
                if ocr.get("json_data"):
                    f.write("📋 EXTRACTED DATA:\n")
                    f.write(json.dumps(ocr["json_data"], ensure_ascii=False, indent=2))

            # Save .json
            with open(output_dir / f"{file_path.stem}.json", "w", encoding="utf-8") as f:
                json.dump({
                    "classification": {
                        "document_type": cls["document_type"],
                        "type_ar": cls["name_ar"],
                        "type_en": cls["name_en"],
                        "confidence": cls["confidence"],
                        "method": cls.get("method", ""),
                        "model_said": cls.get("model_said", ""),
                        "keywords_matched": cls.get("keyword_matches", 0),
                        "matched_keywords": cls.get("matched_keywords", []),
                    },
                    "file_info": {
                        "file_name": file_path.name,
                        "processing_time": ocr["processing_time"],
                        "classify_time": ocr.get("classify_time", 0),
                        "ocr_time": ocr.get("ocr_time", 0),
                        "char_count": ocr["char_count"],
                        "quality": ocr["quality"],
                    },
                    "extracted_json": ocr.get("json_data", {}),
                    "ocr_text": ocr.get("clean_ocr_text", ""),
                }, f, ensure_ascii=False, indent=2)

            return {
                "file_name": file_path.name,
                "status": "success",
                "document_type": cls["document_type"],
                "type_en": cls["name_en"],
                "confidence": cls["confidence"],
                "method": cls.get("method", ""),
                "processing_time": ocr["processing_time"],
                "quality": ocr["quality"],
            }

        except Exception as e:
            logger.error(f"❌ {file_path.name}: {e}")
            return {
                "file_name": file_path.name,
                "status": "error",
                "document_type": "error",
                "type_en": "Error",
                "confidence": 0,
                "method": "",
                "processing_time": 0,
                "quality": "error",
            }

    def process_batch(
        self,
        input_dir: Optional[Path] = None,
        output_dir: Optional[Path] = None,
    ):
        input_dir = input_dir or config.INPUT_DIR
        output_dir = output_dir or config.OUTPUT_DIR

        files = self.get_files(input_dir)
        if not files:
            logger.warning("No documents!")
            return

        logger.info(f"📂 {len(files)} documents | Mode: {config.PROCESSING_MODE}")
        start = time.time()
        results = []
        ok = fail = 0

        for f in tqdm(files, desc="📜 Processing"):
            r = self.process_single(f, output_dir)
            results.append(r)
            if r["status"] == "success":
                ok += 1
            else:
                fail += 1

        elapsed = time.time() - start

        # CSV
        csv_file = output_dir / "classification_report.csv"
        with open(csv_file, "w", encoding="utf-8-sig", newline="") as f:
            if results:
                writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
                writer.writeheader()
                writer.writerows(results)

        types = {}
        for r in results:
            t = r["type_en"]
            types[t] = types.get(t, 0) + 1

        logger.info(f"\n{'='*60}")
        logger.info(f"✅ {ok} ok | {fail} failed | {elapsed:.1f}s")
        logger.info(f"📋 TYPES:")
        for t, c in sorted(types.items(), key=lambda x: -x[1]):
            logger.info(f"   {t}: {c}")
        logger.info(f"{'='*60}")