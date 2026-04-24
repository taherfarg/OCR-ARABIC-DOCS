"""
Arabic Legal OCR + Classification (Dual Model).

Modes:
    classify_only → Gemma-3 only (fast, ~11s/doc)
    ocr_only      → Qwen only (accurate text)
    full          → Both models (classification + accurate OCR)

Usage:
    python main.py --mode single --input doc.pdf
    python main.py --mode batch
    python main.py --mode interactive
    python main.py --mode batch --processing classify_only
"""
import argparse
import json
import sys
from pathlib import Path

from loguru import logger
import config

logger.remove()
logger.add(sys.stderr, format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | {message}", level="INFO")
logger.add(config.LOG_DIR / "ocr.log", rotation="10 MB", level="DEBUG")


def single_mode(input_path):
    from ocr_engine import OCREngine
    from classifier import DocumentClassifier

    engine = OCREngine()
    classifier = DocumentClassifier()

    ocr = engine.process_first_page(input_path)

    classify_text = ocr.get("clean_ocr_text") or str(ocr.get("json_data", ""))
    cls = classifier.classify(classify_text, ocr.get("json_data"))

    # Display
    print(f"\n{'='*60}")
    print(f"📋 CLASSIFICATION")
    print(f"{'='*60}")
    print(f"  File:       {ocr['file_name']}")
    print(f"  Type:       {cls['name_en']} / {cls['name_ar']}")
    print(f"  Confidence: {cls['confidence']}%")
    print(f"  Method:     {cls.get('method', '')}")
    if cls.get("model_said"):
        print(f"  Model said: {cls['model_said']}")
    print(f"  Time:       {ocr['processing_time']}s")
    print(f"  Quality:    {ocr['quality']}")

    if ocr.get("clean_ocr_text"):
        print(f"\n{'='*60}")
        print(f"📜 OCR TEXT (Qwen):")
        print(f"{'='*60}")
        print(ocr["clean_ocr_text"])

    if ocr.get("json_data"):
        print(f"\n{'='*60}")
        print(f"📋 EXTRACTED JSON (Gemma-3):")
        print(f"{'='*60}")
        print(json.dumps(ocr["json_data"], ensure_ascii=False, indent=2))

    # Save
    stem = Path(input_path).stem
    with open(config.OUTPUT_DIR / f"{stem}.json", "w", encoding="utf-8") as f:
        json.dump({
            "classification": cls,
            "extracted_json": ocr.get("json_data", {}),
            "ocr_text": ocr.get("clean_ocr_text", ""),
            "file_info": {
                "processing_time": ocr["processing_time"],
                "quality": ocr["quality"],
            },
        }, f, ensure_ascii=False, indent=2)
    logger.info(f"💾 {config.OUTPUT_DIR / stem}.json")


def batch_mode(input_dir=None, output_dir=None):
    from batch_processor import BatchProcessor
    BatchProcessor().process_batch(
        input_dir=Path(input_dir) if input_dir else None,
        output_dir=Path(output_dir) if output_dir else None,
    )


def interactive_mode():
    from ocr_engine import OCREngine
    from classifier import DocumentClassifier

    print(f"\n{'='*60}")
    print(f"  📜 Arabic Legal OCR + Classification")
    print(f"  Mode: {config.PROCESSING_MODE}")
    print(f"{'='*60}")

    engine = OCREngine()
    classifier = DocumentClassifier()

    while True:
        path = input("\n📄 Path (or 'quit'): ").strip()
        if path.lower() in ("quit", "exit", "q"):
            break
        if not Path(path).exists():
            print(f"❌ Not found: {path}")
            continue

        try:
            ocr = engine.process_first_page(path)
            text = ocr.get("clean_ocr_text") or str(ocr.get("json_data", ""))
            cls = classifier.classify(text, ocr.get("json_data"))

            print(f"\n📋 {cls['name_en']} / {cls['name_ar']}")
            print(f"📊 {cls['confidence']}% ({cls.get('method', '')})")
            print(f"⏱  {ocr['processing_time']}s")

            if ocr.get("clean_ocr_text"):
                print(f"\n📜 OCR:\n{ocr['clean_ocr_text'][:500]}")

        except Exception as e:
            print(f"❌ {e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["single", "batch", "interactive"], default="interactive")
    parser.add_argument("--input", type=str)
    parser.add_argument("--output", type=str)
    parser.add_argument("--processing", choices=["classify_only", "ocr_only", "full"], default=None)
    args = parser.parse_args()

    # Override processing mode if specified
    if args.processing:
        config.PROCESSING_MODE = args.processing

    # Early validation: check quantization prerequisites before any model loading
    from model_loader import validate_quantization_config
    validate_quantization_config()

    if args.mode == "single":
        if not args.input:
            parser.error("--input required")
        single_mode(args.input)
    elif args.mode == "batch":
        batch_mode(args.input, args.output)
    else:
        interactive_mode()


if __name__ == "__main__":
    main()