"""
Arabic Legal Document OCR - Pure Text Extraction.

Usage:
    python main.py --mode single --input document.jpg
    python main.py --mode batch
    python main.py --mode interactive
"""
import argparse
import sys
from pathlib import Path

from loguru import logger
import config

logger.remove()
logger.add(sys.stderr, format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | {message}", level="INFO")
logger.add(config.LOG_DIR / "ocr.log", rotation="10 MB", level="DEBUG")


def single_mode(input_path, output_path=None, prompt=None):
    from ocr_engine import OCREngine

    engine = OCREngine(prompt=prompt)
    result = engine.ocr_document(input_path)

    # Print to console
    print("\n" + "=" * 60)
    print("EXTRACTED TEXT")
    print("=" * 60)
    for page in result["pages"]:
        if result["total_pages"] > 1:
            print(f"\n--- Page {page['page_number']} ---\n")
        print(page["text"])
    print("=" * 60)

    # Save to file
    if output_path:
        out = Path(output_path)
    else:
        out = config.OUTPUT_DIR / f"{Path(input_path).stem}.txt"

    with open(out, "w", encoding="utf-8") as f:
        for page in result["pages"]:
            if result["total_pages"] > 1:
                f.write(f"--- page {page['page_number']} ---\n\n")
            f.write(page["text"])
            f.write("\n\n")

    logger.info(f"Saved: {out}")


def batch_mode(input_dir=None, output_dir=None, prompt=None):
    from batch_processor import BatchProcessor

    processor = BatchProcessor(custom_prompt=prompt)
    processor.process_batch(
        input_dir=Path(input_dir) if input_dir else None,
        output_dir=Path(output_dir) if output_dir else None
    )


def interactive_mode():
    from ocr_engine import OCREngine

    print("\n" + "=" * 60)
    print("  Arabic Document OCR - Pure Text Mode")
    print("=" * 60)

    engine = OCREngine()

    while True:
        path = input("\nDocument path (or 'quit'): ").strip()
        if path.lower() in ('quit', 'exit', 'q'):
            print("Goodbye!")
            break

        if not Path(path).exists():
            print(f"Not found: {path}")
            continue

        try:
            result = engine.ocr_document(path)

            print("\n" + "=" * 60)
            for page in result["pages"]:
                if result["total_pages"] > 1:
                    print(f"\n--- Page {page['page_number']} ({page['processing_time']}s) ---\n")
                print(page["text"])
            print("=" * 60)

            # Auto-save
            out = config.OUTPUT_DIR / f"{Path(path).stem}.txt"
            with open(out, "w", encoding="utf-8") as f:
                for page in result["pages"]:
                    f.write(page["text"])
                    f.write("\n\n")
            print(f"Saved: {out}")

        except Exception as e:
            print(f"Error: {e}")


def main():
    parser = argparse.ArgumentParser(description="Arabic Document OCR - Pure Text")
    parser.add_argument("--mode", choices=["single", "batch", "interactive"], default="interactive")
    parser.add_argument("--input", type=str, help="Input file or directory")
    parser.add_argument("--output", type=str, help="Output file or directory")
    parser.add_argument("--lang", choices=["ar", "en"], default="ar", help="Prompt language")
    args = parser.parse_args()

    prompt = config.OCR_PROMPT if args.lang == "ar" else config.OCR_PROMPT_EN

    if args.mode == "single":
        if not args.input:
            parser.error("--input required")
        single_mode(args.input, args.output, prompt)
    elif args.mode == "batch":
        batch_mode(args.input, args.output, prompt)
    elif args.mode == "interactive":
        interactive_mode()


if __name__ == "__main__":
    main()
