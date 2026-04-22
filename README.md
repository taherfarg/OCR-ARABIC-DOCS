# OCR-ARABIC-DOCS

Arabic Legal Document OCR — Pure text extraction using [bakrianoo/arabic-legal-documents-ocr-1.0](https://huggingface.co/bakrianoo/arabic-legal-documents-ocr-1.0) (Gemma 3 based).

Extracts raw text from scanned Arabic legal documents exactly as-is — no JSON, no categorization, just the text in the exact order it appears on the page.

## Features

- **Pure text extraction** — preserves original word order and line breaks
- **GPU accelerated** — ~23s per page on RTX 5060 Ti (vs 3+ min on CPU)
- **PDF & image support** — JPG, PNG, BMP, TIFF, PDF
- **Batch processing** — process entire directories at once
- **Arabic & English prompts** — switchable with `--lang`

## Requirements

- Python 3.10+
- NVIDIA GPU with 16GB+ VRAM
- PyTorch with CUDA support

## Setup

```bash
# Create virtual environment
python -m venv venv
venv\Scripts\activate   # Windows
# source venv/bin/activate  # Linux/Mac

# Install PyTorch with CUDA (check your CUDA version with nvidia-smi)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# Install dependencies
pip install -r requirements.txt
```

## Usage

```bash
# Single file
python main.py --mode single --input input_documents/page_001.jpg

# Batch process all files in input_documents/
python main.py --mode batch

# Interactive mode
python main.py --mode interactive

# Use English prompt
python main.py --mode batch --lang en
```

## Output

Plain `.txt` files saved to `output_results/`:

```
بسم الله الرحمن الرحيم

المملكة العربية السعودية
هيئة الخبراء بمجلس الوزراء

نظام المعاملات المدنية
باب تمهيدي
...
```

## Project Structure

```
├── main.py              # Entry point (single/batch/interactive)
├── config.py            # Configuration & prompts
├── model_loader.py      # GPU model loader
├── ocr_engine.py        # Pure text OCR engine
├── batch_processor.py   # Batch processing
├── preprocess.py        # Image preprocessing
├── input_documents/     # Place documents here
├── output_results/      # Extracted text output
└── requirements.txt     # Dependencies
```

## GPU Diagnostic

```bash
python diagnose.py
```

Verifies CUDA availability and GPU access before running OCR.
