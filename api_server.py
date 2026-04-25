"""
FastAPI REST API server.
"""
import io
import json
import tempfile
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
from loguru import logger

import config
from ocr_engine import OCREngine

app = FastAPI(
    title="Arabic Legal Document OCR API",
    version="1.0.0"
)

_engine: Optional[OCREngine] = None


def get_engine() -> OCREngine:
    global _engine
    if _engine is None:
        _engine = OCREngine()
    return _engine


@app.get("/")
async def root():
    return {"service": "Arabic Legal OCR", "status": "running"}


@app.get("/health")
async def health():
    return {"status": "healthy"}


@app.post("/ocr")
async def extract_document(
    file: UploadFile = File(...),
    prompt: Optional[str] = Form(None)
):
    suffix = Path(file.filename).suffix.lower()
    all_formats = config.SUPPORTED_IMAGE_FORMATS | {config.SUPPORTED_PDF_FORMAT}

    if suffix not in all_formats:
        raise HTTPException(400, f"Unsupported: {suffix}")

    try:
        content = await file.read()
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(content)
            tmp_path = tmp.name

        try:
            engine = get_engine()
            result = engine.extract_from_document(tmp_path, custom_prompt=prompt)
        finally:
            Path(tmp_path).unlink(missing_ok=True)

        return JSONResponse(content=result)
    except Exception as e:
        logger.error(f"Error: {e}")
        raise HTTPException(500, str(e))


def start_server():
    import uvicorn
    uvicorn.run(app, host=config.API_HOST, port=config.API_PORT)
