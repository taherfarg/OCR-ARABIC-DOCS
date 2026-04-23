"""
Dual model loader:
- Gemma-3 for classification
- Qwen2.5-VL for OCR text extraction
Memory managed: loads one at a time on 16GB VRAM
"""
import gc
import torch
from loguru import logger

import config


class GemmaModelManager:
    """Gemma-3 for classification."""
    _model = None
    _processor = None
    _loaded = False

    @classmethod
    def load(cls):
        if cls._loaded:
            return
        
        logger.info(f"📦 Loading Gemma-3: {config.CLASSIFIER_MODEL_ID}")
        
        # Free any Qwen model first
        QwenModelManager.unload()
        gc.collect()
        torch.cuda.empty_cache()

        from transformers import AutoProcessor, Gemma3ForConditionalGeneration

        cls._model = Gemma3ForConditionalGeneration.from_pretrained(
            config.CLASSIFIER_MODEL_ID,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )
        cls._model.eval()
        cls._processor = AutoProcessor.from_pretrained(config.CLASSIFIER_MODEL_ID)
        cls._loaded = True

        used = torch.cuda.memory_allocated() / 1e9
        logger.info(f"✅ Gemma-3 loaded | VRAM: {used:.1f} GB")

    @classmethod
    def unload(cls):
        if cls._model:
            del cls._model
            del cls._processor
            cls._model = None
            cls._processor = None
            cls._loaded = False
            gc.collect()
            torch.cuda.empty_cache()
            logger.info("🗑️ Gemma-3 unloaded")

    @classmethod
    def get_model(cls):
        cls.load()
        return cls._model, cls._processor


class QwenModelManager:
    """Qwen2.5-VL for OCR text extraction."""
    _model = None
    _processor = None
    _loaded = False

    @classmethod
    def load(cls):
        if cls._loaded:
            return

        logger.info(f"📦 Loading Qwen: {config.OCR_MODEL_ID}")

        # Free Gemma model first
        GemmaModelManager.unload()
        gc.collect()
        torch.cuda.empty_cache()

        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

        cls._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            config.OCR_MODEL_ID,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            ignore_mismatched_sizes=True,
        )
        cls._model.eval()
        cls._processor = AutoProcessor.from_pretrained(
            config.OCR_MODEL_ID,
            trust_remote_code=True,
        )
        cls._loaded = True

        used = torch.cuda.memory_allocated() / 1e9
        logger.info(f"✅ Qwen loaded | VRAM: {used:.1f} GB")

    @classmethod
    def unload(cls):
        if cls._model:
            del cls._model
            del cls._processor
            cls._model = None
            cls._processor = None
            cls._loaded = False
            gc.collect()
            torch.cuda.empty_cache()
            logger.info("🗑️ Qwen unloaded")

    @classmethod
    def get_model(cls):
        cls.load()
        return cls._model, cls._processor