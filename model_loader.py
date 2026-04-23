"""
Dual model loader:
- Gemma-3 for classification
- Qwen2.5-VL for OCR text extraction
Memory managed: loads one at a time on 16GB VRAM
Supports optional 4-bit quantization for lower VRAM GPUs.
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

        from transformers import AutoProcessor, Gemma3ForConditionalGeneration, BitsAndBytesConfig

        kwargs = {
            "device_map": "auto",
            "torch_dtype": torch.bfloat16,
        }

        # Optional 4-bit quantization for constrained VRAM
        if config.USE_4BIT_QUANTIZATION:
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
            )
            logger.info("  Using 4-bit quantization")

        cls._model = Gemma3ForConditionalGeneration.from_pretrained(
            config.CLASSIFIER_MODEL_ID,
            **kwargs,
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

        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig

        kwargs = {
            "torch_dtype": torch.bfloat16,  # FIX: was 'dtype' which is ignored
            "device_map": "auto",
            "trust_remote_code": True,
        }

        # Optional 4-bit quantization for constrained VRAM
        if config.USE_4BIT_QUANTIZATION:
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
            )
            logger.info("  Using 4-bit quantization")

        cls._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            config.OCR_MODEL_ID,
            **kwargs,
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
