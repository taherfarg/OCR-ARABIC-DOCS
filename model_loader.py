"""
Model loader - GPU optimized.
"""
import gc
import torch
from transformers import AutoProcessor, Gemma3ForConditionalGeneration
from loguru import logger

import config


class ModelManager:
    _model = None
    _processor = None
    _loaded = False

    @classmethod
    def get_instance(cls):
        if not cls._loaded:
            cls._load_model()
        return cls()

    @classmethod
    def _load_model(cls):
        logger.info(f"Loading model: {config.MODEL_ID}")

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available! Reinstall PyTorch with CUDA.")

        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info(f"GPU: {torch.cuda.get_device_name(0)} | VRAM: {vram:.1f} GB")

        try:
            # Try dtype= first (newer transformers), fall back to torch_dtype=
            try:
                cls._model = Gemma3ForConditionalGeneration.from_pretrained(
                    config.MODEL_ID,
                    device_map=config.DEVICE_MAP,
                    dtype=torch.bfloat16,
                )
            except TypeError:
                cls._model = Gemma3ForConditionalGeneration.from_pretrained(
                    config.MODEL_ID,
                    device_map=config.DEVICE_MAP,
                    torch_dtype=torch.bfloat16,
                )

            cls._model.eval()

            cls._processor = AutoProcessor.from_pretrained(config.MODEL_ID)
            cls._loaded = True

            device = next(cls._model.parameters()).device
            used = torch.cuda.memory_allocated() / 1e9
            logger.info(f"Model on {device} | VRAM: {used:.1f}/{vram:.1f} GB")

        except Exception as e:
            logger.error(f"Failed: {e}")
            raise

    @property
    def model(self):
        return ModelManager._model

    @property
    def processor(self):
        return ModelManager._processor

    @classmethod
    def unload(cls):
        if cls._model is not None:
            del cls._model
            del cls._processor
            cls._model = None
            cls._processor = None
            cls._loaded = False
            gc.collect()
            torch.cuda.empty_cache()
