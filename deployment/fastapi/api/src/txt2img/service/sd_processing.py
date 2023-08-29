from functools import lru_cache

import tomesd
from diffusers import StableDiffusionPipeline

from .processing import DiffusionProcessing


class StableDiffusionProcessing(DiffusionProcessing):
    def generate(
        self,
        prompt: str,
        **kargs,
    ):
        images = self._pipeline(prompt, **kargs).images
        return images

    @lru_cache(maxsize=1)
    def _load_pipeline(
        self,
        model_path: str,
        device: str = 'cpu',
        xformer: bool = False,
        tome_ratio: float = None,
        lora_path: str = None,
        **kargs,
    ):
        self._pipeline = StableDiffusionPipeline.from_pretrained(
            model_path,
            **kargs,
        ).to(device)

        if xformer:
            self._pipeline.enable_xformers_memory_efficient_attention()

        if tome_ratio:
            tomesd.apply_patch(self._pipeline, ratio=tome_ratio)

        if lora_path:
            self._pipeline.load_lora_weights(lora_path)
