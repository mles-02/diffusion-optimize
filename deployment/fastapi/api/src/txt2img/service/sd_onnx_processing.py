from functools import lru_cache

from optimum.onnxruntime import ORTStableDiffusionPipeline

from .processing import DiffusionProcessing


class StableDiffusionOnnxProcessing(DiffusionProcessing):
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
        **kargs,
    ):
        # TODO: Custom device for OpenVINO ('CPU', 'GPU')
        device = 'cpu'
        self._pipeline = ORTStableDiffusionPipeline.from_pretrained(
            model_path,
            **kargs,
        ).to(device)
