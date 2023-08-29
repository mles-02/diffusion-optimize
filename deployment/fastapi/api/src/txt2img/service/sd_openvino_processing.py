from functools import lru_cache

from optimum.intel import OVStableDiffusionPipeline

from .processing import DiffusionProcessing


class StableDiffusionOVProcessing(DiffusionProcessing):
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
        device = 'CPU'
        self._pipeline = OVStableDiffusionPipeline.from_pretrained(
            model_path,
            device=device,
            compile=False,
            **kargs,
        )
        # TODO: Statically reshape to speedup before compile
        # batch_size, num_images, height, width = 1, 1, 512, 512
        # pipeline.reshape(batch_size, height, width, num_images)
        self._pipeline.compile()
