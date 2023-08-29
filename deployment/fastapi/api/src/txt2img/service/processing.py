from abc import ABC, abstractmethod


class DiffusionProcessing(ABC):
    def __init__(self, model_path, device, **kargs):
        self._load_pipeline(model_path=model_path, device=device, **kargs)

    @abstractmethod
    def generate(
        self,
        prompt: str,
        **kargs,
    ):
        raise NotImplementedError

    @abstractmethod
    def _load_pipeline(self):
        raise NotImplementedError
