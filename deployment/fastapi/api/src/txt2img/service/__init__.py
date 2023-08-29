from src.txt2img.service.sd_onnx_processing import StableDiffusionOnnxProcessing
from src.txt2img.service.sd_openvino_processing import StableDiffusionOVProcessing
from src.txt2img.service.sd_processing import StableDiffusionProcessing


def get_diffusion_processing(type):
    return {
        'StableDiffusionProcessing': StableDiffusionProcessing,
        'StableDiffusionOnnxProcessing': StableDiffusionOnnxProcessing,
        'StableDiffusionOVProcessing': StableDiffusionOVProcessing,
    }.get(type)
