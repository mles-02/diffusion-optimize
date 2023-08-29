from fastapi import APIRouter, status
from fastapi.responses import JSONResponse
from src.txt2img.schemas import TextToImageResponse
from src.txt2img.service import get_diffusion_processing
from src.txt2img.utils import encode_pil_to_base64

router = APIRouter()


diffusion_processing = None


def load_diffusion_processing(diffusion_name, **kargs):
    global diffusion_processing
    diffusion_processing = get_diffusion_processing(diffusion_name)(**kargs)


@router.post('/load_model')
async def load_txt2img_pipeline(model_type):
    """
    mode_type:
        lora-diffusion
        finetuning-diffusion
        finetuning-diffusion-tome
        onnx-diffusion
        onnx-diffusion-u8
        openvino-diffusion
    """
    if model_type == 'lora-diffusion':
        load_diffusion_processing(
            diffusion_name='StableDiffusionProcessing',
            model_path='runwayml/stable-diffusion-v1-5',
            lora_path='quangnguyennn/pokemon-lora-xformer-sophia',
            device='cpu',
        )
    elif model_type == 'finetuning-diffusion':
        load_diffusion_processing(
            diffusion_name='StableDiffusionProcessing',
            model_path='Zero-nnkn/stable-diffusion-2-pokemon',
            device='cpu',
        )
    elif model_type == 'finetuning-diffusion-tome':
        load_diffusion_processing(
            diffusion_name='StableDiffusionProcessing',
            model_path='Zero-nnkn/stable-diffusion-2-pokemon',
            tome_ratio=0.5,
            device='cpu',
        )
    elif model_type == 'onnx-diffusion':
        load_diffusion_processing(
            diffusion_name='StableDiffusionOnnxProcessing',
            model_path='Zero-nnkn/stable-diffusion-2-pokemon',
            revision='onnx',
            device='cpu',
        )
    elif model_type == 'onnx-diffusion-u8':
        load_diffusion_processing(
            diffusion_name='StableDiffusionOnnxProcessing',
            model_path='Zero-nnkn/stable-diffusion-2-pokemon',
            revision='onnx-u8',
            device='cpu',
        )
    elif model_type == 'openvino-diffusion':
        load_diffusion_processing(
            diffusion_name='StableDiffusionOVProcessing',
            model_path='Zero-nnkn/stable-diffusion-2-pokemon',
            revision='openvino',
            device='cpu',
        )
    else:
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={'message': 'wrong model type'},
        )

    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={'message': f'Load model {model_type}'},
    )


def generate(processing, **kargs):
    images = processing.generate(**kargs)
    b64images = list(map(encode_pil_to_base64, images))
    return b64images


@router.post('/generate')
async def generate_txt2img(
    prompt: str,
    num_inference_steps: int = 20,
):
    b64images = generate(
        processing=diffusion_processing, prompt=prompt, num_inference_steps=num_inference_steps
    )
    return TextToImageResponse(images=b64images)
