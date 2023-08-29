from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
from src.config import settings
from src.txt2img.router import load_diffusion_processing
from src.txt2img.router import router as txt2img_router

app = FastAPI(title='Stable Diffusion API')


app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_headers=settings.CORS_HEADERS,
    allow_credentials=True,
    allow_methods=['*'],
)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    # Get the original 'detail' list of errors
    details = exc.errors()
    error_details = []

    for error in details:
        error_details.append({'error': f"{error['msg']} {str(error['loc'])}"})
    return JSONResponse(content={'message': error_details})


@app.on_event('startup')
async def startup_event():
    load_diffusion_processing(
        diffusion_name='StableDiffusionProcessing',
        model_path='Zero-nnkn/stable-diffusion-2-pokemon',
        device='cpu',
    )


@app.get('/', include_in_schema=False)
async def root() -> None:
    return RedirectResponse('/docs')


@app.get('/health', status_code=status.HTTP_200_OK, tags=['health'])
async def perform_healthcheck() -> None:
    return JSONResponse(content={'message': 'success'})


app.include_router(txt2img_router, prefix='/txt2img', tags=['txt2img'])
