# Optimize Stable Diffusion Text-to-Image
Investigate optimization method for both inference speed and memory-consumption of StableDiffusion with [🤗 diffusers](https://github.com/huggingface/optimum). You can see more details [here](https://huggingface.co/docs/diffusers/optimization/opt_overview).

## Pytorch Pipeline
### Half precision weights (Float16)
Load and run model directly in float16 (Only for GPU).
```py
import torch
from diffusers import DiffusionPipeline

pipe = DiffusionPipeline.from_pretrained(
    "Zero-nnkn/stable-diffusion-2-pokemon",
    torch_dtype=torch.float16,
)
pipe = pipe.to("cuda")
```

### Token Merging
[Token Merging for Stable Diffusion](https://github.com/dbolya/tomesd) is a technique for transformers speedup by merging redundant tokens.
```py
import torch
import tomesd
from diffusers import DiffusionPipeline

pipe = DiffusionPipeline.from_pretrained(
    "Zero-nnkn/stable-diffusion-2-pokemon",
).to("cuda")

# Apply ToMe with a 50% merging ratio
tomesd.apply_patch(pipe, ratio=0.5) # Can also use pipe.unet in place of pipe here
```

### Memory Efficient Attention
Use [FlashAttention](https://github.com/Dao-AILab/flash-attention) with [xFormers](https://github.com/facebookresearch/xformers) (Only for GPU).
```py
from diffusers import DiffusionPipeline
import torch

pipe = DiffusionPipeline.from_pretrained(
    "Zero-nnkn/stable-diffusion-2-pokemon",
    use_safetensors=True,
).to("cuda")

pipe.enable_xformers_memory_efficient_attention()

# Disable
# pipe.disable_xformers_memory_efficient_attention()
```


## Serialization
Convert model to ONNX and OpenVINO format. [🤗 Optimum](https://github.com/huggingface/optimum) provides pipeline compatible with ONNX runtime and OpenVINO. You can see ways to export `StableDiffusionPipeline` in [export.ipynb](./export.ipynb)

Example 1: Export pytorch pipeline to ONNX
```py
from optimum.onnxruntime import ORTStableDiffusionPipeline

model_id = "Zero-nnkn/stable-diffusion-2-pokemon"
pipeline = ORTStableDiffusionPipeline.from_pretrained(model_id, export=True)
pipeline.save_pretrained("onnx")
```

Example 2: Export pytorch pipeline to OpenVINO. Note that this pipeline can only execute on Intel devices (CPU or GPU).
```py
from optimum.intel import OVStableDiffusionPipeline

model_id = "Zero-nnkn/stable-diffusion-2-pokemon"
pipeline = OVStableDiffusionPipeline.from_pretrained(model_id, export=True)

pipeline.save_pretrained("openvino")
```

## Other tools
- [Meta AITemplate](https://github.com/facebookincubator/AITemplate)
- [Microsoft DeepSpeed](https://github.com/microsoft/DeepSpeed)