# Fine-tune Stable Diffusion Text-to-Image
## Prepare
**Install requirements**
```sh
pip install --upgrade diffusers[torch]
pip install datasets
pip install transformers
pip install xformers
pip install bitsandbytes
```

**Optional: Initialize Accelerate**
```sh
accelerate config # Setuo your config
```

**Optinal: Login Huggingface** (if you want to push your model to hub)
```sh
from huggingface_hub import notebook_login
notebook_login()
```

## Pipeline Fine-tuning
We fine-tune [Stable Diffusion 2.1](https://huggingface.co/stabilityai/stable-diffusion-2-1-base) using the [Pokemon dataset](https://huggingface.co/datasets/lambdalabs/pokemon-blip-captions) with [🤗 diffusers](https://github.com/huggingface/diffusers). You can see more example in [diffusers fine-tuning](https://github.com/huggingface/diffusers/tree/main/examples/text_to_image) or [LambdaLabsML diffusion fine-tuning](https://github.com/LambdaLabsML/examples/tree/main/stable-diffusion-finetuning)

We found that it is possible to fine-tune the pipeline on a Nvidia T4 GPU (Google Colab) / Nvidia P100 GPU (Kaggle). You can see [our model](https://huggingface.co/Zero-nnkn/stable-diffusion-2-pokemon) trained 5000 steps with batch size 1 (~ 5 hours P100).

```sh
accelerate launch --mixed_precision="fp16" train_text_to_image.py \
  --pretrained_model_name_or_path="stabilityai/stable-diffusion-2-1-base" \
  --dataset_name="lambdalabs/pokemon-blip-captions" \
  --use_ema \
  --resolution=512 --center_crop --random_flip \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --gradient_checkpointing \
  --max_train_steps=5000 \
  --checkpointing_steps=2000 \
  --learning_rate=2e-06 \
  --max_grad_norm=1 \
  --use_8bit_adam \
  --enable_xformers_memory_efficient_attention \
  --lr_scheduler="constant" --lr_warmup_steps=0 \
  --output_dir="stable-diffusion-2-pokemon" \
  --hub_model_id="your_hub_model_id" \
  --push_to_hub \
  --resume_from_checkpoint="latest"
```

We also finetune our stable diffusion by using Low Rank Adaptation (LoRA) with Sophia optimizer to boost up the training time as well as to reduce the GPU memory usage for our model.

```
cd diffusers/examples/text_to_image/
export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export OUTPUT_DIR="/sddata/finetune/lora/pokemon"
export HUB_MODEL_ID="sophia-pokemon-lora"
export DATASET_NAME="lambdalabs/pokemon-blip-captions"

accelerate launch --mixed_precision="fp16" train_text_to_image_lora.py
                  --pretrained_model_name_or_path=$MODEL_NAME \
                  --dataset_name=$DATASET_NAME  \
                  --dataloader_num_workers=8  \
                  --resolution=512 \
                  --center_crop \
                  --random_flip \
                  --train_batch_size=1 \
                  --gradient_accumulation_steps=4 \
                  --max_train_steps=15000 \
                  --learning_rate=1e-04 \
                  --max_grad_norm=1 \
                  --lr_scheduler="cosine" \
                  --lr_warmup_steps=0 \
                  --output_dir=${OUTPUT_DIR} \
                  --push_to_hub \
                  --hub_model_id=${HUB_MODEL_ID} \
                  --report_to=wandb \
                  --checkpointing_steps=500 \
                  --validation_prompt="Totoro" \
                  --seed=1332
                  --enable_xformers_memory_efficient_attention 
```
```

Note some important arguments (See [here](https://huggingface.co/docs/transformers/perf_train_gpu_one) for more detailts):
- `mixed_precision`: Mixed precision training.
- `gradient_accumulation_steps`: Number of updates steps to accumulate before performing a backward/update pass.
- `gradient_checkpointing`: Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.
- `use_8bit_adam`: Use 8-bit version of Adam for training.
- `enable_xformers_memory_efficient_attention`: Use [xFormers](https://github.com/facebookresearch/xformers) for training (recommend).
