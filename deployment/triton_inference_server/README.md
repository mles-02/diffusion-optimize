# Optimized Stable Diffusion

## Pipeline
Before starting, clone this repository and navigate to the root folder. Use three different terminals for an easier user experience.

### Step 1: Prepare the Server Environment
* First, run the Triton Inference Server Container.
```sh
# Replace yy.mm with year and month of release. Eg. 23.07
docker run --gpus=all -it --shm-size=256m --rm -p8000:8000 -p8001:8001 -p8002:8002 -v ${PWD}:/workspace/ -v ${PWD}/model_repository:/models nvcr.io/nvidia/tritonserver:yy.mm-py3 bash
```
* Next, install all the dependencies required by the models running in the python backend and login with your [huggingface token](https://huggingface.co/settings/tokens)(Account on [HuggingFace](https://huggingface.co/) is required).

```sh
# PyTorch & Transformers Lib
pip install torch torchvision torchaudio
pip install transformers ftfy scipy accelerate
pip install diffusers==0.9.0
pip install transformers[onnxruntime]
huggingface-cli login
```

### Step 2: Exporting and converting the models
Use the NGC PyTorch container, to export and convert the models.

```sh
docker run -it --gpus all -p 8888:8888 -v ${PWD}:/mount nvcr.io/nvidia/pytorch:yy.mm-py3

pip install transformers ftfy scipy
pip install transformers[onnxruntime]
pip install diffusers==0.9.0
huggingface-cli login
cd /mount
python export.py

# Accelerating VAE with TensorRT
trtexec --onnx=vae.onnx --saveEngine=vae.plan --minShapes=latent_sample:1x4x64x64 --optShapes=latent_sample:4x4x64x64 --maxShapes=latent_sample:8x4x64x64 --fp16

# Place the models in the model repository
mkdir model_repository/vae/1
mkdir model_repository/text_encoder/1
mv vae.plan model_repository/vae/1/model.plan
mv encoder.onnx model_repository/text_encoder/1/model.onnx
```

### Step 3: Launch the Server
From the server container, launch the Triton Inference Server.
```sh
tritonserver --model-repository=/models
```

### Step 4: Run the client
Use the client container and run the client.
```sh
docker run -it --net=host -v ${PWD}:/workspace/ nvcr.io/nvidia/tritonserver:yy.mm-py3-sdk bash

# Client with no GUI
python3 client.py

# Client with GUI
pip install gradio packaging
python3 gui/client.py --triton_url="localhost:8001"
```
Note: First Inference query may take more time than successive queries
