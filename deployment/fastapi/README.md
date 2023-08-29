<!-- lora-diffusion
finetuning-diffusion
finetuning-diffusion-tome
onnx-diffusion
onnx-diffusion-u8
openvino-diffusion -->

# Deploy Stable Diffusion on Google Cloud
We follow this guide: https://github.com/AUTOMATIC1111/stable-diffusion-webui/discussions/9131

**Note that in this example, we just deploy with cpu.**

## 0. Prerequisites
- Set up a Google Cloud billing account. It neccessery for you to create a VM with GPU (If you don't want to use GPU, you are ok with free trial).
- Setup a firewall rule for private access
    - **Targets**: `Specified target tags`; **Target tags**: `diffusion-demo` (You need to add this tag to VM instance).
    - **Source filter**: `IPv4 ranges` -> Put your IP range in **Source range** box (You can use `0.0.0.0/0` to open for everyone in the Internet).
    - **Protocols and ports**: `TCP` -> Put your port you want to open in **Ports** box (e.g `8080`)

## 1. Create VM Instance
- Create a VM instance based on your demand. Example:
    - 1 GPU NVIDIA T4
    - n1-standard-4 (4 vCPU, 15 GB memory)
    - Ubuntu 22.04 LTS
    - 128GB Storage

- In the **Firewall** section: `Allow HTTP traffic`


- Optional: Add your SSH key to instance:
    - Create your ssh-key in local machine with: `ssh-keygen`
    - Copy public key in `your_fine_name.pub` file and add to your instance: Click to instance -> Edit -> SSH Keys -> Add Item -> Paste your public key -> Save
    - Access your VM with SSH.

## 2. Deploy on VM Instance
### 2.1 Setup
- Python
```sh
sudo apt-get update
sudo apt install python3-pip
```

- GPU driver:
```sh
sudo apt install ubuntu-drivers-common
sudo ubuntu-drivers autoinstall

# Reboot
sudo reboot
```

- (Optional) Miniconda
```sh
curl -sL "https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh" > "Miniconda3.sh"

bash Miniconda3.sh
```

- Docker
```sh
sudo snap install docker
sudo snap start docker
```

### 2.2 Run with Docker
Install
```sh
git clone https://github.com/mles-02/diffustion-optimize.git
cd diffustion-optimize/deployment/fastapi
```

Create a `.env` file (we provided example `.env.example`). Change HOST and PORT in `Dockerfile.cpu` and `docker-compose.yaml` file (note that port is the port that you exposed in fireware rule). Then run `docker-compose`:
```sh
sudo docker-compose -d --build
```

View log with
```sh
sudo docker-compose logs -f
```