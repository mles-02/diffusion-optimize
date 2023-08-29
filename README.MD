# Finetune, Optimize and Deploy with Stable Diffusion

<div align="center">
<p>
    <img src='./assets/teaser.jpg' width=30% class="center">
</p>

We introduce a complete pipeline from finetune to model optimization, finally deploying to the Cloud with a text-to-image model Stable Diffusion.

</div>


Authors:
- Khoi Nguyen: [zero-nnkn](https://github.com/zero-nnkn)
- Dang Quang: [QuangNguyen2609](https://github.com/QuangNguyen2609)

Advisors:
- Ba Ngoc: [bangoc123](https://github.com/bangoc123)

## 📝 Documentation
### 1. [Finetuning](./finetuning/README.md)
Currently, we investigate methods for finetune Stable Diffusion with [🤗 diffuers](https://github.com/huggingface/diffusers).

Due to hardware limitations, our finetune model cannot achieve the best results, and is only to demonstrate the feasibility of these methods.

### 2. [Optimization](./optimization/README.md)
We have researched and applied the optimization methods from [🤗 diffuers](https://github.com/huggingface/diffusers)
 and [🤗 optimum](https://github.com/huggingface/optimum).

### 3. [Deployment](./deployment/fastapi/README.md)
We provided instructions to deploy Stable Diffusion pipeline on Google Cloud. Note that we are currently only deploying on CPU. Work with GPU will be reserved for future works.


## ✔️ TODO
- [ ] Build UI for Text-to-Image
- [ ] Complete deployment with Triton Inference Server
- [ ] Investigate and write reports for optimization methods in GPU
- [ ] Evaluation of model quality

