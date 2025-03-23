# Survey of Video Diffusion Models: Foundations, Implementations, and Applications


<div style="text-align:center; font-size: 18px;">
    <p>
    <a href="https://yimuwangcs.github.io">Yimu Wang</a>, 
    <a href="to be updated" >Xuye Liu,</a>
    <a href="hto be updated" >Wei Pang,</a>
    <a href="to be updated" >Li Ma,</a>
    <a href="to be updated" >Shuai Yuan,</a>
    <a href="to be updated" >Paul Debevec,</a>
    <a href="to be updated" >Ning Yu</a>
     </p>
</div>

<p align="center">
(Yimu, Xuye, Wei, Li, and Shuai contributed equally. Ning is the corresponding author.)
</p>

- [News] <span style="color:red;"> **Our survey is on Arxiv now.**</span>


## Contact
If you have any suggestions or find our work helpful, feel free to contact us

Homepage: [Yimu Wang](https://yimuwangcs.github.io)

Email: yimu.wang@uwaterloo.ca

If you find our survey is useful in your research or applications, please consider giving us a star 🌟 and citing it by the following BibTeX entry.

```
To update
```

## Table of Contents 

- [Survey of Video Diffusion Models: Foundations, Implementations, and Applications](#survey-of-video-diffusion-models-foundations-implementations-and-applications)
  - [Contact](#contact)
  - [Table of Contents](#table-of-contents)
      - [Table samples](#table-samples)
- [Foundation](#foundation)
  - [Video generative paradigm](#video-generative-paradigm)
    - [GAN video models](#gan-video-models)
    - [Auto-regressive video models](#auto-regressive-video-models)
    - [Video diffusion models](#video-diffusion-models)
    - [Auto-regressive video diffusion models](#auto-regressive-video-diffusion-models)
  - [Learning foundation](#learning-foundation)
    - [Denoising diffusion probabilistic models (DDPM)](#denoising-diffusion-probabilistic-models-ddpm)
    - [Denoising diffusion implicit models (DDIM)](#denoising-diffusion-implicit-models-ddim)
    - [Elucidated diffusion models (EDM)](#elucidated-diffusion-models-edm)
    - [Flow matching and rectified flow](#flow-matching-and-rectified-flow)
    - [Learning from feedback and reward models](#learning-from-feedback-and-reward-models)
    - [One-shot and few-shot learning](#one-shot-and-few-shot-learning)
    - [Training-free methods](#training-free-methods)
    - [Token learning](#token-learning)
  - [Guidance](#guidance)
    - [Classifier guidance](#classifier-guidance)
    - [Classifier-free guidance](#classifier-free-guidance)
  - [Diffusion model frameworks](#diffusion-model-frameworks)
    - [Pixel diffusion and latent diffusion](#pixel-diffusion-and-latent-diffusion)
    - [Optical-flow-based diffusion models](#optical-flow-based-diffusion-models)
    - [Noise scheduling](#noise-scheduling)
    - [Agent-based diffusion models](#agent-based-diffusion-models)
  - [Architecture](#architecture)
    - [UNet](#unet)
    - [Diffusion transformers](#diffusion-transformers)
    - [VAE for latent space compression](#vae-for-latent-space-compression)
    - [Text encoder](#text-encoder)
- [Implementation](#implementation)
  - [Datasets](#datasets)
  - [Training engineering](#training-engineering)
  - [Evaluation metrics and benchmarking findings](#evaluation-metrics-and-benchmarking-findings)
  - [Industry solutions](#industry-solutions)
- [Applications](#applications)
  - [Conditions](#conditions)
    - [Image condition](#image-condition)
    - [Spatial condition](#spatial-condition)
    - [Camera parameter condition](#camera-parameter-condition)
    - [Audio condition](#audio-condition)
    - [High-level video condition](#high-level-video-condition)
    - [Other conditions](#other-conditions)
  - [Enhancement](#enhancement)
    - [Video denoising and deblurring](#video-denoising-and-deblurring)
    - [Video inpainting](#video-inpainting)
    - [Video interpolation and extrapolation/prediction](#video-interpolation-and-extrapolationprediction)
    - [Video super-resolution](#video-super-resolution)
    - [Combining multiple video enhancement tasks](#combining-multiple-video-enhancement-tasks)
  - [Personalization](#personalization)
  - [Consistency](#consistency)
  - [Long video](#long-video)
  - [3D-aware video diffusion](#3d-aware-video-diffusion)
    - [Training on 3D dataset](#training-on-3d-dataset)
    - [Architecture for 3D diffusion models](#architecture-for-3d-diffusion-models)
    - [Camera conditioning](#camera-conditioning)
    - [Inference-time tricks](#inference-time-tricks)
- [Benefits to other domains](#benefits-to-other-domains)
  - [Video representation learning](#video-representation-learning)
  - [Video retrieval](#video-retrieval)
  - [Video QA and captioning](#video-qa-and-captioning)
  - [3D and 4D generation](#3d-and-4d-generation)
    - [Video diffusion for 3D generation](#video-diffusion-for-3d-generation)
    - [Video diffusion for 4D generation](#video-diffusion-for-4d-generation)

#### Table samples
| Title | arXiv | Github| WebSite | Pub. & Date
|:-----:|:-----:|:-----:|:-----:|:-----:|
| [Identity-Preserving Text-to-Video Generation by Frequency Decomposition](https://arxiv.org/abs/2411.17440) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2411.17440) |[![Star](https://img.shields.io/github/stars/PKU-YuanGroup/ConsisID.svg?style=social&label=Star)](https://github.com/PKU-YuanGroup/ConsisID)|[![Website](https://img.shields.io/badge/Website-9cf)](https://pku-yuangroup.github.io/ConsisID/) | Nov., 2024



# Foundation

## Video generative paradigm

### GAN video models

### Auto-regressive video models

### Video diffusion models

### Auto-regressive video diffusion models

## Learning foundation

### Denoising diffusion probabilistic models (DDPM)

### Denoising diffusion implicit models (DDIM)

### Elucidated diffusion models (EDM)

### Flow matching and rectified flow

### Learning from feedback and reward models

### One-shot and few-shot learning

### Training-free methods

### Token learning

## Guidance

### Classifier guidance

### Classifier-free guidance

## Diffusion model frameworks

### Pixel diffusion and latent diffusion

### Optical-flow-based diffusion models

### Noise scheduling

### Agent-based diffusion models

## Architecture

### UNet

### Diffusion transformers

### VAE for latent space compression

### Text encoder

# Implementation

## Datasets

## Training engineering

## Evaluation metrics and benchmarking findings

## Industry solutions

# Applications

## Conditions

### Image condition

### Spatial condition

### Camera parameter condition

### Audio condition

### High-level video condition

### Other conditions

## Enhancement

### Video denoising and deblurring

### Video inpainting

### Video interpolation and extrapolation/prediction

### Video super-resolution

### Combining multiple video enhancement tasks

## Personalization

## Consistency

## Long video

## 3D-aware video diffusion

### Training on 3D dataset

### Architecture for 3D diffusion models

### Camera conditioning

### Inference-time tricks

# Benefits to other domains

## Video representation learning

## Video retrieval

## Video QA and captioning

## 3D and 4D generation

### Video diffusion for 3D generation

### Video diffusion for 4D generation
