# Survey of Video Diffusion Models: Foundations, Implementations, and Applications

<p align="center">
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
</p>

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
| []() | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)]() | [![Star](https://img.shields.io/github/stars/PKU-YuanGroup/ConsisID.svg?style=social&label=Star)]()|[![Website](https://img.shields.io/badge/Website-9cf)]() | ICLR 2024 |



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

| Title | arXiv | Github| WebSite | Pub. & Date
|:-----:|:-----:|:-----:|:-----:|:-----:|
| [StyleGAN-V: A Continuous Video Generator with the Price, Image Quality and Perks of StyleGAN2](https://openaccess.thecvf.com/content/CVPR2022/papers/Skorokhodov_StyleGAN-V_A_Continuous_Video_Generator_With_the_Price_Image_Quality_CVPR_2022_paper.pdf) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2112.14683) | [![Star](https://img.shields.io/github/stars/PKU-YuanGroup/ConsisID.svg?style=social&label=Star)](https://github.com/universome/stylegan-v)|[![Website](https://img.shields.io/badge/Website-9cf)](https://skor.sh/stylegan-v.html) | CVPR 2022 |
| [Conditional Image-to-Video Generation with Latent Flow Diffusion Models](https://openaccess.thecvf.com/content/CVPR2023/papers/Ni_Conditional_Image-to-Video_Generation_With_Latent_Flow_Diffusion_Models_CVPR_2023_paper.pdf) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2303.13744) | [![Star](https://img.shields.io/github/stars/PKU-YuanGroup/ConsisID.svg?style=social&label=Star)](https://github.com/nihaomiao/CVPR23_LFDM)| - | CVPR 2023 |
| [MoStGAN-V: Video Generation with Temporal Motion Styles](https://openaccess.thecvf.com/content/CVPR2023/papers/Shen_MoStGAN-V_Video_Generation_With_Temporal_Motion_Styles_CVPR_2023_paper.pdf) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2304.02777) | [![Star](https://img.shields.io/github/stars/PKU-YuanGroup/ConsisID.svg?style=social&label=Star)](https://github.com/xiaoqian-shen/MoStGAN-V)| - | CVPR 2023 |
| [Stablevideo: Text-driven consistency-aware diffusion video editing](https://openaccess.thecvf.com/content/ICCV2023/papers/Chai_StableVideo_Text-driven_Consistency-aware_Diffusion_Video_Editing_ICCV_2023_paper.pdf) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2308.09592) |[![Star](https://img.shields.io/github/stars/PKU-YuanGroup/ConsisID.svg?style=social&label=Star)](https://github.com/rese1f/StableVideo)|[![HuggingFace Demo](https://img.shields.io/badge/Website-9cf)](https://huggingface.co/spaces/wchai/StableVideo) | ICCV 2023 |
| [Preserve your own correlation: A noise prior for video diffusion models](https://openaccess.thecvf.com/content/ICCV2023/papers/Ge_Preserve_Your_Own_Correlation_A_Noise_Prior_for_Video_Diffusion_ICCV_2023_paper.pdf) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2305.10474) | - |[![Website](https://img.shields.io/badge/Website-9cf)](https://research.nvidia.com/labs/dir/pyoco/) | ICCV 2023 |
| [Scenescape: Text-driven consistent scene generation](https://openreview.net/forum?id=NU2kGsA4TT) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2302.01133) | [![Star](https://img.shields.io/github/stars/PKU-YuanGroup/ConsisID.svg?style=social&label=Star)](https://github.com/RafailFridman/SceneScape)|[![Website](https://img.shields.io/badge/Website-9cf)](https://scenescape.github.io) | NeurIPS 2023 |
| [How i warped your noise: a temporally-correlated noise prior for diffusion models](https://openreview.net/forum?id=pzElnMrgSD) | - | - | [![Website](https://img.shields.io/badge/Website-9cf)](https://warpyournoise.github.io) | ICLR 2024 |
| [Tokenflow: Consistent diffusion features for consistent video editing](https://openreview.net/forum?id=lKK50q2MtV) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2307.10373) | [![Star](https://img.shields.io/github/stars/PKU-YuanGroup/ConsisID.svg?style=social&label=Star)](https://github.com/omerbt/TokenFlow)|[![Website](https://img.shields.io/badge/Website-9cf)](https://diffusion-tokenflow.github.io) | ICLR 2024 |
| [Seine: Short-to-long video diffusion model for generative transition and prediction.](https://openreview.net/forum?id=FNq3nIvP4F) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2310.20700) | [![Star](https://img.shields.io/github/stars/PKU-YuanGroup/ConsisID.svg?style=social&label=Star)](https://github.com/Vchitect/SEINE)|[![Website](https://img.shields.io/badge/Website-9cf)](https://vchitect.github.io/SEINE-project/) | ICLR 2024 |
| [VideoBooth: Diffusion-based Video Generation with Image Prompts](https://openaccess.thecvf.com/content/CVPR2024/papers/Jiang_VideoBooth_Diffusion-based_Video_Generation_with_Image_Prompts_CVPR_2024_paper.pdf) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2312.00777) | [![Star](https://img.shields.io/github/stars/PKU-YuanGroup/ConsisID.svg?style=social&label=Star)](https://github.com/Vchitect/VideoBooth)|[![Website](https://img.shields.io/badge/Website-9cf)](https://vchitect.github.io/VideoBooth-project/) | CVPR 2024 |
| [VidToMe: Video Token Merging for Zero-Shot Video Editing](https://openaccess.thecvf.com/content/CVPR2024/papers/Li_VidToMe_Video_Token_Merging_for_Zero-Shot_Video_Editing_CVPR_2024_paper.pdf) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2312.10656) | [![Star](https://img.shields.io/github/stars/PKU-YuanGroup/ConsisID.svg?style=social&label=Star)](https://github.com/lixirui142/VidToMe)|[![Website](https://img.shields.io/badge/Website-9cf)](https://vidtome-diffusion.github.io) | CVPR 2024 |
| [Streetscapes: Large-scale consistent street view generation using autoregressive video diffusion](https://dl.acm.org/doi/10.1145/3641519.3657513) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2407.13759) | - |[![Website](https://img.shields.io/badge/Website-9cf)](https://boyangdeng.com/streetscapes/) | SIGGRAPH 2024 |
| [Motion-i2v: Consistent and controllable image-to-video generation with explicit motion modeling](https://dl.acm.org/doi/10.1145/3641519.3657497) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2401.15977) | [![Star](https://img.shields.io/github/stars/PKU-YuanGroup/ConsisID.svg?style=social&label=Star)](https://github.com/G-U-N/Motion-I2V)|[![Website](https://img.shields.io/badge/Website-9cf)](https://xiaoyushi97.github.io/Motion-I2V/) | SIGGRAPH 2024 |
| [Consisti2v: Enhancing visual consistency for image-to-video generation](https://openreview.net/forum?id=vqniLmUDvj) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2402.04324) | [![Star](https://img.shields.io/github/stars/PKU-YuanGroup/ConsisID.svg?style=social&label=Star)](https://github.com/TIGER-AI-Lab/ConsistI2V)|[![Website](https://img.shields.io/badge/Website-9cf)](https://tiger-ai-lab.github.io/ConsistI2V/) | TMLR 2024 |
| [StreamingT2V: Consistent, Dynamic, and Extendable Long Video Generation from Text](https://arxiv.org/abs/2403.14773) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2403.14773) | [![Star](https://img.shields.io/github/stars/PKU-YuanGroup/ConsisID.svg?style=social&label=Star)](https://github.com/Picsart-AI-Research/StreamingT2V)|[![Website](https://img.shields.io/badge/Website-9cf)](https://streamingt2v.github.io) | Arxiv 2024 |
| [Flexifilm: Long video generation with flexible conditions](https://arxiv.org/abs/2404.18620) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2404.18620) | [![Star](https://img.shields.io/github/stars/PKU-YuanGroup/ConsisID.svg?style=social&label=Star)](https://github.com/Y-ichen/FlexiFilm)|[![Website](https://img.shields.io/badge/Website-9cf)](https://y-ichen.github.io/FlexiFilm-Page/) | Arxiv 2024 |
| [Cinemo: Consistent and Controllable Image Animation with Motion Diffusion Models  ](https://arxiv.org/abs/2407.15642) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2407.15642) | [![Star](https://img.shields.io/github/stars/PKU-YuanGroup/ConsisID.svg?style=social&label=Star)](https://github.com/maxin-cn/Cinemo)|[![Website](https://img.shields.io/badge/Website-9cf)](https://maxin-cn.github.io/cinemo_project/) | CVPR 2025 |

| [GLOBER: Coherent Non-autoregressive Video Generation via GLOBal Guided Video DecodER](https://openreview.net/forum?id=TRbklCR2ZW&referrer=%5Bthe%20profile%20of%20Jing%20Liu%5D(%2Fprofile%3Fid%3D~Jing_Liu1)) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2309.13274) | [![Star](https://img.shields.io/github/stars/PKU-YuanGroup/ConsisID.svg?style=social&label=Star)](https://github.com/iva-mzsun/GLOBER)| - | NeurIPS 2023 |
| [MOSO: Decomposing MOtion, Scene and Object for Video Prediction](https://openaccess.thecvf.com/content/CVPR2023/papers/Sun_MOSO_Decomposing_MOtion_Scene_and_Object_for_Video_Prediction_CVPR_2023_paper.pdf) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2303.03684) | [![Star](https://img.shields.io/github/stars/PKU-YuanGroup/ConsisID.svg?style=social&label=Star)](https://github.com/iva-mzsun/MOSO)| - | CVPR 2023 |
| [EMO: Emote Portrait Alive - Generating Expressive Portrait Videos with Audio2Video Diffusion Model under Weak Conditions](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/11028.pdf) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2402.17485) | [![Star](https://img.shields.io/github/stars/PKU-YuanGroup/ConsisID.svg?style=social&label=Star)](https://github.com/HumanAIGC/EMO)|[![Website](https://img.shields.io/badge/Website-9cf)](https://humanaigc.github.io/emote-portrait-alive/) | ECCV 2024 |
| [VideoComposer: Compositional Video Synthesis with Motion Controllability](https://openreview.net/forum?id=h4r00NGkjR) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2306.02018) | [![Star](https://img.shields.io/github/stars/PKU-YuanGroup/ConsisID.svg?style=social&label=Star)](https://github.com/ali-vilab/videocomposer)|[![Website](https://img.shields.io/badge/Website-9cf)](https://videocomposer.github.io) | NeurIPS 2023 |
| [Generating Videos with Dynamics-aware Implicit Generative Adversarial Networks](https://openreview.net/forum?id=Czsdv-S4-w9) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2202.10571) | [![Star](https://img.shields.io/github/stars/PKU-YuanGroup/ConsisID.svg?style=social&label=Star)](https://github.com/sihyun-yu/digan)|[![Website](https://img.shields.io/badge/Website-9cf)](https://sihyun.me/digan/) | ICLR 2022 |
| [CAMEL: CAusal Motion Enhancement tailored for Lifting
Text-driven Video Editing](https://openaccess.thecvf.com/content/CVPR2024/papers/Zhang_CAMEL_CAusal_Motion_Enhancement_Tailored_for_Lifting_Text-driven_Video_Editing_CVPR_2024_paper.pdf) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=10655591) | [![Star](https://img.shields.io/github/stars/PKU-YuanGroup/ConsisID.svg?style=social&label=Star)](https://github.com/zhangguiwei610/CAMEL)| - | CVPR 2024 |
| [Towards Smooth Video Composition](https://openreview.net/forum?id=W918Ora75q) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2212.07413) | [![Star](https://img.shields.io/github/stars/PKU-YuanGroup/ConsisID.svg?style=social&label=Star)](https://github.com/genforce/StyleSV)|[![Website](https://img.shields.io/badge/Website-9cf)](https://genforce.github.io/StyleSV/) | ICLR 2023 |
| [TRIP: Temporal Residual Learning with Image Noise Prior for Image-to-Vieo Diffusion Models](https://openaccess.thecvf.com/content/CVPR2024/papers/Zhang_TRIP_Temporal_Residual_Learning_with_Image_Noise_Prior_for_Image-to-Video_CVPR_2024_paper.pdf) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2403.17005) | - |[![Website](https://img.shields.io/badge/Website-9cf)](https://trip-i2v.github.io/TRIP/) | CVPR 2024 |
| [StoryDiffusion: Consistent Self-Attention for Long-Range Image and Video Generation](https://openreview.net/pdf/992e1d8483d14f713dff3f74f664f722bfa72930.pdf) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2405.01434) | [![Star](https://img.shields.io/github/stars/PKU-YuanGroup/ConsisID.svg?style=social&label=Star)](https://github.com/HVision-NKU/StoryDiffusion)|[![Website](https://img.shields.io/badge/Website-9cf)](https://storydiffusion.github.io) | NeurIPS 2024 |




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
