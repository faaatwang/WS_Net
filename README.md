# Wavelet-Guided Frequency-Domain Adaptive Learning (WS-Net)

Official PyTorch implementation of:

**Wavelet-Guided Frequency-Domain Adaptive Learning: Balancing Adversarial Defense and High-Fidelity Image Reconstruction**

---

## 🔥 Overview

Adversarial attacks introduce imperceptible perturbations that severely compromise deep neural networks.  
While reconstruction-based defenses offer model-agnostic robustness, they often sacrifice image fidelity.

We propose **WS-Net**, a **wavelet-guided frequency-domain adaptive denoising network** that:

- Decomposes images into low- and high-frequency components via **Discrete Wavelet Transform (DWT)**
- Applies denoising selectively to **noise-prone high-frequency bands**
- Preserves structural information critical for high-quality reconstruction

At the core is a **Swin Network Denoiser (SND)** using shifted-window self-attention to distinguish noise from details.

WS-Net effectively balances:

✅ Adversarial robustness  
✅ High-fidelity image reconstruction  

## 🧠 Method Pipeline

![img](Wavelet-Guided Frequency-Domain Adaptive Learning (WS-Net).assets/clip_image002.png)

---

## 📊 Experimental Results (ImageNet)

### Adversarial robustness 

| Setting        | Top-1 Accuracy |
| -------------- | -------------- |
| Clean Images   | **71.4%**      |
| Gaussian Noise | **70.2%**      |
| CW Attack      | **68.5%**      |
| DeepFool       | **67.9%**      |

### Image Quality

![img](Wavelet-Guided Frequency-Domain Adaptive Learning (WS-Net).assets/clip_image002-17699997824452.png)

| Metric | Value       |
| ------ | ----------- |
| PSNR   | **30.2 dB** |
| SSIM   | **0.87**    |

WS-Net consistently outperforms state-of-the-art defense methods.

---

## 📁 Project Structure

WS-Net/
 │
 ├── model/
 │   └── WS_Net.py
 │
 ├── training/
 │   └── TrainModel.py
 │
 ├── test.py
 ├── train.py
 └── README.md

## 🚀 Training

```shell
python train.py \
    --model WS_Net \
    --epochs 50 \
    --device cuda \
    --pretrained ""
```

## 🔍 Testing / Inference

```shell
python test.py \
    --img_path path/to/image.png \
    --model_path checkpoint/WS_Net.pth
```

## 📬 Contact

For questions or collaborations:

Email: hzw@gs.zzu.edu.cn