# Wavelet-Guided Frequency-Domain Adaptive Learning (WS-Net)

This code is the official implementation of the manuscript:

**Wavelet-Guided Frequency-Domain Adaptive Learning: Balancing Adversarial Defense and High-Fidelity Image Reconstruction**

submitted to *The Visual Computer* .

If you use this code in your research, please kindly cite the corresponding manuscript.

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

<img width="576" height="218" alt="clip_image002" src="https://github.com/user-attachments/assets/3d7fc17a-f0cf-4d1a-a476-4be699a29513" />


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

<img width="552" height="533" alt="clip_image002-17699997824452" src="https://github.com/user-attachments/assets/03342cc1-6d36-4443-b34a-5ecba51dc8d4" />


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


