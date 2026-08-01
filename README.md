# AI-Generated Image Detector

A deep learning system that classifies images as **REAL** or **AI-GENERATED**. The model combines a CNN operating on raw pixels with a second CNN branch analyzing the image's frequency-domain (FFT) representation, since generative models leave subtle spectral artifacts that aren't visible to the naked eye but are statistically detectable.

**Live demo:** https://deepfake-detector-c4cdq8sjhqywu5y8zgjgak.streamlit.app/

## Demo video

https://github.com/user-attachments/assets/2b8c4d77-b5fc-4429-8235-cc98a18ccfe2

## Overview

- **Dataset:** CIFAKE — 120,000 images (real CIFAR-10 photos vs. Stable Diffusion-generated equivalents), 32x32 resolution.
- **Approach:** Dual-branch CNN — one branch processes the RGB image, the other processes its FFT magnitude spectrum; both are fused before the final classification layer.
- **Result:** 95.11% test accuracy (up from a 94.42% RGB-only baseline), confirming that frequency-domain artifacts carry real, exploitable signal for this task.

## Repository structure

- `training/` — Jupyter notebook used to train the model (data loading, baseline CNN, FFT analysis, dual-branch model, evaluation, Grad-CAM)
- `backend/` — FastAPI backend + trained model weights (for local/API-based deployment)
- `streamlit_app.py` — Streamlit app used for the live public demo
- `requirements.txt` — Dependencies for the Streamlit deployment

## Key results

| Model | Test Accuracy | Test Loss |
|---|---|---|
| Baseline CNN (RGB only) | 94.42% | 0.1555 |
| Dual-branch CNN (RGB + FFT) | 95.11% | 0.1383 |

## Running locally

**Streamlit app:**

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

**FastAPI backend + web frontend:**

```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

Then open `http://127.0.0.1:8000`.

## Methodology summary

1. Baseline CNN trained on raw RGB images.
2. Averaged FFT spectra computed across 1000+ real and fake images, revealing a systematic difference in mid- and high-frequency energy between the two classes.
3. A second CNN branch added to process the FFT magnitude map directly, fused with the RGB branch before classification.
4. Grad-CAM used to interpret model attention, showing the classifier often relies on background/corner artifacts rather than semantic object features — consistent with the frequency-domain finding.

## Limitations

- Trained and evaluated at 32x32 resolution (CIFAR-scale); performance on higher-resolution or real-world photos is untested.
- Trained exclusively on Stable Diffusion-generated images; generalization to other generators (Midjourney, DALL-E, GANs) has not been verified.