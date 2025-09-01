# 🧠 Brain Tumor Segmentation & Survival Prediction

This repository provides a dual-model pipeline for **brain tumor segmentation** and **survival prediction**, combining the robustness of a custom **3D ResUNet** (in research development) with the performance of **Swin UNETR** (in deployment). It includes a Jupyter notebook for experimentation and a Streamlit web app for live inference.

---

## 📂 Contents

- `brain-tumor-seg-final.ipynb` – Development notebook using a custom **3D ResUNet** for training, evaluation, and survival prediction.
- `app.py` – Streamlit web app using a pretrained **Swin UNETR** model for real-time segmentation on user-uploaded MRI scans.

---

## 🧪 Model Overview

| Feature                   | 3D ResUNet (Notebook)           | Swin UNETR (App)               |
|---------------------------|---------------------------------|------------------------------- |
| Location                  | `brain-tumor-seg-final.ipynb`   | `app.py`                        |
| Purpose                   | Training & Research             | Inference & Deployment          |
| Input Channels            | T1, T1CE, T2, FLAIR             | T1, T1CE, T2, FLAIR            |
| Outputs                   | WT, TC, ET segmentation         | WT, TC, ET segmentation         |
| Metrics Achieved          | Dice: WT 93%, TC 80%, ET 78%    | Model deployed for practical use|
| Extra                     | Survival prediction from features | Imgur upload & slice preview  |

---

## 📊 Notebook: `brain-tumor-seg-final.ipynb`

- Uses **custom 3D ResUNet** architecture
- Trains on BraTS 2020 dataset
- Applies multi-channel input for rich spatial features
- Custom loss function: Dice + Cross Entropy
- Performs segmentation + extracts tumor features
- Builds survival prediction model based on segmentation-derived data

---

## 🌐 Web App: `app.py`

- Built with **Streamlit** for user-friendly interaction
- Utilizes **Swin UNETR** (from MONAI) for segmentation
- Requires upload of 4 aligned `.nii` or `.nii.gz` files:
  - T1, T1CE, T2, FLAIR
- Displays central axial slice with segmentation overlay
- Automatically uploads the output slice to **Imgur** and returns a shareable link
- Login access for secure usage

---
