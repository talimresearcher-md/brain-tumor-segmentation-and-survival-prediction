# 🧠 Brain Tumor Segmentation & Survival Prediction

This repository provides a **dual-model pipeline** for brain tumor analysis, integrating a **research-focused 3D ResUNet** with a **deployment-ready Swin UNETR**.  
It supports **MRI-based tumor segmentation (WT, TC, ET)** and **survival prediction**, offering both a **Jupyter notebook for experimentation** and a **Streamlit web app for real-time inference**.

---

## 📂 Repository Contents
- **`brain-tumor-seg-final.ipynb`** – Research notebook using a custom **3D ResUNet** for:
  - Training and evaluation on BraTS 2020
  - Segmentation (WT, TC, ET)
  - Feature extraction for tumor characteristics
  - Survival prediction modeling  
- **`app.py`** – Streamlit web app with a pretrained **Swin UNETR** for:
  - Real-time segmentation of user-uploaded MRI scans
  - Visualization of central axial slices with segmentation overlay
  - Automatic upload to Imgur for shareable output links
  - Secure login for controlled access

---

## 🧪 Model Overview

| Feature        | 3D ResUNet (Notebook)                | Swin UNETR (App)         |
|----------------|--------------------------------------|--------------------------|
| **Location**   | `brain-tumor-seg-final.ipynb`        | `app.py`                 |
| **Purpose**    | Training & Research                  | Inference & Deployment   |
| **Inputs**     | T1, T1CE, T2, FLAIR                  | T1, T1CE, T2, FLAIR      |
| **Outputs**    | WT, TC, ET segmentation              | WT, TC, ET segmentation  |
| **Metrics**    | Dice: WT **93%**, TC **80%**, ET **78%** | Optimized for live use   |
| **Extras**     | Survival prediction from features    | Imgur upload & preview   |

---

## 📊 Notebook Highlights (`brain-tumor-seg-final.ipynb`)
- Custom **3D ResUNet** architecture with **Dice + Cross Entropy loss**
- Multi-channel MRI inputs for richer spatial features
- End-to-end workflow:
  1. Data preprocessing (BraTS 2020)
  2. Tumor segmentation (WT, TC, ET)
  3. Feature extraction (size, intensity, shape)
  4. Survival prediction modeling

---

## 🌐 Web App Highlights (`app.py`)
- Built with **Streamlit** for an intuitive interface
- Uses pretrained **Swin UNETR** (from MONAI)  
- Workflow:
  1. Upload 4 aligned `.nii` or `.nii.gz` files: **T1, T1CE, T2, FLAIR**
  2. Segmentation overlay shown on central axial slice
  3. Results automatically uploaded to **Imgur** with shareable link  

---

## 📌 Key Metrics
- **3D ResUNet (Research Notebook):**
  - Dice Score → WT: **93%**, TC: **80%**, ET: **78%**  
- **Swin UNETR (Web App):**
  - Deployed for **real-time clinical-style segmentation**  

---

## 🙏 Acknowledgments
- **BraTS 2020 Dataset** for training & evaluation  
- **MONAI** for the Swin UNETR implementation  
- **Streamlit** for deployment framework  

