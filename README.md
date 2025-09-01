# 🧠 Brain Tumor Segmentation & Survival Prediction

This repository provides a **dual-model pipeline** for brain tumor analysis, combining the **research-focused 3D ResUNet** with the **deployment-ready Swin UNETR**.  
It enables **MRI-based tumor segmentation (WT, TC, ET)** and **survival prediction** with both a Jupyter notebook for experimentation and a Streamlit web app for real-time inference.



## ✨ Features
- 🧪 **3D ResUNet** for training, evaluation & survival modeling (BraTS 2020 dataset).  
- ⚡ **Swin UNETR (MONAI)** for real-time segmentation in Streamlit.  
- 📊 **Survival prediction** using segmentation-derived features.  
- 🌐 **Web app** with Imgur integration for shareable results.  

---

## ⚙️ Installation & Setup
Clone the repository and install dependencies:  
```bash
git clone https://github.com/username/brain-tumor-seg.git
cd brain-tumor-seg
pip install -r requirements.txt

