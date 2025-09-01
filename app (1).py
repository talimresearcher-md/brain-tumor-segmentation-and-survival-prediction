import os
import tempfile
import matplotlib.pyplot as plt
import numpy as np
import nibabel as nib
from monai.losses import DiceLoss
from monai.inferers import sliding_window_inference
from monai import transforms
from monai.transforms import (
    LoadImaged,
    NormalizeIntensityd,
)
from monai.networks.nets import SwinUNETR
from monai import data
from functools import partial
import torch
import streamlit as st
import requests
from at import upload_image_to_imgur


from authenticator import login



if 'user_state' not in st.session_state:
    st.session_state.user_state = {'logged_in': False}

# Authentication step
login()  

# Only allow access to the app if the user is logged in
if st.session_state.user_state.get('logged_in', False):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    roi = (128, 128, 128)

    model = SwinUNETR(
        img_size=roi,
        in_channels=4,
        out_channels=3,
        feature_size=48,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        dropout_path_rate=0.0,
        use_checkpoint=True,
    ).to(device)

    model.load_state_dict(torch.load('model.pt')["state_dict"])
    model.eval()

    test_transform = transforms.Compose(
        [
            LoadImaged(keys=["image"]),  
            NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),  # Normalize intensity
        ]
    )

    st.title("Brain Tumor Segmentation using Swin UNETR")
    st.write("Upload MRI scans for T1, T1CE, T2, and FLAIR to predict brain tumor segmentation.")

    uploaded_files = st.file_uploader("Choose MRI scan files", type=["nii", "nii.gz"], accept_multiple_files=True)

    if uploaded_files and len(uploaded_files) == 4:
        st.write("Processing images...")

        temp_file_paths = []
        for uploaded_file in uploaded_files:
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
                tmp_file.write(uploaded_file.read())
                temp_file_paths.append(tmp_file.name)

        data_dict = [
            {
                "image": [
                    temp_file_paths[0],
                    temp_file_paths[1],
                    temp_file_paths[2],
                    temp_file_paths[3],
                ]
            }
        ]
        images = nib.load(temp_file_paths[0]).get_fdata()
        test_ds = data.Dataset(data=data_dict, transform=test_transform)

        test_loader = data.DataLoader(
            test_ds,
            batch_size=1,
            shuffle=False,
            num_workers=2,
            pin_memory=True,
        )

        model_inferer_test = partial(
            sliding_window_inference,
            roi_size=roi,
            sw_batch_size=1,
            predictor=model,
            overlap=0.6,
        )

        with torch.no_grad():
            for batch_data in test_loader:
                image = batch_data["image"].to(device)
                prob = torch.sigmoid(model_inferer_test(image))
                seg = prob[0].detach().cpu().numpy()
                seg = (seg > 0.5).astype(np.int8)
                seg_out = np.zeros((seg.shape[1], seg.shape[2], seg.shape[3]))
                seg_out[seg[1] == 1] = 2
                seg_out[seg[0] == 1] = 1
                seg_out[seg[2] == 1] = 4

        slice_num = seg_out.shape[2] // 2  
        plt.figure("Segmentation", (18, 6))
        plt.subplot(1, 3, 1)
        plt.title("Image")
        plt.imshow(images[:, :, slice_num], cmap="gray") 
        plt.subplot(1, 3, 2)
        plt.title("Segmentation")
        plt.imshow(seg_out[:, :, slice_num])
        plt.tight_layout()
        plt.savefig("segmentation.png")
        plt.close()


        try:
            image_url = upload_image_to_imgur('segmentation.png')
            st.write(f"Image uploaded and viewable [here]({image_url}).")
        except Exception as e:
            st.error(f"Error uploading image: {e}")

    else:
        st.write("Please upload exactly four files for T1, T1CE, T2, and FLAIR.")
else:
    st.write("Please log in to access the app.")
