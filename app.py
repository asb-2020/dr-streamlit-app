import streamlit as st
from PIL import Image
import torch
import joblib
import numpy as np
import cv2
import pydicom
from torchvision import models
import torch.nn as nn
from torchvision.models import resnet50
import json
import gdown
import os

from utils.preprocessing import preprocess_image, extract_physio_features
from utils.gradcam import GradCAM
from collections import OrderedDict

# Google Drive helper
GDRIVE_FILE_ID = "1x8DMBMNwRpsFAzo4U_63UL3oykyhlBDp"
GDRIVE_DEST_PATH = "models/image_model.pth"

def gdrive_download(file_id: str, dest_path: str):
    """Download file from Google Drive if it doesn't exist locally."""
    if not os.path.exists(dest_path):
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        url = f"https://drive.google.com/uc?id={file_id}"
        st.info("Downloading image model from Google Drive...")
        gdown.download(url, dest_path, quiet=False)
    return dest_path

# --- Load models and scalers ---

@st.cache_resource(show_spinner=False)
def load_image_model(checkpoint_path=None):
    model = resnet50(weights=None)
    model.fc = nn.Sequential(
        nn.Linear(2048, 1024),
        nn.ReLU(inplace=False),
        nn.BatchNorm1d(1024),
        nn.Dropout(p=0.25),
        nn.Linear(1024, 512),
        nn.ReLU(inplace=False),
        nn.BatchNorm1d(512),
        nn.Dropout(p=0.25),
        nn.Linear(512, 2)
    )

    if checkpoint_path is not None:
        checkpoint_path = gdrive_download(GDRIVE_FILE_ID, checkpoint_path)
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        state_dict = checkpoint.get('model', checkpoint)
        model.load_state_dict(state_dict)
    model.eval()
    return model
@st.cache_resource(show_spinner=False)
def load_physio_model():
    return joblib.load("models/stacked_model.joblib")

@st.cache_resource(show_spinner=False)
def load_scaler():
    return joblib.load("models/scaler.joblib")

def load_and_prepare_image(uploaded_img):
    try:
        if uploaded_img.name.endswith(".dcm"):
            dicom = pydicom.dcmread(uploaded_img)
            pixel_array = dicom.pixel_array
            if len(pixel_array.shape) == 2:  # grayscale
                pixel_array = np.stack([pixel_array]*3, axis=-1)
            elif pixel_array.shape[2] == 1:
                pixel_array = np.repeat(pixel_array, 3, axis=-1)
            pixel_array = (pixel_array / pixel_array.max()) * 255.0
            pixel_array = pixel_array.astype(np.uint8)
            pil_img = Image.fromarray(pixel_array)
        else:
            pil_img = Image.open(uploaded_img).convert("RGB")

        input_tensor = preprocess_image(pil_img)

        return input_tensor, pil_img

    except Exception as e:
        st.error(f"Failed to load or preprocess image: {e}")
        return None, None

def run_image_inference(model, input_tensor):
    with torch.no_grad():
        logits = model(input_tensor)
        probs = torch.softmax(logits, dim=1)
        pred_class = torch.argmax(probs, dim=1).item()
        confidence = probs[0, pred_class].item()
    return pred_class, confidence, logits

def run_physio_inference(model, scaler, features):
    # Handle missing or nan features by replacing with mean or zero
    features_clean = np.nan_to_num(features, nan=0.0)
    features_scaled = scaler.transform([features_clean])
    pred_prob = model.predict_proba(features_scaled)[0, 1]  # probability of DR=1
    return pred_prob

def overlay_gradcam_on_image(orig_img, cam_mask, alpha=0.4):
    # Resize cam_mask to match original image size
    cam_mask_resized = cv2.resize(cam_mask, (orig_img.width, orig_img.height))
    
    # Apply heatmap color map
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_mask_resized), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    # Convert images to float32 for blending and normalize to 0-1
    heatmap = heatmap.astype(np.float32) / 255.0
    orig_img_np = np.array(orig_img).astype(np.float32) / 255.0
    
    # Blend heatmap and original image
    overlay = heatmap * alpha + orig_img_np * (1 - alpha)
    
    # Convert back to uint8
    overlay = np.clip(overlay * 255, 0, 255).astype(np.uint8)
    
    return Image.fromarray(overlay)

def is_physio_data_sufficient(features: dict) -> bool:
    if features.get("glucose_count", 0) < 1000:
        return False
    if features.get("total_steps", 0) < 20000:
        return False
    return True

def main():
    st.title("Diabetic Retinopathy Detection and Risk Assessment")

    st.sidebar.header("Upload inputs")

    uploaded_img = st.sidebar.file_uploader("Upload retinal fundus image (DICOM)", type=['dcm'])
    uploaded_cgm_file = st.sidebar.file_uploader("Upload CGM JSON file", type=["json"])
    uploaded_activity_file = st.sidebar.file_uploader("Upload Activity JSON file", type=["json"])
    hba1c = st.sidebar.number_input("Enter HbA1c (%)", min_value=3.0, max_value=15.0, step=0.1)

    if st.sidebar.button("Run Prediction"):

        if uploaded_img is None:
            st.error("Please upload a retinal image.")
            return
        
        cgm_json_str = None
        if uploaded_cgm_file is not None:
            try:
                cgm_json_str = uploaded_cgm_file.read().decode("utf-8")
                json.loads(cgm_json_str)
            except Exception as e:
                st.error(f"Invalid CGM JSON file: {e}")
                return

        # Read Activity JSON string if file uploaded
        activity_json_str = None
        if uploaded_activity_file is not None:
            try:
                activity_json_str = uploaded_activity_file.read().decode("utf-8")
                json.loads(activity_json_str)
            except Exception as e:
                st.error(f"Invalid Activity JSON file: {e}")
                return

        # Image preprocessing

        input_tensor, pil_img = load_and_prepare_image(uploaded_img)
        if input_tensor is None or pil_img is None:
            st.error("Image preprocessing failed.")
            return


        img_model = load_image_model(GDRIVE_DEST_PATH)
        physio_model = load_physio_model()
        scaler = load_scaler()

        # Image model inference
        pred_class, confidence, logits = run_image_inference(img_model, input_tensor)

        # GradCAM visualization
        target_layer = img_model.layer4[-1]  # ResNet50 last conv layer
        gradcam = GradCAM(img_model, target_layer)
        cam_mask = gradcam(input_tensor, class_idx=pred_class)
        gradcam.remove_hooks()
        overlay_img = overlay_gradcam_on_image(pil_img, cam_mask)

        # Display results
        st.subheader("Image-based Model Prediction")
        st.write(f"Predicted class: {'DR present' if pred_class == 1 else 'No DR'}")
        st.write(f"Confidence: {confidence:.3f}")
        st.image(overlay_img, caption="Grad-CAM Overlay", use_container_width=True)

        # Physiological features and model
        if cgm_json_str and activity_json_str and hba1c > 0.0:
            physio_feats = extract_physio_features(cgm_json_str, activity_json_str, hba1c)
            feat_dict = {
                "glucose_mean": physio_feats[0],
                "glucose_std": physio_feats[1],
                "glucose_min": physio_feats[2],
                "glucose_max": physio_feats[3],
                "glucose_count": physio_feats[4],
                "time_in_range_70_180": physio_feats[5],
                "percent_time_in_range_70_180": physio_feats[6],
                "total_steps": physio_feats[7],
                "mean_steps": physio_feats[8],
                "activity_count": physio_feats[9],
                "total_sedentary_minutes": physio_feats[10],
                "avg_daily_steps": physio_feats[11],
                "percent_sedentary_time": physio_feats[12],
                "hba1c": physio_feats[13],
            }

            if is_physio_data_sufficient(feat_dict):
                physio_pred_prob = run_physio_inference(physio_model, scaler, physio_feats)
                st.subheader("Physiological Model Risk Score")
                st.write(f"Predicted DR risk probability (from CGM, activity, HbA1c): {physio_pred_prob:.3f}")
            else:
                st.warning(
                    "Physiological data insufficient for reliable risk prediction "
                    "(too few CGM or activity samples). Skipping physiological model prediction."
                )
        else:
            st.info("Physiological model prediction skipped due to missing inputs.")


if __name__ == "__main__":
    main()