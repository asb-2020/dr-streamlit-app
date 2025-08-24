import numpy as np
import cv2
from PIL import Image
import io
import json
import pandas as pd
from dateutil import parser as dateparser

import torch
from torchvision import transforms

# --- Image preprocessing ---

def apply_clahe(img: np.ndarray) -> np.ndarray:
    """Apply CLAHE to the green channel of the RGB image."""
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl,a,b))
    img_clahe = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
    return img_clahe

def crop_circle(img: np.ndarray) -> np.ndarray:
    """Crop circular fundus region (assumed black background)."""
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return img
    cnt = max(contours, key=cv2.contourArea)
    (x, y), radius = cv2.minEnclosingCircle(cnt)
    x, y, radius = int(x), int(y), int(radius)
    mask = np.zeros_like(gray)
    cv2.circle(mask, (x,y), radius, 255, -1)
    masked = cv2.bitwise_and(img, img, mask=mask)
    return masked

def preprocess_image(pil_img: Image.Image, image_size=224) -> torch.Tensor:
    """Full preprocessing pipeline from PIL image to tensor."""
    img_np = np.array(pil_img.convert("RGB"))
    img_clahe = apply_clahe(img_np)
    img_cropped = crop_circle(img_clahe)
    img_resized = cv2.resize(img_cropped, (image_size, image_size))
    
    # Normalize as per ImageNet stats (assuming pretrained ResNet50)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])
    ])
    tensor = transform(img_resized)
    return tensor.unsqueeze(0)  # Add batch dimension

# --- Physiological data preprocessing ---

def parse_iso_timestamp(ts: str):
    try:
        return dateparser.parse(ts)
    except Exception as e:
        raise ValueError(f"Failed to parse timestamp '{ts}': {e}")

def load_cgm_json(json_str: str) -> pd.DataFrame:
    raw = json.loads(json_str)
    entries = raw.get("body", {}).get("cgm", [])
    if not entries:
        return pd.DataFrame()
    data = []
    for entry in entries:
        try:
            ts = entry["effective_time_frame"]["time_interval"]["start_date_time"]
            val = entry["blood_glucose"]["value"]
            data.append({"time": ts, "glucose_mg_dL": val})
        except KeyError:
            continue
    df = pd.DataFrame(data)
    df["time"] = pd.to_datetime(df["time"], errors='coerce', utc=True)
    df = df.dropna(subset=["glucose_mg_dL"])
    df["glucose_mg_dL"] = pd.to_numeric(df["glucose_mg_dL"], errors='coerce')
    return df

def load_activity_json(json_str: str) -> pd.DataFrame:
    raw = json.loads(json_str)
    activities = raw.get('body', {}).get('activity', [])
    rows = []
    for act in activities:
        try:
            start_ts = act['effective_time_frame']['time_interval']['start_date_time']
            end_ts = act['effective_time_frame']['time_interval']['end_date_time']
            start_time = parse_iso_timestamp(start_ts)
            end_time = parse_iso_timestamp(end_ts)
            steps = act.get('base_movement_quantity', {}).get('value', 0)
            steps = int(steps) if isinstance(steps, (int, float)) or (isinstance(steps, str) and steps.isdigit()) else 0
            activity_name = act.get('activity_name', '')
            rows.append({
                'start_time': start_time,
                'end_time': end_time,
                'activity_name': activity_name,
                'steps': steps
            })
        except KeyError:
            continue
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values('start_time').reset_index(drop=True)
    return df

def compute_cgm_features(df_cgm: pd.DataFrame) -> dict:
    if df_cgm.empty:
        return {
            "glucose_mean": np.nan,
            "glucose_std": np.nan,
            "glucose_min": np.nan,
            "glucose_max": np.nan,
            "glucose_count": 0,
            "time_in_range_70_180": 0,
            "percent_time_in_range_70_180": 0.0,
        }
    glucose = df_cgm["glucose_mg_dL"]
    time_in_range = glucose.between(70, 180)
    time_in_range_count = time_in_range.sum()
    total_count = glucose.count()
    return {
        "glucose_mean": glucose.mean(),
        "glucose_std": glucose.std(),
        "glucose_min": glucose.min(),
        "glucose_max": glucose.max(),
        "glucose_count": total_count,
        "time_in_range_70_180": time_in_range_count,
        "percent_time_in_range_70_180": time_in_range_count / total_count if total_count > 0 else 0.0,
    }

def compute_activity_features(df_activity: pd.DataFrame) -> dict:
    if df_activity.empty:
        return {
            'total_steps': 0,
            'mean_steps': 0,
            'activity_count': 0,
            'total_sedentary_minutes': 0,
            'avg_daily_steps': 0,
            'percent_sedentary_time': 0.0
        }
    total_steps = df_activity['steps'].sum()
    mean_steps = df_activity['steps'].mean()
    count = len(df_activity)
    sedentary_df = df_activity[df_activity['activity_name'].str.lower() == 'sedentary']
    sedentary_duration_min = ((sedentary_df['end_time'] - sedentary_df['start_time']).dt.total_seconds().sum()) / 60
    total_time_min = ((df_activity['end_time'] - df_activity['start_time']).dt.total_seconds().sum()) / 60
    days_tracked = df_activity['start_time'].dt.date.nunique() if not df_activity.empty else 1
    avg_daily_steps = total_steps / days_tracked if days_tracked > 0 else 0
    percent_sedentary_time = sedentary_duration_min / total_time_min if total_time_min > 0 else 0
    return {
        'total_steps': total_steps,
        'mean_steps': mean_steps,
        'activity_count': count,
        'total_sedentary_minutes': sedentary_duration_min,
        'avg_daily_steps': avg_daily_steps,
        'percent_sedentary_time': percent_sedentary_time
    }

def extract_physio_features(cgm_json_str: str, activity_json_str: str, hba1c: float):
    """Parse physiological JSON strings and extract feature vector including HbA1c."""
    df_cgm = load_cgm_json(cgm_json_str)
    df_activity = load_activity_json(activity_json_str)
    cgm_feats = compute_cgm_features(df_cgm)
    activity_feats = compute_activity_features(df_activity)
    features = {
        **cgm_feats,
        **activity_feats,
        "hba1c": hba1c
    }
    # Convert dict to numpy array in consistent order
    feat_order = [
        "glucose_mean", "glucose_std", "glucose_min", "glucose_max", "glucose_count",
        "time_in_range_70_180", "percent_time_in_range_70_180",
        "total_steps", "mean_steps", "activity_count", "total_sedentary_minutes",
        "avg_daily_steps", "percent_sedentary_time",
        "hba1c"
    ]
    feat_vector = np.array([features.get(f, np.nan) for f in feat_order], dtype=np.float32)
    return feat_vector