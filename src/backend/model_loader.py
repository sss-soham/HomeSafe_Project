import gdown
import os
import streamlit as st
from tensorflow import keras
from pathlib import Path

# Use secret for file ID
MODEL_PATH = "model/mobilenetv2_finetuned.h5"

@st.cache_resource
def load_model():
    """Download and load model dynamically."""
    Path("model").mkdir(exist_ok=True)  # Ensure model directory exists

    if not os.path.exists(MODEL_PATH):
        st.write("Downloading model from Google Drive...")
        try:
            file_id = st.secrets["google_drive"]["file_id"]
            url = f"https://drive.google.com/uc?id={file_id}"
            gdown.download(url, MODEL_PATH, quiet=False)
        except Exception as e:
            st.error(f"Model download failed: {e}")
            return None  # Exit function if download fails

    try:
        model = keras.models.load_model(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        os.remove(MODEL_PATH)  # Remove the corrupted file
        return None  # Prevent app from breaking

model = load_model()
