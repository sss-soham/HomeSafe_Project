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
    # Create model directory if it doesn't exist
    Path("model").mkdir(exist_ok=True)
    
    if not os.path.exists(MODEL_PATH):
        st.write("Downloading model...")
        file_id = st.secrets["google_drive"]["file_id"]
        url = f'https://drive.google.com/uc?id={file_id}'
        gdown.download(url, MODEL_PATH, quiet=False)
    
    try:
        model = keras.models.load_model(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model()