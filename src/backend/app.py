import streamlit as st
import numpy as np
from PIL import Image
from model_loader import model
from process_image import preprocess_image

# Streamlit UI
st.title("HomeSafe: AI-Powered Housing Inspection")
st.write("Upload an image to detect cracks.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    try:
        # Display uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Preprocess image
        processed_image = preprocess_image(image)

        # Predict
        prediction = model.predict(processed_image)
        confidence = prediction[0][0]

        # Show result
        if confidence > 0.5:
            st.error(f"Crack Detected! Confidence: {confidence:.2f}")
        else:
            st.success(f"No Crack Detected. Confidence: {1 - confidence:.2f}")
    
    except Exception as e:
        st.error(f"An error occurred: {e}")
