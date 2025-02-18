# process_image.py
import numpy as np
import tensorflow as tf
from PIL import Image

def preprocess_image(image):
    """Preprocess the image for model prediction."""
    try:
        image = image.resize((224, 224))  # Resize for MobileNetV2
        image = np.array(image) / 255.0   # Normalize pixel values
        
        # Ensure image has 3 channels (RGB)
        if len(image.shape) == 2:  # If grayscale
            image = np.stack((image,)*3, axis=-1)
        elif image.shape[-1] == 4:  # If RGBA
            image = image[..., :3]
            
        image = np.expand_dims(image, axis=0)  # Add batch dimension
        return image
    except Exception as e:
        raise Exception(f"Error preprocessing image: {e}")