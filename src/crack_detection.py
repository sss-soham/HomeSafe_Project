import numpy as np
import tensorflow as tf
from keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Load the fine-tuned model
model_path = 'model/mobilenetv2_finetuned.h5'
model = load_model(model_path)

def preprocess_image(image_path, target_size=(224, 224)):
    """Preprocess the image for prediction."""
    image = load_img(image_path, target_size=target_size)
    image = img_to_array(image) / 255.0  # Normalize
    return np.expand_dims(image, axis=0)

def detect_crack(image_path):
    """
    Predict whether the image has a crack or not.
    Returns True if crack detected, False otherwise.
    """
    image = preprocess_image(image_path)
    prediction = model.predict(image, verbose=0)[0][0]
    return prediction > 0.5  # Crack if probability > 0.5