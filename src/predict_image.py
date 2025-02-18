import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import sys

# Define paths
model_path = "./model/mobilenetv2_final.h5"  # Path to the trained model
output_folder = "./predictions/"  # Folder to save annotated images

# Load the trained model
print("Loading model...")
try:
    model = load_model(model_path)
    print("Model loaded successfully.")
except FileNotFoundError:
    print(f"Error: Model file not found at {model_path}. Please check the path.")
    exit()

# Function to preprocess input images
def preprocess_image(img_path):
    try:
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        img_array = img_array / 255.0  # Normalize pixel values
        return img_array
    except Exception as e:
        print(f"Error while preprocessing image {img_path}: {e}")
        return None

# Main function to handle a single image upload
def process_uploaded_image(uploaded_file_path):
    os.makedirs(output_folder, exist_ok=True)  # Ensure output folder exists
    
    try:
        print(f"Processing uploaded image: {uploaded_file_path}")
        img_data = preprocess_image(uploaded_file_path)
        if img_data is None:
            return
        pred = model.predict(img_data)
        pred_class = np.argmax(pred, axis=1)[0]
        confidence = np.max(pred)

        # Annotate the image with predictions
        img_original = image.load_img(uploaded_file_path)
        plt.imshow(img_original)
        plt.axis('off')
        plt.title(f"Prediction: Class {pred_class}, Confidence: {confidence:.2f}")
        
        # Save annotated image
        output_file_path = os.path.join(output_folder, "annotated_uploaded_image.jpg")
        plt.savefig(output_file_path)
        print(f"Annotated image saved to {output_file_path}")
        plt.close()
    except Exception as e:
        print(f"Error processing the uploaded image: {e}")

# Run the script with a provided file path
if len(sys.argv) != 2:
    print("Usage: python predict_image.py <path_to_uploaded_image>")
    exit()

uploaded_file_path = sys.argv[1]
if not os.path.exists(uploaded_file_path):
    print(f"Error: File not found at {uploaded_file_path}")
    exit()

process_uploaded_image(uploaded_file_path)