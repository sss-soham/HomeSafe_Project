import os
import pandas as pd
import tensorflow as tf
from keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.utils import load_img, img_to_array
import numpy as np

# Define base directory
base_dir = os.path.dirname(os.path.abspath(__file__))

# Define paths
model_path = os.path.join(base_dir, '../model/mobilenetv2_final.h5')
input_folder = os.path.join(base_dir, '../sample_images/batch_input')
output_csv = os.path.join(base_dir, '../reports/batch_predictions.csv')

# Load the trained model
print("Loading model...")
try:
    model = load_model(model_path)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# Prepare input folder
if not os.path.exists(input_folder):
    print(f"Input folder not found: {input_folder}")
    exit()

# Prepare batch images
print("Processing images...")
image_size = (224, 224)
batch_data = []
image_names = []

for file_name in os.listdir(input_folder):
    if file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
        file_path = os.path.join(input_folder, file_name)
        try:
            img = load_img(file_path, target_size=image_size)
            img_array = img_to_array(img) / 255.0  # Normalize pixel values
            batch_data.append(img_array)
            image_names.append(file_name)
        except Exception as e:
            print(f"Error processing image {file_name}: {e}")

if not batch_data:
    print("No valid images found for prediction.")
    exit()

# Convert list to numpy array
batch_data = np.array(batch_data)

# Perform batch prediction
print("Predicting...")
predictions = model.predict(batch_data, verbose=1)

# Convert predictions to class labels
predicted_labels = np.argmax(predictions, axis=1)
class_indices = {0: 'No Crack', 1: 'Crack'}  # Adjust based on your dataset

# Prepare results for CSV
results = []
for i, label in enumerate(predicted_labels):
    results.append({
        "Image Name": image_names[i],
        "Prediction": class_indices[label]
    })

# Save results to CSV
print("Saving predictions...")
os.makedirs(os.path.dirname(output_csv), exist_ok=True)
df = pd.DataFrame(results)
df.to_csv(output_csv, index=False)
print(f"Predictions saved to {output_csv}")
