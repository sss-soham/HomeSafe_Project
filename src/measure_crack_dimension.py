import os
import csv
import numpy as np
import tensorflow as tf
from keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import cv2

# Define paths
model_path = 'model/mobilenetv2_finetuned.h5'
input_folder = 'input_images'
output_folder = 'results/processed_images/'
csv_output_path = 'results/crack_metrics.csv'

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Load the fine-tuned model
print("Loading model...")
model = load_model(model_path)
print("Model loaded successfully.")

# Function to preprocess a single image
def preprocess_image(image_path, target_size=(224, 224)):
    image = load_img(image_path, target_size=target_size)
    image = img_to_array(image)
    image = image / 255.0  # Rescale pixel values
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Function to perform crack detection
def detect_crack(image_path):
    image = preprocess_image(image_path)
    prediction = model.predict(image, verbose=0)[0][0]  # Get the probability of crack
    return prediction > 0.5  # Return True if crack detected, else False

# Function to perform edge detection
def perform_edge_detection(image_path, output_path):
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, threshold1=50, threshold2=150)
    cv2.imwrite(output_path, edges)
    return edges

# Function to measure crack dimensions
def measure_crack_dimensions(edges):
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    max_width = 0
    max_length = 0
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        max_width = max(max_width, w)
        max_length = max(max_length, h)
    return max_width, max_length

# Prepare CSV for saving crack metrics
with open(csv_output_path, mode='w', newline='') as csvfile:
    fieldnames = ['Image', 'Max Width (pixels)', 'Max Length (pixels)']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

    # Process each image in the folder
    print("Processing images...")
    for filename in os.listdir(input_folder):
        file_path = os.path.join(input_folder, filename)
        if os.path.isfile(file_path):
            print(f"Processing {filename}...")
            
            # Step 1: Crack Detection
            has_crack = detect_crack(file_path)
            if has_crack:
                print(f"Crack detected in {filename}. Proceeding with further analysis...")
                
                # Step 2: Edge Detection
                output_path = os.path.join(output_folder, f"edges_{filename}")
                edges = perform_edge_detection(file_path, output_path)
                print(f"Edge detection completed for {filename}. Saved to {output_path}")
                
                # Step 3: Crack Dimension Measurement
                max_width, max_length = measure_crack_dimensions(edges)
                print(f"Crack dimensions for {filename}: Max Width={max_width}, Max Length={max_length}")
                
                # Save metrics to CSV
                writer.writerow({
                    'Image': filename,
                    'Max Width (pixels)': max_width,
                    'Max Length (pixels)': max_length
                })
            else:
                print(f"No crack detected in {filename}. Skipping further analysis.")

print("Processing complete. Results saved to:", output_folder)
print("Crack metrics saved to:", csv_output_path)
