import os
import numpy as np
import tensorflow as tf
from keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import cv2

# Define paths
model_path = 'model/mobilenetv2_finetuned.h5'
input_folder = 'input_images/'
edge_output_folder = 'results/processed_images/'
measurement_output_folder = 'results/crack_measurements/'

# Create output folders if they don't exist
os.makedirs(edge_output_folder, exist_ok=True)
os.makedirs(measurement_output_folder, exist_ok=True)

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

# Function to measure crack dimensions
def measure_crack_dimensions(edge_image_path, output_file_path):
    """
    Measure the dimensions of the crack from the edge-detected image.
    Saves the results to a text file.
    """
    # Load the edge-detected image
    image = cv2.imread(edge_image_path, cv2.IMREAD_GRAYSCALE)
    
    # Find contours
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    total_length = 0
    max_width = 0

    for contour in contours:
        # Calculate arc length (crack length)
        length = cv2.arcLength(contour, closed=False)
        total_length += length

        # Calculate bounding box (crack width)
        _, _, width, height = cv2.boundingRect(contour)
        max_width = max(max_width, max(width, height))

    # Save results
    with open(output_file_path, 'w') as f:
        f.write(f"Crack Length (pixels): {total_length:.2f}\n")
        f.write(f"Maximum Crack Width (pixels): {max_width:.2f}\n")
    
    print(f"Dimensions saved to {output_file_path}")

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
            edge_output_path = os.path.join(edge_output_folder, f"edges_{filename}")
            perform_edge_detection(file_path, edge_output_path)
            print(f"Edge detection completed for {filename}. Saved to {edge_output_path}")

            # Step 3: Crack Dimension Measurement
            measurement_output_path = os.path.join(measurement_output_folder, f"{os.path.splitext(filename)[0]}_dimensions.txt")
            measure_crack_dimensions(edge_output_path, measurement_output_path)
        else:
            print(f"No crack detected in {filename}. Skipping further analysis.")

print("Processing complete. Results saved to:", edge_output_folder, "and", measurement_output_folder)
