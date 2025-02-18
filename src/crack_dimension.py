import cv2
import os
import numpy as np

# Input and output directories
edge_input_folder = 'results/processed_images/'
measurement_output_folder = 'results/crack_measurements/'

# Create output directory if it doesn't exist
os.makedirs(measurement_output_folder, exist_ok=True)

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

# Process each edge-detected image
for filename in os.listdir(edge_input_folder):
    edge_image_path = os.path.join(edge_input_folder, filename)
    if os.path.isfile(edge_image_path):
        print(f"Measuring dimensions for {filename}...")
        output_file_path = os.path.join(measurement_output_folder, f"{os.path.splitext(filename)[0]}_dimensions.txt")
        measure_crack_dimensions(edge_image_path, output_file_path)
