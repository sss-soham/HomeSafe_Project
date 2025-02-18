import cv2
import os
import sys

def process_image(image_path, output_dir):
    """
    Processes a single image for edge detection and saves the result.
    """
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Skipping: Unable to read image at {image_path}")
        return
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Canny edge detection
    edges = cv2.Canny(gray, threshold1=100, threshold2=200)
    
    # Save the processed image
    filename = os.path.basename(image_path)
    output_path = os.path.join(output_dir, f"edges_{filename}")
    cv2.imwrite(output_path, edges)
    print(f"Processed and saved: {output_path}")

def process_images(input_folder, output_dir):
    """
    Processes all images in the given folder (handles single and multiple images).
    """
    if not os.path.exists(input_folder):
        print(f"Error: The provided folder path '{input_folder}' does not exist.")
        sys.exit(1)
    
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get list of all image files in the folder
    image_files = [
        os.path.join(input_folder, f)
        for f in os.listdir(input_folder)
        if os.path.isfile(os.path.join(input_folder, f))
    ]
    
    if not image_files:
        print(f"No images found in the folder: {input_folder}")
        sys.exit(1)
    
    # Process each image
    for image_file in image_files:
        process_image(image_file, output_dir)

if __name__ == "__main__":
    # Define the input folder path
    input_folder = 'sample_images/batch_input'  # Update this path as needed

    # Define the output directory for processed images
    output_dir = 'processed_images'
    
    # Process the folder
    print(f"Processing images in folder: {input_folder}")
    process_images(input_folder, output_dir)
    print(f"All images processed. Results saved in: {output_dir}")
