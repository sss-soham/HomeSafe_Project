import cv2
import os

def perform_edge_detection(input_image_path, output_image_path):
    """
    Perform edge detection on the input image and save the result.
    """
    image = cv2.imread(input_image_path, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, threshold1=50, threshold2=150)
    cv2.imwrite(output_image_path, edges)
