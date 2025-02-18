import os

# Define thresholds for severity levels
SEVERITY_THRESHOLDS = {
    "Minor": {"length": 50, "width": 2},
    "Moderate": {"length": 150, "width": 5},
}

def analyze_severity(measurement_file_path, output_file_path):
    """
    Analyze the severity of cracks based on their dimensions.
    """
    # Read measurement results
    with open(measurement_file_path, 'r') as f:
        lines = f.readlines()
    
    length = float(lines[0].split(":")[1].strip())
    width = float(lines[1].split(":")[1].strip())

    # Determine severity
    if length < SEVERITY_THRESHOLDS["Minor"]["length"] and width < SEVERITY_THRESHOLDS["Minor"]["width"]:
        severity = "Minor"
    elif length < SEVERITY_THRESHOLDS["Moderate"]["length"] and width < SEVERITY_THRESHOLDS["Moderate"]["width"]:
        severity = "Moderate"
    else:
        severity = "Severe"

    # Save the severity analysis results
    with open(output_file_path, 'w') as f:
        f.write(f"Crack Length: {length:.2f} pixels\n")
        f.write(f"Crack Width: {width:.2f} pixels\n")
        f.write(f"Severity: {severity}\n")

    print(f"Severity analysis saved to {output_file_path}")

if __name__ == "__main__":
    # Example usage
    measurement_folder = "results/crack_measurements/"
    severity_output_folder = "results/severity_analysis/"

    # Create output folder if it doesn't exist
    os.makedirs(severity_output_folder, exist_ok=True)

    # Process all measurement files
    for filename in os.listdir(measurement_folder):
        if filename.endswith("_dimensions.txt"):
            measurement_file = os.path.join(measurement_folder, filename)
            output_file = os.path.join(severity_output_folder, f"severity_{filename}")
            analyze_severity(measurement_file, output_file)
