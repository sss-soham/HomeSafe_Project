import pandas as pd
import os

def load_data(file_path):
    """
    Load dataset from the given file path and perform basic validation.
    Args:
        file_path (str): Path to the dataset file.
    Returns:
        pd.DataFrame: Loaded dataset as a Pandas DataFrame.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file at {file_path} does not exist.")
    
    try:
        data = pd.read_csv(file_path)
        print(f"Dataset loaded successfully with shape: {data.shape}")
        print(f"Columns: {list(data.columns)}")
        return data
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None

# Example usage (update 'your_dataset.csv' with the actual path later)
if __name__ == "__main__":
    file_path = "data/your_dataset.csv"  # Replace with actual file path
    dataset = load_data(file_path)
