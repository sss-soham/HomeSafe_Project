import os
import tensorflow as tf
import numpy as np
from keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix

# Define base directory
base_dir = os.path.dirname(os.path.abspath(__file__))

# Define model path
model_path = os.path.join(base_dir, '../model/mobilenetv2_final.h5')

# Load the trained model
print("Loading model...")
try:
    model = load_model(model_path)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# Prepare test dataset
print("Preparing test dataset...")
test_data_dir = os.path.join(base_dir, '../selected_dataset/test')
if not os.path.exists(test_data_dir):
    print(f"Test directory not found: {test_data_dir}")
    exit()

test_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

test_generator = test_datagen.flow_from_directory(
    test_data_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)

# Evaluate the model on the test dataset
print("Evaluating model...")
y_pred_prob = model.predict(test_generator, verbose=1)  # Probabilities
y_pred = np.argmax(y_pred_prob, axis=1)  # Convert to class indices

# True labels
y_true = test_generator.classes

# Generating classification report
print("Generating classification report...")
target_names = list(test_generator.class_indices.keys())  # Get class labels
try:
    report = classification_report(
        y_true,
        y_pred,
        target_names=target_names
    )
    print(report)
except ValueError as e:
    print(f"Error generating classification report: {e}")
    print("Ensure that the formats of y_true and y_pred match.")

# Optional: Generate confusion matrix
print("Generating confusion matrix...")
conf_matrix = confusion_matrix(y_true, y_pred)
print(f"Confusion Matrix:\n{conf_matrix}")
