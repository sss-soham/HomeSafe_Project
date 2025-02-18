import os
import tensorflow as tf
from keras.models import load_model
# from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

# Set paths
base_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(base_dir, '../model/mobilenetv2_final.h5')
validation_data_dir = os.path.join(base_dir, '../selected_dataset/validation')  # Updated path for validation data

# Parameters
batch_size = 32
img_height = 224
img_width = 224

print("Loading model...")
print(f"Loading model from: {model_path}")
model = load_model(model_path)
print("Model loaded successfully.")

# Data generator for validation data
validation_datagen = ImageDataGenerator(rescale=1.0 / 255)
validation_generator = validation_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False  # Important to keep the order of data for metrics
)

# Evaluate the model
print("Evaluating model...")
results = model.evaluate(validation_generator)
print(f"Evaluation Results - Loss: {results[0]:.4f}, Accuracy: {results[1]:.4f}")

# Get true labels and predictions
y_true = validation_generator.classes  # True labels as integers
y_pred_prob = model.predict(validation_generator)  # Predicted probabilities
y_pred = np.argmax(y_pred_prob, axis=1)  # Convert to class indices

# Generate classification report
print("Generating classification report...")
class_labels = list(validation_generator.class_indices.keys())  # Class names
report = classification_report(y_true, y_pred, target_names=class_labels)
print(report)

# Generate confusion matrix
print("Generating confusion matrix...")
conf_matrix = confusion_matrix(y_true, y_pred)
print("Confusion Matrix:")
print(conf_matrix)

# Save classification report to a file
report_path = os.path.join(base_dir, '../reports/classification_report1.txt')
with open(report_path, 'w') as f:
    f.write("Classification Report\n")
    f.write("=====================\n")
    f.write(report)
print(f"Classification report saved to {report_path}")

# Save confusion matrix to a file
conf_matrix_path = os.path.join(base_dir, '../reports/confusion_matrix1.txt')
with open(conf_matrix_path, 'w') as f:
    f.write("Confusion Matrix\n")
    f.write("=================\n")
    np.savetxt(f, conf_matrix, fmt='%d')
print(f"Confusion matrix saved to {conf_matrix_path}")
