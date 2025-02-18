import numpy as np
import tensorflow as tf
from keras.models import load_model
from keras.metrics import BinaryAccuracy, Precision, Recall
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import datetime
import sys

# Create results directory if it doesn't exist
os.makedirs('results', exist_ok=True)

# Custom progress bar class
class ProgressBar:
    def __init__(self, total_batches):
        self.total_batches = total_batches
        self.current_batch = 0
        
    def update(self, batch, logs=None):
        self.current_batch = batch + 1
        bar_length = 30
        filled_length = int(round(bar_length * self.current_batch / float(self.total_batches)))
        percents = round(100.0 * self.current_batch / float(self.total_batches), 1)
        bar = '=' * filled_length + '-' * (bar_length - filled_length)
        
        # Create the progress bar with correct batch numbering
        sys.stdout.write(f'\r{self.current_batch}/{self.total_batches} [{bar}] {percents}% - 1s/step')
        sys.stdout.flush()

# Custom Keras callback
class CustomCallback(tf.keras.callbacks.Callback):
    def __init__(self, total_batches):
        super().__init__()
        self.progress_bar = ProgressBar(total_batches)
        
    def on_predict_batch_end(self, batch, logs=None):
        self.progress_bar.update(batch)
        
    def on_predict_end(self, logs=None):
        sys.stdout.write('\n')

# Define paths
test_data_dir = 'selected_dataset/test'
model_paths = {
    "Model 1": 'model/mobilenetv2_final.h5',
    "Model 2": 'model/mobilenetv2_finetuned.h5'
}

# Prepare test data generator
test_datagen = ImageDataGenerator(rescale=1.0 / 255)
test_generator = test_datagen.flow_from_directory(
    test_data_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    shuffle=False
)

# Calculate steps for evaluation
steps = test_generator.samples // test_generator.batch_size + \
        (1 if test_generator.samples % test_generator.batch_size > 0 else 0)

def evaluate_model(model, test_generator, steps):
    y_true = []
    y_pred = []
    
    # Reset generator at the start of evaluation
    test_generator.reset()
    
    # Create custom callback
    custom_callback = CustomCallback(steps)
    
    # Process all batches at once
    predictions = model.predict(
        test_generator,
        steps=steps,
        callbacks=[custom_callback],
        verbose=0
    )
    
    # Get all true labels
    y_true = test_generator.labels
    
    # Handle different prediction shapes
    if predictions.ndim > 1 and predictions.shape[1] > 1:
        predictions = predictions[:, 1]
    else:
        predictions = predictions.flatten()
    
    # Ensure predictions are binary
    y_pred_binary = (predictions > 0.5).astype(int)
    
    # Calculate metrics
    accuracy = np.mean(y_true == y_pred_binary) * 100
    
    # Calculate precision
    true_positives = np.sum((y_true == 1) & (y_pred_binary == 1))
    predicted_positives = np.sum(y_pred_binary == 1)
    precision = (true_positives / predicted_positives if predicted_positives > 0 else 0) * 100
    
    # Calculate recall
    actual_positives = np.sum(y_true == 1)
    recall = (true_positives / actual_positives if actual_positives > 0 else 0) * 100
    
    # Calculate F1 score
    f1_score = 2 * (precision * recall) / (precision + recall + 1e-7)
    
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score
    }

# Evaluate both models and save results
results = {}
for model_name, model_path in model_paths.items():
    print(f"\nLoading {model_name} from {model_path}...")
    model = load_model(model_path)
    
    print(f"Evaluating {model_name}...")
    metrics = evaluate_model(model, test_generator, steps)
    results[model_name] = metrics

# Save results to file
result_file_path = os.path.join('results', 'comparison_models.txt')
with open(result_file_path, 'w') as f:
    f.write(f"Model Comparison Results - Generated on {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write("=" * 80 + "\n\n")
    
    f.write("Model Paths:\n")
    for model_name, model_path in model_paths.items():
        f.write(f"{model_name}: {model_path}\n")
    f.write("\n" + "=" * 80 + "\n\n")
    
    f.write("Performance Metrics:\n\n")
    for model_name, metrics in results.items():
        print(f"\n{model_name}:")
        f.write(f"\n{model_name}:\n")
        for key, value in metrics.items():
            metric_line = f"  {key.capitalize()}: {value:.2f}%"
            print(metric_line)
            f.write(metric_line + "\n")

print(f"\nResults have been saved to: {result_file_path}")