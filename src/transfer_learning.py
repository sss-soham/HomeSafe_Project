import os
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Ensure the "model" directory exists
if not os.path.exists("model"):
    os.makedirs("model")

# Load the MobileNetV2 model with pre-trained ImageNet weights
base_model = MobileNetV2(
    input_shape=(224, 224, 3),  # Image input size
    include_top=False,         # Exclude the fully connected layers
    weights="imagenet"         # Pre-trained on ImageNet
)

# Freeze the base model layers to retain pre-trained features
base_model.trainable = False

# Print a summary of the base model
base_model.summary()

# Define the complete model
model = Sequential([
    base_model,                  # Pre-trained MobileNet V2 base
    GlobalAveragePooling2D(),    # Pooling layer to reduce dimensions
    Dropout(0.5),                # Dropout for regularization
    Dense(128, activation="relu"),  # Fully connected layer
    Dense(2, activation="softmax")  # Output layer for two classes
])

# Print the complete model summary
model.summary()

# Compile the model
model.compile(
    optimizer=Adam(learning_rate=0.001),  # Adam optimizer with a learning rate of 0.001
    loss="categorical_crossentropy",      # Categorical cross-entropy for binary classification
    metrics=["accuracy"]                  # Evaluate model performance with accuracy
)

# Save model summary to a file with UTF-8 encoding
with open("model_summary.txt", "w", encoding="utf-8") as f:
    model.summary(print_fn=lambda x: f.write(x + "\n"))

print("Model summary saved to model_summary.txt")

# Save the complete model for training
model.save("model/mobilenetv2_final.h5")
print("Complete model saved to model/mobilenetv2_final.h5")

# Path to train and validation directories
train_dir = "selected_dataset/train"
validation_dir = "selected_dataset/validation"

# Data augmentation for the training set
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True
)

# No augmentation for the validation set
validation_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

# Data generators
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),  # Match input shape to MobileNetV2
    batch_size=32,
    class_mode="categorical"
)

validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(224, 224),  # Match input shape to MobileNetV2
    batch_size=32,
    class_mode="categorical"
)

print("Data generators created successfully.")
