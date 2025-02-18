import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.optimizers import Adam

# Base directory
base_dir = os.path.dirname(os.path.abspath(__file__))

# Paths
dataset_dir = os.path.join(base_dir, '../selected_dataset/train')
validation_dir = os.path.join(base_dir, '../selected_dataset/validation')
model_save_path = os.path.join(base_dir, '../model/mobilenetv2_finetuned.h5')

# Hyperparameters
learning_rate = 0.001
batch_size = 32
epochs = 10

# Data preparation
train_datagen = ImageDataGenerator(rescale=1.0/255)
validation_datagen = ImageDataGenerator(rescale=1.0/255)

train_generator = train_datagen.flow_from_directory(
    dataset_dir,
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode='sparse'  # Changed to 'sparse' for sparse categorical crossentropy
)

validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode='sparse'  # Changed to 'sparse' for sparse categorical crossentropy
)

# Model creation
base_model = MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
base_model.trainable = False  # Freeze the base model

model = Sequential([
    base_model,
    Flatten(),
    Dense(128, activation='relu'),
    Dense(2, activation='softmax')  # Output layer for binary classification
])

# Compile model
model.compile(
    optimizer=Adam(learning_rate=learning_rate),
    loss='sparse_categorical_crossentropy',  # Updated to sparse categorical crossentropy
    metrics=['accuracy']
)

# Train model
print("Starting training...")
history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=epochs
)

# Save the model
model.save(model_save_path)
print(f"Model saved to {model_save_path}")
