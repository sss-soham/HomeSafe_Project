import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

# Paths to the dataset
dataset_path = "selected_dataset"
train_dir = os.path.join(dataset_path, "train")
validation_dir = os.path.join(dataset_path, "validation")
test_dir = os.path.join(dataset_path, "test")

# Image size and batch size
IMG_SIZE = (224, 224)  # Resized image dimensions for MobileNet
BATCH_SIZE = 32        # Number of images processed in one batch

# Data Augmentation for Training Set
train_datagen = ImageDataGenerator(
    rescale=1./255,          # Normalize pixel values to [0, 1]
    rotation_range=20,       # Randomly rotate images by 20 degrees
    width_shift_range=0.2,   # Randomly shift images horizontally by 20%
    height_shift_range=0.2,  # Randomly shift images vertically by 20%
    shear_range=0.2,         # Shear transformation
    zoom_range=0.2,          # Randomly zoom images
    horizontal_flip=True,    # Randomly flip images horizontally
    fill_mode='nearest'      # Fill empty pixels after transformations
)

# No augmentation for Validation and Test Sets, just normalization
val_test_datagen = ImageDataGenerator(rescale=1./255)

# Load training data with augmentation
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=IMG_SIZE,    # Resize images to match MobileNet input
    batch_size=BATCH_SIZE,   # Number of images per batch
    class_mode='binary'      # Binary classification: Crack or Non-Crack
)

# Load validation data
validation_generator = val_test_datagen.flow_from_directory(
    validation_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

# Load test data
test_generator = val_test_datagen.flow_from_directory(
    test_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

# Get a batch of training images
x_batch, y_batch = next(train_generator)

# Plot the images
plt.figure(figsize=(10, 10))
for i in range(9):  # Display 9 images
    plt.subplot(3, 3, i + 1)
    plt.imshow(x_batch[i])
    plt.axis("off")
plt.show()