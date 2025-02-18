import os
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import matplotlib.pyplot as plt

# Paths
model_path = "model/mobilenetv2_model.h5"
train_dir = "selected_dataset/train"
validation_dir = "selected_dataset/validation"
final_model_path = "model/mobilenetv2_final.h5"

# Ensure the plots directory exists
os.makedirs("plots/fine_tune", exist_ok=True)

# Load the pre-trained model
model = load_model(model_path)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),  # Lower learning rate for fine-tuning
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# Data augmentation for training set
train_datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input
)

# No augmentation for validation set
validation_datagen = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input
)

# Data generators
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode="categorical"
)

validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode="categorical"
)

# Callbacks
callbacks = [
    ModelCheckpoint(final_model_path, save_best_only=True, monitor="val_loss"),
    EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
]

# Fine-tune the model
history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=25,  # Adjust as necessary
    callbacks=callbacks
)

# Plot accuracy
plt.plot(history.history["accuracy"], label="Train Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.title("Fine-Tuning Accuracy")
plt.grid()
plt.savefig("plots/fine_tune/accuracy_plot.png")
plt.show()

# Plot loss
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.title("Fine-Tuning Loss")
plt.grid()
plt.savefig("plots/fine_tune/loss_plot.png")
plt.show()
