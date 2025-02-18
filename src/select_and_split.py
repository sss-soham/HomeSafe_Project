import os
import shutil
import random

# Paths
dataset_path = "dataset"  # Path to the original dataset
output_path = "selected_dataset"  # Output folder for selected and split dataset
classes = ["Crack", "Non-Crack"]  # Folder names for each class
target_count = 5000  # Number of images to select per class

# Splits
train_split = 0.8
validation_split = 0.1
test_split = 0.1

# Create output directories for train, validation, test
if not os.path.exists(output_path):
    os.makedirs(output_path)

for split in ["train", "validation", "test"]:
    for cls in classes:
        os.makedirs(os.path.join(output_path, split, cls), exist_ok=True)

# Randomly select and split images
for cls in classes:
    class_path = os.path.join(dataset_path, cls)
    images = os.listdir(class_path)

    # Randomly shuffle and select target_count images
    random.shuffle(images)
    selected_images = images[:target_count]

    # Split selected images
    train_count = int(target_count * train_split)
    validation_count = int(target_count * validation_split)
    test_count = target_count - train_count - validation_count

    for i, image_name in enumerate(selected_images):
        src_path = os.path.join(class_path, image_name)

        # Determine split folder
        if i < train_count:
            split = "train"
        elif i < train_count + validation_count:
            split = "validation"
        else:
            split = "test"

        # Determine class folder name 
        dest_path = os.path.join(output_path, split, cls, image_name)

        # Copy image to the target folder
        shutil.copy(src_path, dest_path)

print("Random image selection and splitting complete!")