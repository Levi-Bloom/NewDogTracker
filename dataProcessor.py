import os
import shutil

# Paths
dataset_dir = "dogBigBatch"  # Original dataset with folder names as labels
output_dir = "yoloData"  # YOLO-compatible dataset output folder

# YOLO-compatible folders
images_dir = os.path.join(output_dir, "images/all")
labels_dir = os.path.join(output_dir, "labels/all")
os.makedirs(images_dir, exist_ok=True)
os.makedirs(labels_dir, exist_ok=True)

# Prepare the dataset
classes = sorted(os.listdir(dataset_dir))  # Sorted list of class names (folder names)
class_to_id = {cls: idx for idx, cls in enumerate(classes)}  # Map class names to numeric IDs

print("Preparing dataset...")
for cls in classes:
    cls_path = os.path.join(dataset_dir, cls)
    if not os.path.isdir(cls_path):  # Skip non-directories
        continue

    cls_images = [img for img in os.listdir(cls_path) if img.endswith(('.jpg', '.png'))]
    for img_name in cls_images:
        img_path = os.path.join(cls_path, img_name)
        label_path = os.path.join(labels_dir, f"{os.path.splitext(img_name)[0]}.txt")

        # Generate YOLO label file with class ID and dummy bounding box
        with open(label_path, "w") as f:
            f.write(f"{class_to_id[cls]} 0.5 0.5 1.0 1.0\n")

        # Copy image to YOLO dataset folder
        shutil.copy(img_path, images_dir)

# Sanitize class names for YOLO
sanitized_classes = [
    cls.replace("-", "_").replace(" ", "_").lower() for cls in classes
]

# Create data.yaml file
data_yaml_path = os.path.join(output_dir, "data.yaml")
print("Creating data.yaml...")
with open(data_yaml_path, "w") as f:
    f.write(f"train: {os.path.abspath(images_dir)}\n")
    f.write(f"val: {os.path.abspath(images_dir)}\n")  # Use the same dataset for validation
    f.write(f"nc: {len(sanitized_classes)}\n")
    f.write(f"names: {sanitized_classes}\n")

print(f"Dataset preparation complete. YOLO dataset saved to: {output_dir}")