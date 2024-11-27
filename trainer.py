from ultralytics import YOLO

# Paths
data_yaml_path = "yoloData/data.yaml"  # Path to the prepared data.yaml
model_save_path = "models"  # Path to save the trained model

# Train YOLO Model
model = YOLO('yolov8n.pt')  # Load YOLO model (use a lightweight version for initial testing)

model.train(
    data=data_yaml_path,  # Path to the prepared data.yaml file
    epochs=50,  # Adjust based on your dataset size
    batch=16,
    imgsz=640,
    device="cpu"  # Change to "0" if you have a GPU
)

# Export the trained model
model.export(format='torchscript', save_dir=model_save_path)
print("Model training complete and saved.")