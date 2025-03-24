from ultralytics import YOLO
import cv2
# Load a model
model = YOLO("yolo11n.pt")  # load a pretrained model (recommended for training)

# Train the model with MPS
results = model.train(data="datasets/data.yaml", epochs=100, batch=32, imgsz=512, device="mps")
