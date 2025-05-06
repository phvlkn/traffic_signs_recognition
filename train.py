from ultralytics import YOLO
model = YOLO("runs/detect/train4/weights/best.pt")  
results = model.train(data="datasets/data.yaml", epochs=100, batch=22, imgsz=512, device="mps")
