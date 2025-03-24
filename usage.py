import cv2
from ultralytics import YOLO


model = YOLO("runs/detect/train4/weights/best.pt")


cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model.predict(frame, imgsz=512, conf=0.5)

    annotated_frame = results[0].plot()

    cv2.imshow("YOLO Real-Time Detection", annotated_frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()