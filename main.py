import cv2
import tkinter as tk
from tkinter import filedialog
from ultralytics import YOLO
from PIL import Image, ImageTk



CLASS_NAMES = {
    0: "Info sign",
    1: "Bus stop",
    2: "Main road",
    3: "Yield",
    4: "No stop",
    5: "Parking",
    6: "Crosswalk",
    7: "Speed limit",
    8: "No entry"
}


class YOLOv8App:
    def __init__(self, root):
        self.root = root
        self.root.title("YOLOv8 Распознавание знаков")

        
        self.model = YOLO("best.pt")

        
        self.cap = None
        self.running = False
        self.current_source = None
        self.video_file = ""

        
        self.create_widgets()

    def create_widgets(self):
        
        button_frame = tk.Frame(self.root)
        button_frame.pack(pady=10)

        
        self.camera_btn = tk.Button(button_frame, text="Камера", command=self.start_camera)
        self.camera_btn.pack(side=tk.LEFT, padx=5)

        self.file_btn = tk.Button(button_frame, text="Файл", command=self.open_file)
        self.file_btn.pack(side=tk.LEFT, padx=5)

        self.stop_btn = tk.Button(button_frame, text="Стоп", command=self.stop, state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, padx=5)

        
        self.video_label = tk.Label(self.root)
        self.video_label.pack(pady=10)

        
        self.info_label = tk.Label(self.root, text="Выберите источник видео")
        self.info_label.pack(pady=5)

    def start_camera(self):
        self.stop()
        self.current_source = "camera"
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self.info_label.config(text="Ошибка: камера не найдена")
            return

        self.info_label.config(text="Используется камера")
        self.running = True
        self.stop_btn.config(state=tk.NORMAL)
        self.update_frame()

    def open_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4 *.avi *.mov")])
        if not file_path:
            return

        self.stop()
        self.video_file = file_path
        self.current_source = "file"
        self.cap = cv2.VideoCapture(file_path)

        if not self.cap.isOpened():
            self.info_label.config(text="Ошибка: не удалось открыть файл")
            return

        self.info_label.config(text=f"Файл: {file_path.split('/')[-1]}")
        self.running = True
        self.stop_btn.config(state=tk.NORMAL)
        self.update_frame()

    def stop(self):
        self.running = False
        if self.cap is not None:
            self.cap.release()
        self.stop_btn.config(state=tk.DISABLED)

    def update_frame(self):
        if not self.running:
            return

        ret, frame = self.cap.read()
        if not ret:
            if self.current_source == "file":
                self.info_label.config(text="Видео закончилось")
                self.stop()
            return

        
        results = self.model(frame)
        boxes = results[0].boxes
        annotated_frame = frame.copy()

        if boxes is not None:
            for box in boxes:
                cls_id = int(box.cls[0].item())
                label = CLASS_NAMES.get(cls_id, f"Класс {cls_id}")

                x1, y1, x2, y2 = map(int, box.xyxy[0])

                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(annotated_frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        
        cv2image = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(cv2image)
        imgtk = ImageTk.PhotoImage(image=img)

        self.video_label.imgtk = imgtk
        self.video_label.configure(image=imgtk)

        
        self.root.after(10, self.update_frame)

    def on_closing(self):
        self.stop()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = YOLOv8App(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()