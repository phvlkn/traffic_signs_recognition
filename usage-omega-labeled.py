import cv2
import tkinter as tk
from tkinter import filedialog
from ultralytics import YOLO
from PIL import Image, ImageTk
import threading


class SignLabelMapper:
    def __init__(self):
        self.label_map = {
            0: "Знак дополнительной информации",
            1: "Место остановки автобуса и (или) троллейбуса",
            2: "Главная дорога",
            3: "Уступите дорогу",
            4: "Остановка запрещена",
            5: "Парковка (парковочное место)",
            6: "Пешеходный переход",
            7: "Ограничение максимальной скорости (5, 20, 30, 40, 60, 80)",
            8: "Движение без остановки запрещено - Въезд запрещен"
        }

    def get_label(self, label_id):
        return self.label_map.get(label_id, f"Неизвестный знак ({label_id})")



    def __init__(self, root):
        self.root = root
        self.root.title("YOLOv8 Object Detection")

        # Загрузка модели
        self.model = YOLO("runs/detect/train11/weights/best.pt")

        # Переменные
        self.mapper = SignLabelMapper()
        self.cap = None
        self.running = False
        self.current_source = None
        self.video_file = ""

        # Создание GUI
        self.create_widgets()

    def create_widgets(self):
        # Фрейм для кнопок
        button_frame = tk.Frame(self.root)
        button_frame.pack(pady=10)

        # Кнопки выбора источника
        self.camera_btn = tk.Button(button_frame, text="Камера", command=self.start_camera)
        self.camera_btn.pack(side=tk.LEFT, padx=5)

        self.file_btn = tk.Button(button_frame, text="Файл", command=self.open_file)
        self.file_btn.pack(side=tk.LEFT, padx=5)

        self.stop_btn = tk.Button(button_frame, text="Стоп", command=self.stop, state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, padx=5)

        # Метка для отображения видео
        self.video_label_name = self.mapper.get_label( tk.Label(self.root)
        self.video_label.pack(pady=10)

        # Метка для информации
        self.info_label_name = self.mapper.get_label( tk.Label(self.root, text="Выберите источник видео")
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

        # Обработка кадра с YOLO
        results = self.model(frame)
        annotated_frame = results[0].plot()

        # Конвертация для Tkinter
        cv2image = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(cv2image)
        imgtk = ImageTk.PhotoImage(image=img)

        self.video_label.imgtk = imgtk
        self.video_label.configure(image=imgtk)

        # Рекурсивный вызов
        self.root.after(10, self.update_frame)

    def on_closing(self):
        self.stop()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = YOLOv8App(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()