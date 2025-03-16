import sys
import os
from PyQt5.QtWidgets import (QApplication, QMainWindow, QLabel, QPushButton,
                             QVBoxLayout, QHBoxLayout, QFileDialog, QWidget,
                             QScrollArea, QListWidget, QMessageBox)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt, QThread, pyqtSignal
import cv2
import numpy as np
from PIL import Image
import pyttsx3
import threading

try:
    from ultralytics import YOLO
except ImportError:
    QMessageBox.critical(None, "Lỗi",
                         "Không thể import ultralytics. Hãy cài đặt thư viện bằng cách chạy: pip install ultralytics")
    sys.exit(1)


class YoloThread(QThread):
    detection_complete = pyqtSignal(object, np.ndarray)

    def __init__(self, model_path, image_path, conf_threshold):
        super().__init__()
        self.model_path = model_path
        self.image_path = image_path
        self.conf_threshold = conf_threshold

    def run(self):
        try:
            # Tải model
            if self.model_path == "default":
                model = YOLO("yolov8n.pt")
            else:
                model = YOLO(self.model_path)

            # Thực hiện dự đoán
            results = model(self.image_path, conf=self.conf_threshold)

            # Tạo ảnh kết quả
            for r in results:
                im_array = r.plot()  # Vẽ kết quả lên ảnh

            # Gửi tín hiệu khi hoàn thành
            self.detection_complete.emit(results, im_array)
        except Exception as e:
            print(f"Lỗi trong quá trình xử lý: {e}")


class ObjectDetectionApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Nhận diện đối tượng với YOLOv8")
        self.setGeometry(100, 100, 1000, 700)

        # Khởi tạo engine cho việc phát âm
        self.engine = pyttsx3.init()
        # Thiết lập ngôn ngữ tiếng Việt nếu có
        voices = self.engine.getProperty('voices')
        for voice in voices:
            if "vietnamese" in voice.id.lower():
                self.engine.setProperty('voice', voice.id)
                break

        # Biến lưu trữ đường dẫn ảnh
        self.image_path = None
        self.result_image = None
        self.results = None
        self.image_filename = None

        # Khởi tạo layout chính
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        main_layout = QHBoxLayout()
        left_layout = QVBoxLayout()
        right_layout = QVBoxLayout()

        # Layout bên trái - Hiển thị ảnh
        self.image_label = QLabel("Chưa có ảnh")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(500, 400)
        self.image_label.setStyleSheet("border: 2px solid gray;")

        scroll_area = QScrollArea()
        scroll_area.setWidget(self.image_label)
        scroll_area.setWidgetResizable(True)
        left_layout.addWidget(scroll_area)

        # Buttons
        btn_layout = QHBoxLayout()

        self.load_button = QPushButton("Tải ảnh")
        self.load_button.clicked.connect(self.load_image)
        btn_layout.addWidget(self.load_button)

        self.detect_button = QPushButton("Nhận diện")
        self.detect_button.clicked.connect(self.detect_objects)
        self.detect_button.setEnabled(False)
        btn_layout.addWidget(self.detect_button)

        self.save_button = QPushButton("Lưu kết quả")
        self.save_button.clicked.connect(self.save_results)
        self.save_button.setEnabled(False)
        btn_layout.addWidget(self.save_button)

        self.speak_button = QPushButton("Đọc kết quả")
        self.speak_button.clicked.connect(self.speak_results)
        self.speak_button.setEnabled(False)
        btn_layout.addWidget(self.speak_button)

        left_layout.addLayout(btn_layout)

        # Layout bên phải - Hiển thị kết quả
        right_layout.addWidget(QLabel("Các đối tượng được phát hiện:"))

        self.result_list = QListWidget()
        right_layout.addWidget(self.result_list)

        # Thêm layouts vào layout chính
        main_layout.addLayout(left_layout, 3)
        main_layout.addLayout(right_layout, 1)

        self.central_widget.setLayout(main_layout)

        # Thiết lập model
        self.model_path = "default"  # Sử dụng model mặc định
        self.conf_threshold = 0.25  # Ngưỡng tin cậy

    def load_image(self):
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getOpenFileName(self, "Chọn ảnh", "", "Image Files (*.png *.jpg *.jpeg *.bmp)")

        if file_path:
            self.image_path = file_path
            self.image_filename = os.path.basename(file_path)
            pixmap = QPixmap(file_path)

            # Hiển thị ảnh với kích thước vừa với label
            self.image_label.setPixmap(pixmap.scaled(
                self.image_label.width(), self.image_label.height(),
                Qt.KeepAspectRatio, Qt.SmoothTransformation
            ))

            self.detect_button.setEnabled(True)
            self.save_button.setEnabled(False)
            self.speak_button.setEnabled(False)
            self.result_list.clear()
            self.setWindowTitle(f"Nhận diện đối tượng - {self.image_filename}")

    def detect_objects(self):
        if not self.image_path:
            return

        # Disable buttons during detection
        self.detect_button.setEnabled(False)
        self.load_button.setEnabled(False)
        self.detect_button.setText("Đang xử lý...")

        # Create and start detection thread
        self.thread = YoloThread(self.model_path, self.image_path, self.conf_threshold)
        self.thread.detection_complete.connect(self.handle_detection_results)
        self.thread.start()

    def handle_detection_results(self, results, im_array):
        self.results = results
        self.result_image = im_array

        # Hiển thị ảnh kết quả
        h, w, c = im_array.shape
        q_image = QImage(im_array.data, w, h, w * c, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)

        self.image_label.setPixmap(pixmap.scaled(
            self.image_label.width(), self.image_label.height(),
            Qt.KeepAspectRatio, Qt.SmoothTransformation
        ))

        # Hiển thị thông tin về các đối tượng được nhận diện
        self.result_list.clear()

        object_counts = {}  # Đếm số lượng đối tượng theo loại
        for r in results:
            boxes = r.boxes
            names_dict = r.names

            for box in boxes:
                cls = int(box.cls[0])
                confidence = float(box.conf[0])
                class_name = names_dict[cls]

                # Cập nhật số lượng
                if class_name in object_counts:
                    object_counts[class_name] += 1
                else:
                    object_counts[class_name] = 1

        # Thêm thông tin tổng số đối tượng
        self.result_list.insertItem(0, f"ĐÃ PHÁT HIỆN {sum(object_counts.values())} ĐỐI TƯỢNG:")
        for class_name, count in object_counts.items():
            self.result_list.insertItem(1, f"- {class_name}: {count}")

        # Enable buttons after detection
        self.detect_button.setEnabled(True)
        self.load_button.setEnabled(True)
        self.save_button.setEnabled(True)
        self.speak_button.setEnabled(True)
        self.detect_button.setText("Nhận diện")

    def save_results(self):
        if self.result_image is None:
            return

        # Tạo tên file mặc định từ tên file gốc
        filename, ext = os.path.splitext(self.image_filename)
        default_save_name = f"{filename}_res{ext}"

        file_dialog = QFileDialog()
        save_path, _ = file_dialog.getSaveFileName(self, "Lưu kết quả", default_save_name, "JPEG (*.jpg);;PNG (*.png)")

        if save_path:
            if not (save_path.endswith(".jpg") or save_path.endswith(".png")):
                save_path += ".jpg"

            cv2.imwrite(save_path, cv2.cvtColor(self.result_image, cv2.COLOR_RGB2BGR))
            QMessageBox.information(self, "Thông báo", f"Đã lưu kết quả tại: {save_path}")

    def speak_results(self):
        if not self.results:
            return

        # Đếm số lượng đối tượng theo loại
        object_counts = {}
        for r in self.results:
            boxes = r.boxes
            names_dict = r.names

            for box in boxes:
                cls = int(box.cls[0])
                class_name = names_dict[cls]

                if class_name in object_counts:
                    object_counts[class_name] += 1
                else:
                    object_counts[class_name] = 1

        # Tạo văn bản để đọc bằng tiếng Việt
        speech_text = f"Đã phát hiện {sum(object_counts.values())} đối tượng. Bao gồm: "
        for class_name, count in object_counts.items():
            speech_text += f"{count} {class_name}, "

        # Đọc trong một thread riêng để không làm treo giao diện
        def speak_thread(text):
            self.engine.say(text)
            self.engine.runAndWait()

        threading.Thread(target=speak_thread, args=(speech_text,)).start()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ObjectDetectionApp()
    window.show()
    sys.exit(app.exec_())