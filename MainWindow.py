import os
import sys
import cv2
import time
import inspect
import numpy as np
import Dataset
import FaceDetector
import Recognizer
import Config
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QGroupBox,
    QPushButton,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QTableView,
    QProgressBar,
    QStatusBar,
    QHeaderView,
    QComboBox,
    QMessageBox,
    QCheckBox,
    QSlider,
)
from PyQt5.QtGui import QStandardItemModel, QStandardItem, QImage, QPixmap, QCloseEvent
from PyQt5.QtCore import Qt, QTimer

script_directory = os.path.dirname(
    os.path.abspath(inspect.getfile(inspect.currentframe()))
)


class CameraWindows(QMainWindow):
    def __init__(self, parent=None):
        super(CameraWindows, self).__init__(parent)
        self.parent = parent
        self.setWindowTitle("Realtime Face Recognition")
        self.face_detector = FaceDetector.Detector()
        self.face_recognition_enabled = False
        self.face_remove_background_enabled = False

        self.available_cameras = []
        index = 0
        while True:
            cap = cv2.VideoCapture(index)
            if not cap.isOpened():
                break
            else:
                self.available_cameras.append(index)
                cap.release()
            index += 1

        if len(self.available_cameras) == 0:
            QMessageBox.critical(self, "Error", "No camera available!")
            super().close()
            return None

        self.camera_index = self.available_cameras[0]
        self.camera = cv2.VideoCapture(self.camera_index)
        frame_width = int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT))

        view_layout = QHBoxLayout()
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(frame_width, frame_height)
        view_layout.addWidget(self.image_label)

        face_layout = QVBoxLayout()
        self.face_label = QLabel()
        self.face_label.setAlignment(Qt.AlignCenter)
        self.face_label.setMinimumSize(Config.RECOGNITION_SIZE, Config.RECOGNITION_SIZE)
        self.predict_label = QLabel()
        self.predict_label.setAlignment(Qt.AlignCenter)
        self.predict_label.setMinimumSize(
            Config.RECOGNITION_SIZE, Config.RECOGNITION_SIZE
        )
        face_layout.addWidget(self.face_label)
        face_layout.addWidget(self.predict_label)
        view_layout.addLayout(face_layout)

        btn_layout = QHBoxLayout()

        self.label_select_camera = QLabel("Select Camera:")
        self.label_select_camera.setFixedWidth(128)
        self.btn_select_camera = QComboBox()
        for i in self.available_cameras:
            self.btn_select_camera.addItem(str(i))
        self.btn_select_camera.currentIndexChanged.connect(self.on_camera_changed)
        self.btn_select_camera.setCurrentIndex(0)

        self.checkbox_remove_background = QCheckBox("Remove Background")
        self.checkbox_remove_background.setChecked(False)
        self.checkbox_remove_background.stateChanged.connect(
            self.on_remove_background_changed
        )

        self.btn_screenshot = QPushButton("Screenshot")
        self.btn_screenshot.clicked.connect(self.on_screenshot)

        self.btn_recognize_face = QPushButton("Recognize Face")
        self.btn_recognize_face.clicked.connect(self.toggle_face_recognition)

        btn_layout.addWidget(self.label_select_camera)
        btn_layout.addWidget(self.btn_select_camera)
        btn_layout.addWidget(self.checkbox_remove_background)
        btn_layout.addWidget(self.btn_screenshot)
        btn_layout.addWidget(self.btn_recognize_face)

        main_layout = QVBoxLayout()
        main_layout.addLayout(view_layout)
        main_layout.addLayout(btn_layout)

        central_widget = QWidget(self)
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

        self.status_bar = QStatusBar(self)
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage(f"Current Camera: {self.camera_index}")

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)  # 30ms

    def open_camera(self):
        self.camera = cv2.VideoCapture(self.camera_index)
        frame_width = int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.image_label.setMinimumSize(frame_width, frame_height)

    def face_detection(self, face):
        if self.face_remove_background_enabled:
            face = self.face_detector.enhance_image(face, remove_background=True)
        else:
            face = self.face_detector.enhance_image(face, remove_background=False)

        paddle_image = np.zeros(
            (Config.RECOGNITION_SIZE, Config.RECOGNITION_SIZE), dtype=np.uint8
        )
        x_offset = (Config.RECOGNITION_SIZE - face.shape[1]) // 2
        y_offset = (Config.RECOGNITION_SIZE - face.shape[0]) // 2
        paddle_image[
            y_offset : y_offset + face.shape[0], x_offset : x_offset + face.shape[1]
        ] = face
        face_rgb = cv2.cvtColor(paddle_image, cv2.COLOR_BGR2RGB)
        height, width, channel = face_rgb.shape
        bytes_per_line = channel * width
        qt_face = QImage(
            face_rgb.data, width, height, bytes_per_line, QImage.Format_RGB888
        )
        self.face_label.setPixmap(QPixmap.fromImage(qt_face))

        idx = self.parent.recognizer.predict(paddle_image)
        if idx != -1:
            # print(f"Predicted: Label: {self.parent.dataset.face_list[idx].label}")
            orig_img = self.parent.dataset.face_list[idx].image
            face_rgb = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
            height, width, channel = face_rgb.shape
            bytes_per_line = channel * width
            qt_predict = QImage(
                face_rgb.data, width, height, bytes_per_line, QImage.Format_RGB888
            )
            self.predict_label.setPixmap(QPixmap.fromImage(qt_predict))
            return (
                self.parent.dataset.face_list[idx].label,
                self.parent.dataset.face_list[idx].name,
            )
        else:
            self.predict_label.clear()
            return -1, "Stranger"

    def update_frame(self):
        if self.camera is None:
            return

        ret, frame = self.camera.read()
        if not ret:
            return

        if self.face_recognition_enabled:
            detections = self.face_detector.detect_faces(frame)
            (x, y, w, h) = self.face_detector.get_most_confident_face_region(
                detections, frame.shape[1], frame.shape[0]
            )
            nw = int(w * Config.FACE_EXTEND_FACTOR)
            nh = int(h * Config.FACE_EXTEND_FACTOR)
            nx = int(x - (nw - w) / 2)
            ny = int(y - (nh - h + Config.RECOGNITION_SIZE * 0.3) / 2)
            if nx < 0:
                nx = 0
            if ny < 0:
                ny = 0
            if nx + nw > frame.shape[1]:
                nw = frame.shape[1] - nx
            if ny + nh > frame.shape[0]:
                nh = frame.shape[0] - ny

            face = frame[ny : ny + nh, nx : nx + nw]
            if face.size > 0:
                if nw > nh:
                    scale = Config.RECOGNITION_SIZE / nw
                else:
                    scale = Config.RECOGNITION_SIZE / nh

                face = cv2.resize(
                    face,
                    (int(nw * scale), int(nh * scale)),
                    interpolation=cv2.INTER_AREA,
                )

                label, name = self.face_detection(face)
                if label != -1:
                    cv2.putText(
                        frame,
                        f"{name}",
                        (x + 4, y - 24),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 255, 0),
                        2,
                    )
                else:
                    cv2.putText(
                        frame,
                        f"Stranger",
                        (x + 4, y - 24),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 0, 255),
                        2,
                    )

                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cv2.rectangle(frame, (nx, ny), (nx + nw, ny + nh), (0, 255, 0), 2)

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width, channel = frame_rgb.shape
        bytes_per_line = channel * width
        qt_image = QImage(
            frame_rgb.data, width, height, bytes_per_line, QImage.Format_RGB888
        )
        self.image_label.setPixmap(QPixmap.fromImage(qt_image))

    def on_camera_changed(self, index):
        self.camera_index = self.available_cameras[index]
        self.camera = cv2.VideoCapture(self.camera_index)
        frame_width = int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.image_label.setMinimumSize(frame_width, frame_height)

    def on_screenshot(self):
        ret, frame = self.camera.read()
        if not ret:
            return

        file_name = f"Screenshot_{int(time.localtime().tm_mon)}_{int(time.localtime().tm_mday)}_{int(time.localtime().tm_hour)}_{int(time.localtime().tm_min)}_{int(time.localtime().tm_sec)}.jpg"
        cv2.imwrite(os.path.join(script_directory, file_name), frame)

        self.status_bar.showMessage(f"Screenshot saved to {file_name}")

    def on_remove_background_changed(self, state):
        if state == Qt.Checked:
            self.face_remove_background_enabled = True
        else:
            self.face_remove_background_enabled = False

    def toggle_face_recognition(self):
        if self.face_recognition_enabled:
            self.btn_recognize_face.setText("Recognize Face")
            self.face_recognition_enabled = False
        else:
            if self.parent.recognizer.is_model_trained:
                self.btn_recognize_face.setText("Stop Recognizing")
                self.face_recognition_enabled = True
            else:
                QMessageBox.critical(self, "Error", "Model not trained yet!")

    def closeEvent(self, event: QCloseEvent):
        if self.camera is not None:
            self.camera.release()
            self.camera = None
        event.accept()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.camera_window = None
        self.dataset = Dataset.Dataset(self)
        self.recognizer = Recognizer.Recognizer()

        self.setWindowTitle("Face Recognition")
        self.resize(800, 600)

        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)

        main_layout = QVBoxLayout()
        central_widget.setLayout(main_layout)

        top_layout = QHBoxLayout()

        left_group_box = QGroupBox("Opeartions")
        left_layout = QVBoxLayout()
        left_group_box.setLayout(left_layout)

        self.btn_train = QPushButton("Train Model")
        self.btn_open = QPushButton("Open Camera")
        self.btn_load_data = QPushButton("Load Dataset")
        self.btn_save_data = QPushButton("Save Dataset")
        self.btn_import = QPushButton("Import From Folder")

        left_layout.addWidget(self.btn_train)
        left_layout.addWidget(self.btn_open)
        left_layout.addWidget(self.btn_load_data)
        left_layout.addWidget(self.btn_save_data)
        left_layout.addWidget(self.btn_import)
        left_layout.addStretch()

        self.btn_train.clicked.connect(self.on_train_model)
        self.btn_open.clicked.connect(self.on_open_camera)
        self.btn_load_data.clicked.connect(self.dataset.load_dataset)
        self.btn_save_data.clicked.connect(self.dataset.save_dataset)
        self.btn_import.clicked.connect(self.dataset.import_dataset)

        right_layout = QVBoxLayout()

        self.label_imageview = QLabel("")
        self.label_imageview.setAlignment(Qt.AlignCenter)
        self.label_imageview.setStyleSheet("border: 1px solid gray;")
        self.label_imageview.setMinimumSize(Config.VIEW_SIZE, Config.VIEW_SIZE)

        self.table_view = QTableView(self)
        self.table_view.setSelectionMode(QTableView.SingleSelection)
        self.table_view.setSelectionBehavior(QTableView.SelectRows)
        self.table_view.setStyleSheet("border: 1px solid gray;")

        self.table_model = QStandardItemModel(0, 4, self)
        self.table_model.setHorizontalHeaderLabels(
            ["ID", "Name", "Path", "Add to Model"]
        )
        self.table_view.setModel(self.table_model)

        header = self.table_view.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeToContents)

        self.table_model.itemChanged.connect(self.on_item_changed)
        self.table_view.selectionModel().selectionChanged.connect(
            self.on_selection_changed
        )

        right_layout.addWidget(self.label_imageview)
        right_layout.addWidget(self.table_view)

        top_layout.addWidget(left_group_box)
        top_layout.addLayout(right_layout)

        main_layout.addLayout(top_layout)

        progress_layout = QHBoxLayout()
        self.progress_bar = QProgressBar(self)
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)

        self.label_progress = QLabel("Training Progress:")
        self.label_progress.setFixedWidth(160)

        progress_layout.addWidget(self.label_progress)
        progress_layout.addWidget(self.progress_bar)

        main_layout.addLayout(progress_layout)

        self.status_bar = QStatusBar(self)
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready")

        # 一个滑块用于调整阈值
        self.threshold_slider = QSlider(Qt.Horizontal)
        self.threshold_slider.setMinimum(0)
        self.threshold_slider.setMaximum(200)
        self.threshold_slider.setValue(100)  # 初始值
        self.threshold_slider.valueChanged.connect(self.on_threshold_changed)

        self.threshold_label = QLabel("Threshold: 100.0")
        self.threshold_label.setFixedWidth(100)

        # 将滑块添加到布局中
        left_layout.addWidget(QLabel("Threshold:"))
        left_layout.addWidget(self.threshold_slider)
        left_layout.addWidget(self.threshold_label)

    def on_threshold_changed(self, value):
        # 将滑块的值映射到阈值范围
        threshold = value * 0.01
        self.recognizer.distance_threshold = threshold
        self.threshold_label.setText(f"Threshold: {threshold:.2f}")

    def on_train_model(self):
        if not self.dataset.face_list:
            QMessageBox.critical(self, "Error", "No dataset available!")
            return

        start = time.perf_counter()
        self.progress_bar.setValue(0)

        data_matrix, labels = self.dataset.get_data()
        train_matrix, train_labels, test_matrix, test_labels = (
            self.dataset.split_dataset(data_matrix, labels, Config.TRAIN_RATIO)
        )
        self.recognizer.load_data(train_matrix, train_labels, test_matrix, test_labels)
        self.recognizer.fit(Config.KEEP_COMPONENTS)

        self.progress_bar.setValue(35)

        accuracy = self.recognizer.evaluate()

        self.progress_bar.setValue(45)

        train_matrix, train_labels, test_matrix, test_labels = (
            self.dataset.split_dataset(data_matrix, labels, 1)
        )
        self.recognizer.load_data(train_matrix, train_labels, test_matrix, test_labels)
        # print(f"Shape{train_matrix.shape}")
        self.recognizer.fit(Config.KEEP_COMPONENTS)

        self.progress_bar.setValue(100)

        end = time.perf_counter()
        self.status_bar.showMessage(
            f"Training done in {end - start:.3f} seconds. Accuracy: {accuracy * 100:.2f}%"
        )

    def on_open_camera(self):
        if self.camera_window is None:
            self.camera_window = CameraWindows(self)
            self.camera_window.show()
        else:
            self.camera_window.open_camera()
            self.camera_window.show()

    def slot_add_face(self, index, name, path, status):
        # row_index = self.model.rowCount()
        item_id = QStandardItem(str(index))
        item_id.setEditable(False)
        item_name = QStandardItem(name)
        item_name.setEditable(False)
        item_path = QStandardItem(path)
        item_path.setEditable(False)
        item_status = QStandardItem()
        item_status.setEditable(False)
        item_status.setCheckable(True)
        if status:
            item_status.setCheckState(Qt.Checked)
        else:
            item_status.setCheckState(Qt.Unchecked)
        self.table_model.appendRow([item_id, item_name, item_path, item_status])

    def slot_clear_table(self):
        self.table_model.clear()

    def on_item_changed(self, item):
        if item.column() != 3:
            return

        row = item.row()
        self.on_checkbox_toggled(row)

    def on_selection_changed(self, selected, deselected):
        if selected.indexes():
            row = selected.indexes()[0].row()
            face = self.dataset.face_list[row].image
            # id = int(self.table_model.item(row, 0).text())
            # face = None
            # for f in self.dataset.face_list:
            #     if f.id == id:
            #         face = f.image
            #         break

            face = cv2.resize(
                face,
                (Config.VIEW_SIZE, Config.VIEW_SIZE),
                interpolation=cv2.INTER_CUBIC,
            )
            face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            height, width, channel = face_rgb.shape
            bytes_per_line = channel * width
            qt_image = QImage(
                face_rgb.data, width, height, bytes_per_line, QImage.Format_RGB888
            )
            self.label_imageview.setPixmap(QPixmap.fromImage(qt_image))

    def on_checkbox_toggled(self, row):
        status = self.table_model.item(row, 3).checkState()
        self.dataset.face_list[row].status = status
        # face_id = int(self.table_model.item(row, 0).text())
        # for f in self.dataset.face_list:
        #     if f.id == face_id:
        #         f.status = status
        #         break
        # print(f"Face {face_id} status changed to {status}")

    def closeEvent(self, event: QCloseEvent):
        if self.camera_window is not None:
            self.camera_window.close()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    mainWindow = MainWindow()
    mainWindow.show()
    sys.exit(app.exec_())
