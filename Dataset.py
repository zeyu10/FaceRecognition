import os
import sys
import cv2
import pickle
import numpy as np
import FaceDetector
import Config
from PyQt5.QtWidgets import QFileDialog, QMessageBox
from PyQt5.QtCore import Qt, QObject, pyqtSignal


class Face:
    def __init__(self, label, id, name, image_path, image, status=True):
        self.label = label
        self.id = id
        self.name = name
        self.image_path = image_path
        self.image = image
        self.status = status


class Dataset(QObject):
    add_face_signal = pyqtSignal(int, str, str, bool)
    clear_table_signal = pyqtSignal()

    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent
        self.add_face_signal.connect(parent.slot_add_face)
        self.clear_table_signal.connect(parent.slot_clear_table)
        self.label_index = 0
        self.id_index = 1
        self.face_list = []
        self.face_detector = FaceDetector.Detector()

    def import_dataset(self):
        folder_path = QFileDialog.getExistingDirectory(self.parent, "Select Folder")
        if not folder_path:
            return

        msg_box = QMessageBox()
        msg_box.setWindowTitle("Query")
        msg_box.setText("Do you want to remove the background of the faces?")
        msg_box.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
        msg_box.setDefaultButton(QMessageBox.Yes)
        ret = msg_box.exec()
        remove_background = ret == QMessageBox.Yes

        print("Selected folder:", folder_path)
        subject_dirs = sorted(
            [
                d
                for d in os.listdir(folder_path)
                if os.path.isdir(os.path.join(folder_path, d))
            ]
        )
        for subject_dir in subject_dirs:
            subject_name = subject_dir
            subject_path = os.path.join(folder_path, subject_dir)
            image_files = sorted(
                [
                    f
                    for f in os.listdir(subject_path)
                    if f.endswith((".jpg", ".jpeg", ".png"))
                ]
            )
            for image_file in image_files:
                image_path = os.path.join(subject_path, image_file)
                image = cv2.imread(image_path)
                if image is None:
                    print(f"Failed to read image: {image_path}")
                    continue

                print(f"Processing {image_path}...")

                if image.shape[0] * image.shape[1] > 62500:
                    print(f"Too Big Image: {image_path}, Cutting...")
                    detections = self.face_detector.detect_faces(image)
                    (x, y, w, h) = self.face_detector.get_most_confident_face_region(
                        detections, image.shape[1], image.shape[0]
                    )
                    nw = int(w * Config.FACE_EXTEND_FACTOR)
                    nh = int(h * Config.FACE_EXTEND_FACTOR)
                    nx = int(x - (nw - w) / 2)
                    ny = int(y - (nh - h + Config.RECOGNITION_SIZE * 0.2) / 2)
                    if nx < 0:
                        nx = 0
                    if ny < 0:
                        ny = 0
                    if nx + nw > image.shape[1]:
                        nw = image.shape[1] - nx
                    if ny + nh > image.shape[0]:
                        nh = image.shape[0] - ny
                    face = image[ny : ny + nh, nx : nx + nw]
                    if nw > nh:
                        scale = Config.RECOGNITION_SIZE / nw
                    else:
                        scale = Config.RECOGNITION_SIZE / nh
                    image = cv2.resize(
                        face,
                        (int(nw * scale), int(nh * scale)),
                        interpolation=cv2.INTER_AREA,
                    )

                h, w = image.shape[:2]
                if w > h:
                    scale = Config.RECOGNITION_SIZE / w
                else:
                    scale = Config.RECOGNITION_SIZE / h
                image = cv2.resize(
                    image,
                    (int(w * scale), int(h * scale)),
                    interpolation=cv2.INTER_AREA,
                )

                if remove_background:
                    image = self.face_detector.enhance_image(image, True)
                else:
                    image = self.face_detector.enhance_image(image, False)

                paddle_image = np.zeros(
                    (Config.RECOGNITION_SIZE, Config.RECOGNITION_SIZE), dtype=np.uint8
                )
                x_offset = (Config.RECOGNITION_SIZE - image.shape[1]) // 2
                y_offset = (Config.RECOGNITION_SIZE - image.shape[0]) // 2
                paddle_image[
                    y_offset : y_offset + image.shape[0],
                    x_offset : x_offset + image.shape[1],
                ] = image

                face = Face(
                    self.label_index,
                    self.id_index,
                    subject_name,
                    image_path,
                    paddle_image,
                )
                self.face_list.append(face)
                self.add_face_signal.emit(self.id_index, subject_name, image_path, True)
                self.id_index += 1

            self.label_index += 1

    def save_dataset(self):
        file_path, _ = QFileDialog.getSaveFileName(
            self.parent, "Save File", "", "Dataset Files (*.pkl)"
        )
        if not file_path:
            return

        with open(file_path, "wb") as f:
            pickle.dump(self.face_list, f)

    def load_dataset(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self.parent, "Open File", "", "Dataset Files (*.pkl)"
        )
        if not file_path:
            return

        with open(file_path, "rb") as f:
            self.clear_table_signal.emit()
            self.face_list = pickle.load(f)

        for face in self.face_list:
            self.add_face_signal.emit(face.id, face.name, face.image_path, face.status)

        self.id_index = self.face_list[-1].id + 1
        self.label_index = self.face_list[-1].label + 1

    def slot_set_face_status(self, label, status):
        for face in self.face_list:
            if face.label == label:
                face.status = status
                break

    def get_data(self):
        matrix = []
        labels = []
        for face in self.face_list:
            if face.status:
                vector = face.image.reshape((-1, 1))
                matrix.append(vector)
                labels.append(face.label)
        matrix = np.hstack(matrix)
        labels = np.array(labels)
        return matrix, labels

    def split_dataset(self, matrix, labels, train_ratio):
        train_data = []
        train_labels = []
        test_data = []
        test_labels = []
        for label in np.unique(labels):
            indices = np.where(labels == label)[0]
            indices = np.random.permutation(indices)
            train_num = int(len(indices) * train_ratio)
            train_indices = indices[:train_num]
            test_indices = indices[train_num:]
            train_data.append(matrix[:, train_indices])
            test_data.append(matrix[:, test_indices])
            train_labels.extend([label] * len(train_indices))
            test_labels.extend([label] * len(test_indices))

        train_data = np.hstack(train_data)
        test_data = np.hstack(test_data)
        train_labels = np.array(train_labels)
        test_labels = np.array(test_labels)
        return train_data, train_labels, test_data, test_labels
