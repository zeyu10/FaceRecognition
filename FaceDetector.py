import os
import cv2
import inspect
import numpy as np
import Config

script_directory = os.path.dirname(
    os.path.abspath(inspect.getfile(inspect.currentframe()))
)


class Detector:
    def __init__(self):
        self.net = cv2.dnn.readNetFromCaffe(
            os.path.join(script_directory, "Caffemodel", "deploy.prototxt"),
            os.path.join(
                script_directory,
                "Caffemodel",
                "res10_300x300_ssd_iter_140000.caffemodel",
            ),
        )

    def detect_faces(self, image):
        (h, w) = image.shape[:2]
        blob = cv2.dnn.blobFromImage(
            cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0)
        )
        self.net.setInput(blob)
        return self.net.forward()

    def get_face_region(self, detections, threshold, w, h):
        if detections.shape[2] == 0:
            return None

        faces = []
        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > threshold:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                faces.append((startX, startY, endX - startX, endY - startY))

        return faces

    def get_most_confident_face_region(self, detections, w, h):
        if detections.shape[2] == 0:
            return None

        max_confidence = 0
        max_face = None
        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > max_confidence:
                max_confidence = confidence
                max_face = i

        if max_face is None:
            return None
        box = detections[0, 0, max_face, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")

        return startX, startY, endX - startX, endY - startY

    def enhance_image(self, image, remove_background=True):
        ycrcb_face = cv2.cvtColor(image, cv2.COLOR_BGR2YCR_CB)
        channels = cv2.split(ycrcb_face)
        channels = list(channels)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        channels[0] = clahe.apply(channels[0])
        cv2.merge(channels, ycrcb_face)
        image = cv2.cvtColor(ycrcb_face, cv2.COLOR_YCR_CB2BGR)
        if remove_background:
            blurred_face = cv2.GaussianBlur(image, (7, 7), 0)
            mask = np.zeros(image.shape[:2], np.uint8)
            bgd_model = np.zeros((1, 65), np.float64)
            fgd_model = np.zeros((1, 65), np.float64)
            rect = (
                int(Config.RECOGNITION_SIZE * 0.04),
                int(Config.RECOGNITION_SIZE * 0.04),
                image.shape[1] - int(Config.RECOGNITION_SIZE * 0.08),
                image.shape[0] - int(Config.RECOGNITION_SIZE * 0.08),
            )
            cv2.grabCut(
                blurred_face, mask, rect, bgd_model, fgd_model, 1, cv2.GC_INIT_WITH_RECT
            )
            mask2 = np.where(
                (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 1, 0
            ).astype("uint8")
            grabbed_face = image * mask2[:, :, np.newaxis]

            image = cv2.cvtColor(grabbed_face, cv2.COLOR_BGR2GRAY)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image
