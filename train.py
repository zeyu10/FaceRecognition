import os
import cv2 as cv
import numpy as np
from PIL import Image

def read_img(img_path):

    faces = []
    names = []

    imagePaths = [os.path.join(img_path, f) for f in os.listdir(img_path)]
    face_detector = cv.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')

    for imagePath in imagePaths:

        img_PIL = Image.open(imagePath).convert('L')
        img_np = np.array(img_PIL, 'uint8')

        faces_detected = face_detector.detectMultiScale(img_np)
        name = os.path.split(imagePath)[-1].split('.')[0]

        for (x, y, w, h) in faces_detected:

            faces.append(img_np[y:y+h, x:x+w])
            names.append(name)

    print("faces:", faces)
    print("names:", names)

    return faces, names

if __name__ == '__main__':

    get_faces, get_names = read_img('data-test')
    label_dict = {name: idx for idx, name in enumerate(set(get_names))}
    get_labels = np.array([label_dict[name] for name in get_names], dtype=np.int32)

    recognizer = cv.face.LBPHFaceRecognizer_create()
    recognizer.train(get_faces, get_labels)
    recognizer.save('data-trainer/trainer.yml')

    if os.path.exists("data-trainer/trainer.yml"):
        print('Completed, trainer.yml is saved')
    else:
        print('ERROR: trainer.yml is not saved')
