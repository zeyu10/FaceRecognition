import os
import cv2 as cv
import numpy as np
from PIL import Image


def face_detect(img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    face_cascade = cv.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
    face = face_cascade.detectMultiScale(gray, 1.1, 4)
    for (x, y, w, h) in face:
        cv.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 1)
    cv.imshow('face_detect_result', img)
    # return img

raw_img = cv.imread('data-test/indoor_022.png')
# cv.imshow('read_img_show', img)

resize_img = cv.resize(raw_img, (100, 100))
# cv.imwrite('data-test/gray_img_022.png', resize_img)

# print('raw', img.shape)
# print('resize', resize_img.shape)

face_detect(resize_img)

while True:
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
cv.destroyAllWindows()
