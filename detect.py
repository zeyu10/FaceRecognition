import os
import cv2 as cv

recognizer = cv.face.LBPHFaceRecognizer_create()
recognizer.read('data-trainer/trainer.yml')

names = []
# names = ["Jerry", "Tom", "Jane", "Doe", "John"]
def face_detect(img):

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    face_detector = cv.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
    face = face_detector.detectMultiScale(gray, 1.1, 4)

    for (x, y, w, h) in face:

        cv.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 1)
        id, confidence = recognizer.predict(gray[y:y+h, x:x+w])

        print(f"Predicted ID: {id}, Confidence: {confidence}")
        # print(f"Total Names: {len(names)}")


        if confidence < 100:
            # cv.putText(img, names[id], (x+2, y+h-5), cv.FONT_HERSHEY_SIMPLEX, 0.8, (150, 255, 0), 1, cv.LINE_AA)
            cv.putText(img, str(id), (x+2, y+h-5), cv.FONT_HERSHEY_SIMPLEX, 0.8, (150, 255, 0), 1, cv.LINE_AA)
        else:
            cv.putText(img, 'unknown', (x+2, y+h-5), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 1, cv.LINE_AA)
    
    cv.imshow('face_detect_result', img)

if __name__ == '__main__':
    
    raw_img = cv.imread('data-test/indoor_005.png')
    # raw_img = cv.imread('data-test/indoor_015.png')
    resize_img = cv.resize(raw_img, (100, 100))

    face_detect(resize_img)

    while True:
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    
    cv.destroyAllWindows()

'''
if __name__ == '__main__':
    
    data_path = "data-ToDo"
    for filename in os.listdir(data_path):
        if filename.endswith(".jpg") or filename.endswith(".png"):

            img_path = os.path.join(data_path, filename)
            img = cv.imread(img_path)
            
            face_detect(img)

    while True:
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    
    cv.destroyAllWindows()
'''