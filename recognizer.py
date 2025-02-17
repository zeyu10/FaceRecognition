import os
import cv2
import time
import random
import inspect
import numpy as np

script_directory = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))

class FaceRecognizer():
    train_faces_percentage = 0.8
    total_train_faces = 0
    total_test_faces = 0
    dataset_train = {}
    dataset_test = {}
    image_width = -1
    image_height = -1
    k = 75 # 保留前k个主成分

    def __init__(self, dataset_path, width, height):
        self.image_width = width
        self.image_height = height
        for root, dirs, files in os.walk(dataset_path):
            data_list = []
            class_name = root.split("\\")[-1]
            for file in files:
                data_list.append(os.path.join(root, file))
            
            random.shuffle(data_list)
            train_size = int(len(data_list) * self.train_faces_percentage)
            self.dataset_train[class_name] = data_list[:train_size]
            self.dataset_test[class_name] = data_list[train_size:]
            self.total_train_faces += len(self.dataset_train[class_name])
            self.total_test_faces += len(self.dataset_test[class_name])

        print(f"Total train faces: {self.total_train_faces}, Total test faces: {self.total_test_faces}")
        

    def train(self):
        if self.total_train_faces <= 2:
            print("Not enough train faces to train the model.")
            return

        print("Training started...")
        start_time = time.perf_counter()
        self.label = []
        self.model = np.empty(shape=(self.image_width * self.image_height, self.total_train_faces),dtype=np.float64)
        index = 0
        self.tracer = []
        for class_name, data_list in self.dataset_train.items():
            for image_path in data_list:
                image = cv2.imread(image_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                image_data = np.array(image, dtype = "float64").flatten()
                self.model[:,index] = image_data[:]
                self.label.append(class_name)
                self.tracer.append(image_path)
                index += 1
        
        self.mean = np.mean(self.model, axis=1)

        for i in range(0, self.total_train_faces):
            self.model[:,i] -= self.mean[:]
        
        cov_matrix = np.cov(self.model, rowvar=False)
        # cov_matrix = np.matrix(self.model.T * self.model) / (self.total_train_faces - 1)
        eig_vals, eig_vecs = np.linalg.eig(cov_matrix)
        sort_indices = np.argsort(eig_vals)[::-1]
        eig_vals = eig_vals[sort_indices]
        eig_vecs = eig_vecs[:,sort_indices]
        
        self.eig_vecs = eig_vecs[:,:self.k]
        self.eig_vals = eig_vals[:self.k]

        self.eig_vecs = np.matmul(self.model, self.eig_vecs)
        norms = np.linalg.norm(self.eig_vecs, axis=0)
        self.eig_vecs /= norms

        self.model = np.matmul(self.eig_vecs.T, self.model)

        end_time = time.perf_counter()
        print(f"Training completed in {end_time - start_time:.3f} seconds.")


    def evaluate(self):
        if self.total_test_faces <= 0:
            print("No test faces to evaluate.")
            return

        print("Evaluating started...")
        start_time = time.perf_counter()
        correct_count = 0
        for class_name, data_list in self.dataset_test.items():
            for image_path in data_list:
                image = cv2.imread(image_path)
                print(f"Evaluating {image_path}...")
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                image_data = np.array(image, dtype = "float64").flatten()
                image_data -= self.mean[:]
                image_data = np.matmul(self.eig_vecs.T, image_data)
                diff = np.empty(shape=(self.k, self.total_train_faces), dtype=np.float64)
                for j in range(0, self.total_train_faces):
                    diff[:,j] = self.model[:,j] - image_data
                norms = np.linalg.norm(diff, axis=0)
                closest_face_index = np.argmin(norms)
                print(f"Predicted class: {self.label[closest_face_index]}, Actual class: {class_name}")
                if self.label[closest_face_index] != class_name:
                    imgs = cv2.hconcat([cv2.imread(image_path),cv2.imread(self.tracer[closest_face_index])])
                    cv2.imshow("Incorrect Match", imgs)
                    cv2.waitKey(0)
                if self.label[closest_face_index] == class_name:
                    correct_count += 1
        end_time = time.perf_counter()
        print(f"Evaluating completed in {end_time - start_time:.3f} seconds.")
        print(f"Accuracy: {correct_count / self.total_test_faces * 100.0}%")

if __name__ == '__main__':
    recognizer = FaceRecognizer(script_directory + "\\att_faces",92,112)
    recognizer.train()
    recognizer.evaluate()
    