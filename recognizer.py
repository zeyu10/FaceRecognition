import numpy as np
import Config


class Recognizer:
    def __init__(self):
        self.train_matrix = None
        self.train_labels = None
        self.test_matrix = None
        self.test_labels = None
        self.mean_face = None
        self.centered_data = None
        self.eigenfaces = None
        self.train_weights = None
        self.is_model_trained = False
        self.k = Config.CANDIDATE_K[0]
        self.distance_threshold = 0.0

    def load_data(self, train_matrix, train_labels, test_matrix, test_labels):
        self.train_matrix = train_matrix
        self.train_labels = train_labels
        self.test_matrix = test_matrix
        self.test_labels = test_labels

    def fit(self, keep_components_ratio):
        self.mean_face = self.train_matrix.mean(axis=1, keepdims=True)
        self.centered_data = self.train_matrix - self.mean_face

        cov_matrix = np.dot(self.centered_data.T, self.centered_data) / (
            self.train_matrix.shape[1] - 1
        )
        eig_vals, eig_vecs = np.linalg.eig(cov_matrix)
        sorted_indices = np.argsort(eig_vals)[::-1]
        eig_vecs = eig_vecs[:, sorted_indices]
        eig_vals = eig_vals[sorted_indices]

        eig_vals_total = np.sum(eig_vals)
        variance_explained = np.cumsum(eig_vals) / eig_vals_total
        num_components = np.argmax(variance_explained >= keep_components_ratio) + 1
        print("Number of components:", num_components)

        eig_vecs = eig_vecs[:, :num_components]
        self.eigenfaces = np.dot(self.centered_data, eig_vecs)
        eigenface_norms = np.linalg.norm(self.eigenfaces, axis=0, keepdims=True)
        eigenface_norms[eigenface_norms == 0] = 1
        self.eigenfaces /= eigenface_norms
        print(f"Eigenfaces shape: {self.eigenfaces.shape}")

        train_weights = np.dot(self.eigenfaces.T, self.centered_data)
        train_weights_norms = np.linalg.norm(train_weights, axis=0, keepdims=True)
        train_weights_norms[train_weights_norms == 0] = 1
        train_weights /= train_weights_norms
        self.train_weights = train_weights

        squared = np.sum(self.train_weights**2, axis=0)
        distance_matrix = (
            squared[:, np.newaxis]
            + squared[np.newaxis, :]
            - 2 * np.dot(self.train_weights.T, self.train_weights)
        )
        np.fill_diagonal(distance_matrix, np.inf)
        min_distances = np.min(distance_matrix, axis=1)

        max_legal = np.max(min_distances)
        mean = np.mean(min_distances)
        std = np.std(min_distances)
        mean_plus_3std = mean + 3 * std
        self.distance_threshold = max(max_legal, mean_plus_3std)

        print(f"Computed threshold: {self.distance_threshold: .2f}")
        self.is_model_trained = True

    def evaluate(self):
        test_data_centered = self.test_matrix - self.mean_face
        test_weights = np.dot(self.eigenfaces.T, test_data_centered)
        test_weights_norms = np.linalg.norm(test_weights, axis=0, keepdims=True)
        test_weights_norms[test_weights_norms == 0] = 1
        test_weights /= test_weights_norms

        test_sq = np.sum(test_weights**2, axis=0)
        train_sq = np.sum(self.train_weights**2, axis=0)
        cross = np.dot(test_weights.T, self.train_weights)
        dist_sq = test_sq[:, np.newaxis] + train_sq - 2 * cross

        predicted_indices = np.zeros(len(self.test_labels), dtype=int)

        best_k = Config.CANDIDATE_K[0]
        best_accuracy = -1
        for k in Config.CANDIDATE_K:
            for i in range(dist_sq.shape[0]):
                distances = dist_sq[i, :]
                best_index = self.kNNR_classifier(distances, k, self.train_labels)
                predicted_indices[i] = best_index
            accuracy = np.mean(self.train_labels[predicted_indices] == self.test_labels)
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_k = k
            print(f"K={k}, Accuracy={accuracy}")

        self.k = best_k
        print("Final Accuracy:", best_accuracy)
        return best_accuracy

    def predict(self, img):
        if not self.is_model_trained:
            print("Model not trained yet!")
            return -1

        img_vector = img.reshape((-1, 1))
        img_centered = img_vector - self.mean_face
        img_weights = np.dot(self.eigenfaces.T, img_centered)
        img_weights_norms = np.linalg.norm(img_weights, axis=0, keepdims=True)
        img_weights_norms[img_weights_norms == 0] = 1
        img_weights /= img_weights_norms

        dists = np.sum((self.train_weights - img_weights) ** 2, axis=0)
        best_index = self.kNNR_classifier(dists, self.k, self.train_labels)

        min_distance = dists[best_index]
        if min_distance > self.distance_threshold:
            return -1
        # print("Predicted label:", self.train_labels[best_index])
        return best_index

    def kNNR_classifier(self, distances, k, train_labels):
        knn_indices = np.argpartition(distances, k - 1)[:k]
        knn_labels = train_labels[knn_indices]

        unique_labels, counts = np.unique(knn_labels, return_counts=True)
        max_count = counts.max()
        candidate_labels = unique_labels[counts == max_count]

        best_index = None
        best_distance = float("inf")
        for label in candidate_labels:
            indices = knn_indices[knn_labels == label]
            local_idx = indices[np.argmin(distances[indices])]
            if distances[local_idx] < best_distance:
                best_distance = distances[local_idx]
                best_index = local_idx

        return best_index
