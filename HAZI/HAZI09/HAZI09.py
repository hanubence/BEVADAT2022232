import numpy as np

from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.datasets import load_digits

class KMeansOnDigits():

    def __init__(self, n_clusters, random_state) -> None:
        self.n_clusters = n_clusters
        self.random_state = random_state

    def load_dataset(self):
        self.digits = load_digits()

    def predict(self):
        self.clusters = KMeans(n_clusters=self.n_clusters, n_init=10).fit_predict(self.digits.data, self.digits.target)

    def get_label(self):
        self.labels = np.zeros_like(self.clusters)
        for i in range(10):
            mask = self.clusters == i
            sub = self.digits.target[mask]
            mode = np.bincount(sub).argmax()
            self.labels[mask] = mode

    def calc_accuracy(self):
        self.accuracy = np.round(accuracy_score(self.labels, self.digits.target),2)

    def confusion_matrix(self):
        self.mat = confusion_matrix(self.labels, self.digits.target)