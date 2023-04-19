import numpy as np
import pandas as pd

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


class LinearRegression:
    def __init__(self, epochs: int = 1000, lr: float = 0.0001):
        self.epochs = epochs
        self.lr = lr

        self.m = 0
        self.c = 0

        iris = load_iris()
        df = pd.DataFrame(iris.data, columns=iris.feature_names)

        self.X = df['petal width (cm)'].values
        self.y = df['sepal length (cm)'].values

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)

    def fit(self, X: np.array, y:np.array):

        n = float(len(self.X_train)) # Number of elements in X
        
        # Performing Gradient Descent 
        self.losses = []
        for i in range(self.epochs): 
            y_pred = self.m*self.X_train + self.c  # The current predicted value of Y

            residuals = y_pred - self.y_train
            loss = np.sum(residuals ** 2)
            self.losses.append(loss)
            D_m = (-2/n) * sum(self.X_train * residuals)  # Derivative wrt m
            D_c = (-2/n) * sum(residuals)  # Derivative wrt c
            self.m = self.m + self.lr * D_m  # Update m
            self.c = self.c + self.lr * D_c  # Update c

    def predict(self, X):
        return self.m*X + self.c
