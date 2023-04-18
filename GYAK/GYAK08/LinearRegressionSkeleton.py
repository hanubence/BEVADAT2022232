import numpy as np
import pandas as pd

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from matplotlib import pyplot as plt


class LinearRegression:
    def __init__(self, t_feature, t_class, epochs: int = 1000, lr: float = 0.0001):
        self.epochs = epochs
        self.lr = lr

        self.m = 0
        self.c = 0

        iris = load_iris()
        self.df = pd.DataFrame(iris.data, columns=iris.feature_names)

        self.X = self.df['petal width (cm)'].values
        self.y = self.df['sepal length (cm)'].values

    def fit(self, X: np.array, y:np.array):

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)

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
            #if i % 100 == 0:
                #print(np.mean(self.y_train-y_pred))

    def predict(self, X):
        self.pred = []
        for x in X:
            y_pred = self.m*x + self.c
            self.pred.append(y_pred)

        print(self.pred)
        print(self.y_test)

    def MAE(self) -> float:
        return np.mean(np.abs(self.pred - self.y_test))
    
    def MSE(self) -> float:
        return np.mean((self.pred - self.y_test)**2)

    def plot_result(self):
        self.y_pred = self.m*self.X_test + self.c

        plt.scatter(self.X_test, self.y_test)
        plt.plot([min(self.X_test), max(self.X_test)], [min(self.y_pred), max(self.y_pred)], color='red') # predicted
        plt.show()
