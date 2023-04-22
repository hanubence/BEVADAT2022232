# %%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error

def load_iris_data():
    return load_iris()

def check_data(iris) -> pd.core.frame.DataFrame:
    return pd.DataFrame(iris.data, columns=iris.feature_names).head(5)

def linear_train_data(iris) -> (np.ndarray, np.ndarray):
    X = iris.data[:, 1:]
    y = iris.data[:, 0]

    return (X,y)

def logistic_train_data(iris) -> (np.ndarray, np.ndarray):
    data = np.c_[iris.data, iris.target]
    data = data[data[:,-1] != 2, :]

    return (data[:,0:4], data[:, -1])


def split_data(X, y) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray):
    return train_test_split(X, y, test_size=0.2, random_state=42)

def train_linear_regression(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)

    return model

def train_logistic_regression(X_train, y_train):
    model = LogisticRegression()
    model.fit(X_train, y_train)

    return model

def predict(model, X_test) -> np.ndarray:
    return model.predict(X_test)

def plot_actual_vs_predicted(y_test, y_pred, color='b'):
    fig, ax = plt.subplots()
    ax.scatter(y_test, y_pred, alpha=1, color='green', s=30)
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k-', lw=2)
    ax.set_xlabel('Actual')
    ax.set_ylabel('Predicted')
    ax.set_title('Actual vs Predicted Target Values')
    return fig

def evaluate_model(y_test, y_pred):
    return mean_squared_error(y_test, y_pred)


