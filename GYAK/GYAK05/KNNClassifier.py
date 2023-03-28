import numpy as np
import seaborn as sns
from typing import Tuple
from scipy.stats import mode
from sklearn.metrics import confusion_matrix

class KNNClassifier:

    def __init__(self, k:int, test_split_ratio:float) -> None:
        self.k = k
        self.test_split_ratio = test_split_ratio

    def train_test_split(features:np.ndarray, labels:np.ndarray, test_split_ratio:float) -> None:
        
        test_size = int(len(features) * test_split_ratio)
        train_size = len(features) - test_size
        assert len(features) == test_size + train_size, "Size mismatch!"

        x_train,y_train = features[:train_size,:],labels[:train_size]
        x_test,y_test = features[train_size:train_size+test_size,:], labels[train_size:train_size + test_size]
        return (x_train,y_train,x_test,y_test)