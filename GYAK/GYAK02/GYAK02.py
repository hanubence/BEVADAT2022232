import numpy as np

def create_array(size: np.array = (2,2)) -> np.array:
    return np.zeros(size, dtype=int)

def set_one(l: np.array) -> np.array:
    np.fill_diagonal(l, 1)
    return l

def do_transpose(l: np.array) -> np.array:
    return np.transpose(l)

def round_array(l: np.array, n: int = 2) -> np.array:
    return np.round(l, n)

def bool_array(l: np.array) -> np.array:
    return np.array(l, dtype=bool)

def invert_bool_array(l: np.array) -> np.array:
    return ~np.array(l, dtype=bool)

def flatten(l: np.array) -> np.array:
    return l.flatten()


