import numpy as np
from numpy.typing import NDArray


def mean_square_error(y_true: NDArray[np.float64], y_predict: NDArray) -> np.float64:
    error = y_true - y_predict
    mean_squared_error = np.mean(error ** 2)

    return mean_squared_error


def root_mean_square_error(y_true: NDArray[np.float64], y_predict: NDArray) -> np.float64:
    error = y_true - y_predict
    mean_squared_error = np.mean(error ** 2)
    root_mean_squared_error = np.sqrt(mean_squared_error)

    return root_mean_squared_error


def mean_absolute_error(y_true: NDArray[np.float64], y_predict: NDArray) -> np.float64:
    error = y_true - y_predict
    absolute_error = np.abs(error)
    mean_absolute_error = np.mean(absolute_error)

    return mean_absolute_error
