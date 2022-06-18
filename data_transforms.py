import numpy as np


# Strict rules for these functions:
# 1. The input should be ONE np.ndarray + any amount of params which can be provided in a dictionary.
# 2. The output should by ONE np.ndarray ONLY. It's shape should be the same as the input np.ndarray.


def zero_mean(x: np.ndarray) -> np.ndarray:
    return x - x.mean()


def car_filter(x: np.ndarray, filter_type: str) -> np.ndarray:
    res = x.copy()
    if filter_type == 'mean':
        term = np.mean(x, axis=1)
    elif filter_type == 'median':
        term = np.median(x, axis=1)
    else:
        raise ValueError(f'CAR filter type "{filter_type}" is not supported')
    for e in range(x.shape[1]):
        res[:, e] = x[:, e] - term
    return res


TRANSFORM_FUNCTIONS = {
    'zero mean': zero_mean,
    'CAR filter': car_filter
}
