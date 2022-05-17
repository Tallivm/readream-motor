import numpy as np

from typing import List, Union


# Main rules for these functions:
# 1. The input is one np.ndarray, plus any amount of params.
# 2. The output is one np.ndarray only. It's shape should be similar to the starting one: if electrode numbers were in
# the second dimension, they should stay there.


def zero_mean_norm(x: np.ndarray) -> np.ndarray:
    return x - x.mean()


def car_filter(x: np.ndarray, car_filter_type: str) -> np.ndarray:
    res = x.copy()
    if car_filter_type == 'mean':
        term = np.mean(x, axis=1)
    elif car_filter_type == 'median':
        term = np.median(x, axis=1)
    else:
        print(f'CAR filter type "{car_filter_type}" is not implemented')
        term = 0
    for e in range(x.shape[1]):
        res[:, e] = x[:, e] - term
    return res


def project_points_to_other(init_space: np.ndarray, needed_points: Union[np.ndarray, List[np.ndarray]]) -> np.ndarray:
    idx = []
    for point in needed_points:
        idx.append((np.abs(init_space - point).sum(1)).argmin())
    return init_space[idx]
