import os
import json
import numpy as np
from datetime import datetime
from sklearn.decomposition import PCA

from typing import List, Tuple


def import_params(param_file: str = 'config.json'):
    with open(param_file, 'r') as f:
        params = json.load(f)
    return params


def log_params() -> str:
    runtime = datetime.now().strftime("%m.%d.%H.%M.%S")
    params = import_params()
    log_dir = params["FILE STRUCTURE"]["LOGS DIRECTORY"]
    logfile = os.path.join(log_dir, generate_name(runtime, '.log'))
    with open(logfile, 'w') as f:
        json.dump(params, f, indent=4)
    return runtime


def generate_name(runtime: str, extension: str = ''):
    params = import_params()
    name = [
        runtime,
        params["DATA BUILDER PARAMETERS"]["DATA FEATURES"],
        '-'.join(d["DATASET NAME"] for d in params["DATASETS"]),
        '-'.join(f'{r[0]}to{r[1]}' for r in params["DATA BUILDER PARAMETERS"]["BANDS OF INTEREST"]),
        f'{params["TRAIN PARAMETERS"]["N EPOCHS"]}ep',
        'permuted' if params["DATA BUILDER PARAMETERS"]["PERMUTE LOCATIONS"] else '',
    ]
    name = '_'.join([str(n) for n in name])
    return name + extension


def transform_with_pca(x: np.ndarray) -> (PCA, np.ndarray):
    pca = PCA(n_components=2)
    pca.fit(x)
    return pca, pca.transform(x)


def project_points_onto_other(points: np.ndarray, project_onto: np.ndarray) -> np.ndarray:
    """
    Project set of points to another set of points to get new coordinates.

    points : coordinates of points in N dimensions.
    project_onto : coordinates of points to which the projection is done, should have same N dimensions.
    """
    idx = []
    for point in points:  # TODO: vectorize
        idx.append((np.abs(project_onto - point).sum(1)).argmin())
    return project_onto[idx]


def split_to_batches(x: List[Tuple[int, int]], batch_size: int) -> List[np.ndarray]:
    """
    Split a set of fragments into batches.

    x : set of tuple indices indicating the source of the fragment.
    batch_size : batch size.
    """
    batches = []
    for i in range(0, len(x), batch_size):
        batches.append(np.array(x[i: i + batch_size]))
    return batches
