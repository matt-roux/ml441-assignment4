"""GPU utilisation of one workload on outer fold 1.

Usage: python scripts/gpu_utilisation.py {knn-cuml,knn-torch,tabm-large,tabm-small}
"""
import copy
import subprocess
import sys
from time import perf_counter

import numpy as np
import pandas as pd
import tabm
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import OrdinalEncoder, QuantileTransformer

from common import CLASSES, NOMINAL, NUMERIC, X, groups, knn_fold, notebook_cell, outer_fold, y

TABM_EPOCHS = 10
QUERY = ["nvidia-smi", "--query-gpu=utilization.gpu,power.draw,memory.used", "--format=csv,noheader,nounits"]


def prepare(name):
    if name.startswith("knn"):
        Z_train, Z_test, codes, _, params = knn_fold(1)
        k, p, weights = params["model__n_neighbors"], params["model__p"], params["model__weights"]
        if name == "knn-cuml":
            from cuml.neighbors import KNeighborsClassifier

            metric = {1: "manhattan", 2: "euclidean"}[p]
            return lambda: KNeighborsClassifier(n_neighbors=k, metric=metric, weights=weights).fit(Z_train, codes).predict_proba(Z_test)
        from knn_gpu_tie_corrected import knn_proba

        return lambda: knn_proba(Z_train, codes, Z_test, k, p, weights)

    namespace = {
        "SMOKE": False, "RANDOM_STATE": 21, "DEVICE": torch.device("cuda"), "CLASS_ORDER": CLASSES,
        "NUMERIC_COLUMNS": NUMERIC, "NOMINAL_COLUMNS": NOMINAL, "copy": copy, "np": np, "pd": pd, "torch": torch,
        "F": F, "tabm": tabm, "f1_score": f1_score, "perf_counter": perf_counter,
        "StratifiedGroupKFold": StratifiedGroupKFold, "OrdinalEncoder": OrdinalEncoder, "QuantileTransformer": QuantileTransformer,
    }
    exec(notebook_cell('MISSING = "__missing__"'), namespace)
    namespace["MAX_EPOCHS"], namespace["PATIENCE"] = TABM_EPOCHS, TABM_EPOCHS
    width, depth = {"tabm-large": (512, 3), "tabm-small": (256, 2)}[name]
    train = outer_fold.ne(1)
    data = X[train], y[train], groups[train]
    return lambda: namespace["fit_tabm"](*data, {"d_block": width, "n_blocks": depth})[1]


name = sys.argv[1]
run = prepare(name)
idle_memory = float(subprocess.run(QUERY, capture_output=True, text=True).stdout.split(",")[2])
sampler = subprocess.Popen(QUERY + ["-lms", "200"], stdout=subprocess.PIPE, text=True)
start = perf_counter()
result = run()
seconds = perf_counter() - start
sampler.terminate()
samples = np.array([[float(value) for value in line.split(",")] for line in sampler.communicate()[0].splitlines() if line.strip()])

summary = {
    "workload": name,
    "seconds": round(seconds, 1),
    "util_median_%": np.median(samples[:, 0]),
    "util_mean_%": round(samples[:, 0].mean(), 1),
    "power_mean_W": round(samples[:, 1].mean(), 1),
    "power_max_W": samples[:, 1].max(),
    "memory_peak_MiB_above_idle": samples[:, 2].max() - idle_memory,
}
if name.startswith("tabm"):
    summary["train_seconds_per_epoch"] = round(result["seconds"] / TABM_EPOCHS, 2)
print(summary, flush=True)
