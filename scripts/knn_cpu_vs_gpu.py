"""scikit-learn (CPU) against cuML (GPU) k-NN on the five outer folds: label changes and ties."""
from time import perf_counter

import numpy as np
import pandas as pd
from cuml.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.metrics import f1_score

from common import OUTPUT, cpu_knn_proba, knn_fold, labels

reproduced = pd.read_csv(OUTPUT / "ass4_baseline_reproduction.csv").query("model == 'knn'").set_index("outer_fold")
rows = []
for fold in range(1, 6):
    Z_train, Z_test, codes, truth, params = knn_fold(fold)
    k, weights = params["model__n_neighbors"], params["model__weights"]
    metric = {1: "manhattan", 2: "euclidean"}[params["model__p"]]

    cpu_proba = cpu_knn_proba(fold)
    assert cpu_proba.index.equals(truth.index)
    cpu = labels(cpu_proba)
    cpu_f1 = f1_score(truth, cpu, average="macro")
    assert np.isclose(cpu_f1, reproduced.loc[fold, "macro_f1"]), "Cached CPU predictions are not the notebook's baseline."

    gpu, seconds = [], []
    for _ in range(2):
        start = perf_counter()
        model = KNeighborsClassifier(n_neighbors=k, metric=metric, weights=weights).fit(Z_train, codes)
        gpu.append(labels(model.predict_proba(Z_test)))
        seconds.append(perf_counter() - start)

    distances, _ = NearestNeighbors(n_neighbors=k + 1, metric=metric).fit(Z_train).kneighbors(Z_test)
    distances = np.asarray(distances)
    tied = distances[:, k - 1] == distances[:, k]
    differs = cpu != gpu[0]
    rows.append({
        "outer_fold": fold,
        "k": k,
        "cpu_macro_f1": cpu_f1,
        "gpu_macro_f1": f1_score(truth, gpu[0], average="macro"),
        "cpu_gpu_disagreement": differs.mean(),
        "gpu_run_to_run_disagreement": (gpu[0] != gpu[1]).mean(),
        "boundary_tie_share": tied.mean(),
        "disagreements_at_a_tie": tied[differs].mean(),
        "gpu_seconds": np.mean(seconds),
    })
    print(pd.Series(rows[-1]).round(4).to_dict(), flush=True)

result = pd.DataFrame(rows).set_index("outer_fold")
result["difference"] = result["gpu_macro_f1"] - result["cpu_macro_f1"]
result.to_csv(OUTPUT / "ass4_knn_cpu_vs_gpu.csv")
print(result.round(4).to_string())

fold_metrics = pd.read_csv(OUTPUT / "ass4_fold_metrics.csv").pivot(index="outer_fold", columns="model", values="macro_f1")
print(f"\nGPU - CPU macro F1: {result['difference'].mean():+.4f} ± {result['difference'].std():.4f} "
      f"(folds: {', '.join(f'{d:+.4f}' for d in result['difference'])})")
print(f"k-NN fold-to-fold std:     {fold_metrics['knn'].std():.4f}")
print(f"Stack S2 - A2 tree (mean): {(fold_metrics['S2'] - fold_metrics['tree']).mean():+.4f}")
print(f"k-NN - A2 tree (mean):     {(fold_metrics['knn'] - fold_metrics['tree']).mean():+.4f}")
