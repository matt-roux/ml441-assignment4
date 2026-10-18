"""Brute-force GPU k-NN in double precision with an explicit tie rule, against scikit-learn."""
from time import perf_counter

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score

from common import CLASSES, OUTPUT, cpu_knn_proba, knn_fold, labels


def knn_proba(Z_train, train_codes, Z_test, k, p, weights, chunk=256):
    train = torch.as_tensor(Z_train, device="cuda").double()
    codes = torch.as_tensor(train_codes, dtype=torch.long, device="cuda")
    proba = []
    for test in torch.as_tensor(Z_test, device="cuda").double().split(chunk):
        # sum in float64 and round to float32, as scikit-learn does, so the same distances tie
        distance = torch.cdist(test, train, p=p, compute_mode="donot_use_mm_for_euclid_dist").float()
        kth = distance.topk(k, largest=False).values[:, -1:]
        closer, tied = distance < kth, distance == kth
        # tied rows fill the remaining slots in training-index order
        chosen = closer | (tied & (tied.cumsum(1) <= k - closer.sum(1, keepdim=True)))
        neighbours = chosen.nonzero(as_tuple=True)[1].view(len(test), k)
        d = distance.gather(1, neighbours)
        if weights == "distance":
            # as in scikit-learn, zero-distance neighbours vote alone
            exact = d == 0
            w = torch.where(exact.any(1, keepdim=True), exact.float(), 1 / d)
        else:
            w = torch.ones_like(d)
        votes = torch.zeros(len(test), len(CLASSES), dtype=w.dtype, device="cuda").scatter_add_(1, codes[neighbours], w)
        proba.append(votes / votes.sum(1, keepdim=True))
    return torch.cat(proba).cpu().numpy()


if __name__ == "__main__":
    rows = []
    for fold in range(1, 6):
        Z_train, Z_test, codes, truth, params = knn_fold(fold)
        k = params["model__n_neighbors"]
        start = perf_counter()
        gpu_proba = knn_proba(Z_train, codes, Z_test, k, params["model__p"], params["model__weights"])
        seconds = perf_counter() - start
        cpu_proba = cpu_knn_proba(fold).to_numpy()
        cpu, gpu = labels(cpu_proba), labels(gpu_proba)
        rows.append({
            "outer_fold": fold,
            "k": k,
            "cpu_macro_f1": f1_score(truth, cpu, average="macro"),
            "gpu_macro_f1": f1_score(truth, gpu, average="macro"),
            "label_disagreement": (cpu != gpu).mean(),
            "rows_with_different_proba": (~np.isclose(cpu_proba, gpu_proba, atol=1e-6).all(axis=1)).mean(),
            "gpu_seconds": seconds,
        })
        print(pd.Series(rows[-1]).round(5).to_dict(), flush=True)
    result = pd.DataFrame(rows).set_index("outer_fold")
    result.to_csv(OUTPUT / "ass4_knn_gpu_tie_corrected.csv")
    print(result.round(5).to_string())
