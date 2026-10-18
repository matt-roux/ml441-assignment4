"""Single-seed and seed-averaged TabM on outer fold 1: macro F1 and DoS recall.

Result (18 Oct): every seed had DoS recall 0.045-0.049; the five-seed average scored 0.0015 above the mean seed.
"""
import copy, json
from time import perf_counter
import numpy as np, pandas as pd, tabm, torch, torch.nn.functional as F
from sklearn.metrics import f1_score, recall_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import OrdinalEncoder, QuantileTransformer
from common import CLASSES, NOMINAL, NUMERIC, X, notebook_cell, outer_fold, y

DEVICE = torch.device("cuda")
ns = {"SMOKE": False, "RANDOM_STATE": 21, "DEVICE": DEVICE, "CLASS_ORDER": CLASSES, "NUMERIC_COLUMNS": NUMERIC,
      "NOMINAL_COLUMNS": NOMINAL, "copy": copy, "np": np, "pd": pd, "torch": torch, "F": F, "tabm": tabm, "f1_score": f1_score,
      "perf_counter": perf_counter, "StratifiedGroupKFold": StratifiedGroupKFold, "OrdinalEncoder": OrdinalEncoder,
      "QuantileTransformer": QuantileTransformer,
      "class_codes": lambda labels: torch.as_tensor(pd.Categorical(labels, CLASSES).codes, dtype=torch.long, device=DEVICE)}
exec(notebook_cell('MISSING = "__missing__"'), ns)
sel = pd.read_csv("output/ass4_tabm_selected.csv").set_index("outer_fold").loc[1]
tuning = pd.read_csv("output/tabm_tuning/fold1.csv")
params = json.loads(tuning.loc[tuning["val_macro_f1"].idxmax(), "params"]); epochs = max(1, round(sel["epochs"]))
train = outer_fold.ne(1)
data = ns["tabm_data"](X[train], y[train]); num, cat = data["encode"](X[~train]); truth = y[~train]
probs = []
for seed in range(21, 26):
    probs.append(ns["member_proba"](ns["train_tabm"](data, params, epochs, seed)[0], num, cat).mean(1))
    for label, p in ((f"seed {seed}", probs[-1]), (f"average of {len(probs)}", np.mean(probs, 0))):
        pred = np.asarray(CLASSES)[p.argmax(1)]
        print(f"{label:14s} macro F1 {f1_score(truth, pred, average='macro'):.4f}  DoS recall {recall_score(truth, pred, labels=['DoS'], average=None)[0]:.3f}", flush=True)
print("params", params, "epochs", epochs)
