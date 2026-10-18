"""Shared data, outer folds and helpers for the scripts. Run them from Assignment4/."""
import json
from pathlib import Path

import nbformat
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, RobustScaler, StandardScaler
from sklearn.utils.validation import check_is_fitted

A2_OUTPUT = Path("../Assignment2/output")
OUTPUT = Path("output")
_NOTEBOOK = nbformat.read("assignment4.ipynb", as_version=4)


def notebook_cell(prefix):
    return next(cell.source for cell in _NOTEBOOK.cells if cell.source.startswith(prefix))


exec(notebook_cell("class TrafficPreprocessor"))  # the notebook stays the single source of the preprocessor

raw = pd.read_csv("../Assignment2/networkTraffic.csv", na_values=["?"])
mapping = pd.read_csv(A2_OUTPUT / "ass2_class_mapping.csv").sort_values("attack_code")
CLASSES = mapping["attack_class"].tolist()
ALPHABETICAL = sorted(CLASSES)
y = raw["attack_cat"].astype(int).map(dict(zip(mapping["attack_code"], mapping["attack_class"])))
X = raw.drop(columns=["id", "attack_cat"])
NOMINAL = ["proto", "state", "service"]
NUMERIC = X.select_dtypes(include=np.number).columns.tolist()
groups = pd.util.hash_pandas_object(X, index=False)
outer_fold = pd.read_csv(OUTPUT / "ass4_outer_folds.csv", index_col="source_index")["outer_fold"]
A2_KNN = pd.read_csv(A2_OUTPUT / "ass2_selected_parameters_full.csv").query("model == 'k-NN'").set_index("outer_fold")
assert outer_fold.index.equals(X.index) and y.notna().all()


def knn_fold(fold):
    """(Z_train, Z_test, train codes, test labels, parameters) of the A2 k-NN on one outer fold."""
    params = json.loads(A2_KNN.loc[fold, "best_parameters"])
    preprocess = TrafficPreprocessor(tuple(NUMERIC), tuple(NOMINAL), scaling="standard", correlation_threshold=0.995)
    preprocess.set_params(**{key.removeprefix("preprocess__"): value for key, value in params.items() if key.startswith("preprocess__")})
    train = outer_fold.ne(fold).to_numpy()
    codes = pd.Categorical(y[train], CLASSES).codes.astype(np.int32)
    return preprocess.fit_transform(X[train]), preprocess.transform(X[~train]), codes, y[~train], params


def labels(proba):
    # ties broken alphabetically, as in the notebook
    return pd.DataFrame(np.asarray(proba), columns=CLASSES)[ALPHABETICAL].idxmax(axis=1).to_numpy()


def cpu_knn_proba(fold):
    return pd.read_parquet(OUTPUT / "base_proba" / f"knn_fold{fold}_test.parquet")[CLASSES]
