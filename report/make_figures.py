"""Render the report figures and print the numbers the text quotes. Run from Assignment4/.

Only the S2 meta-learner is refitted, and it must reproduce the notebook's fold scores.
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler

sys.path.insert(0, "scripts")
from common import ALPHABETICAL, CLASSES, NUMERIC, OUTPUT, X, groups, outer_fold, y  # noqa: E402

FIGURES = Path("report/figures")
FIGURES.mkdir(exist_ok=True)
FOLDS = range(1, 6)
BLUE, ORANGE = "#2a78d6", "#eb6834"  # categorical slots 1 and 2, validated for colour-vision deficiency
plt.rcParams.update({
    "text.usetex": True, "font.family": "serif", "font.serif": ["Times"], "font.size": 10,
    "axes.labelsize": 10, "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 9,
    "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42,
})


def save(fig, name):
    fig.savefig(FIGURES / f"{name}.pdf")
    plt.close(fig)


def proba(name, fold, part):
    return pd.read_parquet(OUTPUT / "base_proba" / f"{name}_fold{fold}_{part}.parquet")[CLASSES]


def labels(table):
    return table[ALPHABETICAL].idxmax(axis=1)  # ties broken alphabetically, as in the notebook


# Stack S2 outer-test predictions, refitted from the cached base-learner probabilities.
fold_metrics = pd.read_csv(OUTPUT / "ass4_fold_metrics.csv").set_index(["model", "outer_fold"])
meta = pd.read_csv(OUTPUT / "ass4_meta_selected.csv").query("stack == 'S2'").set_index("outer_fold")
assert (meta["representation"] == "proba").all() and (meta["class_weight"] == "none").all()
LEARNERS = ("knn_r", "tree_r", "tabm")
predictions = {name: [] for name in ("tree", "S2", "V2", "tabm", "tree_r")}
for fold in FOLDS:
    train = pd.concat([proba(name, fold, "train").add_prefix(f"{name}:") for name in LEARNERS], axis=1)
    test = pd.concat([proba(name, fold, "test").add_prefix(f"{name}:") for name in LEARNERS], axis=1)
    model = Pipeline([("scale", StandardScaler()), ("model", LogisticRegression(C=meta.loc[fold, "C"], max_iter=2000))])
    model.fit(train, y.loc[train.index])
    s2 = labels(pd.DataFrame(model.predict_proba(test), index=test.index, columns=model.classes_))
    score = f1_score(y.loc[s2.index], s2, average="macro")
    assert np.isclose(score, fold_metrics.loc[("S2", fold), "macro_f1"], atol=1e-9), f"S2 fold {fold}: {score}"
    predictions["S2"].append(s2)
    predictions["V2"].append(labels(sum(proba(name, fold, "test") for name in LEARNERS) / 3))
    for name in ("tree", "tabm", "tree_r"):
        predictions[name].append(labels(proba(name, fold, "test")))
pooled = {name: pd.concat(parts) for name, parts in predictions.items()}
print("S2 refit reproduces the notebook's fold macro F-scores.")

# Numbers quoted in the text.
for name, pred in pooled.items():
    truth = y.loc[pred.index]
    print(f"{name:7s} pooled macro F {f1_score(truth, pred, average='macro'):.4f}  "
          f"DoS recall {recall_score(truth, pred, labels=['DoS'], average=None)[0]:.3f}  "
          f"DoS precision {precision_score(truth, pred, labels=['DoS'], average=None)[0]:.3f}  "
          f"DoS->Exploits {pred[truth == 'DoS'].eq('Exploits').mean():.3f}")
matrices = {
    name: pd.DataFrame(confusion_matrix(y.loc[pooled[name].index], pooled[name], labels=CLASSES, normalize="true"),
                       index=CLASSES, columns=CLASSES)
    for name in ("tree", "S2")
}
print(pd.DataFrame({name: np.diag(matrix) for name, matrix in matrices.items()}, index=CLASSES).round(3))
for true, predicted in (("Backdoor", "Exploits"), ("DoS", "Exploits"), ("Analysis", "Exploits"), ("Fuzzers", "Normal"),
                        ("Worms", "Exploits"), ("Shellcode", "Normal")):
    print(f"{true} -> {predicted}: tree {matrices['tree'].loc[true, predicted]:.3f}, S2 {matrices['S2'].loc[true, predicted]:.3f}")
knn_train = pd.concat([proba("knn_r", fold, "train") for fold in FOLDS])
print(f"Revised k-NN out-of-fold rows with one class at probability 1: {knn_train.max(axis=1).eq(1).mean():.3f}; "
      f"zero entries: {knn_train.eq(0).to_numpy().mean():.3f}")
for fold in FOLDS:  # TabM's balanced class weight for Worms in each inner training split
    train_mask = outer_fold.ne(fold)
    from sklearn.model_selection import StratifiedGroupKFold
    inner = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=21)
    y_train = y[train_mask]
    sizes = [(len(fit), int(y_train.iloc[fit].eq("Worms").sum())) for fit, _ in inner.split(y_train, y_train, groups[train_mask])]
    print(f"fold {fold} inner splits (rows, Worms rows, Worms weight): "
          + ", ".join(f"({n:,}, {w}, {n / (10 * w):.0f})" for n, w in sizes))

# Scaling, label-conflict and class-margin numbers quoted in the text.
numeric = X[NUMERIC]
iqr = numeric.quantile(0.75) - numeric.quantile(0.25)
print(f"median skewness {numeric.skew().median():.1f}; sbytes standard deviation / IQR {numeric['sbytes'].std() / iqr['sbytes']:.0f}")
for fold in FOLDS:
    train = numeric[outer_fold.ne(fold)]
    print(f"fold {fold} zero-IQR features:", [c for c in NUMERIC if train[c].quantile(0.75) == train[c].quantile(0.25)])
scaled = RobustScaler().fit_transform(numeric[outer_fold.ne(1)])  # fold-one training rows, as the selected k-NN scales them
body = pd.DataFrame(scaled)[numeric.loc[outer_fold.ne(1), "response_body_len"].ne(0).to_numpy()]
body = body.sample(1500, random_state=21).to_numpy()
diff = np.abs(body[:750, None] - body[None, 750:])
share = diff[..., NUMERIC.index("response_body_len")] / diff.sum(-1)
print(f"response_body_len above half of the numeric Manhattan distance: {np.mean(share > 0.5):.2f} of sampled pairs")
conflicting = y.groupby(groups).transform("nunique").gt(1)
partners = y[groups.isin(groups[conflicting & y.eq("DoS")]) & y.ne("DoS")]
print(f"Exploits share of other rows in DoS conflicting groups {partners.eq('Exploits').mean():.2f}; "
      f"Exploits / DoS rows {y.eq('Exploits').sum() / y.eq('DoS').sum():.1f}")
class_f1 = {name: f1_score(y.loc[pred.index], pred, labels=CLASSES, average=None) for name, pred in pooled.items()}
print(f"V2 minus S2 over the nine classes other than Worms: "
      f"{np.delete(class_f1['V2'] - class_f1['S2'], CLASSES.index('Worms')).mean():.3f}")
for tree in ("tree", "tree_r"):  # mean outer-fold error correlation with TabM, as in the notebook's Q6 table
    errors = [(t.ne(y.loc[t.index]), m.ne(y.loc[m.index])) for t, m in zip(predictions[tree], predictions["tabm"])]
    print(f"{tree} / TabM error correlation {np.mean([np.corrcoef(a, b)[0, 1] for a, b in errors]):.3f}")

# Paired outer-fold differences against the baseline tree.
paired = pd.read_csv(OUTPUT / "ass4_paired_tests.csv").set_index("comparison")
ROWS = {  # comparison label in the CSV -> label in the figure, top to bottom by mean difference
    "Stack S2 − A2 tree": "Stack S2", "Soft vote V2 − A2 tree": "Soft vote V2", "Stack S1 − A2 tree": "Stack S1",
    "Revised tree − A2 tree": "Revised tree", "Revised k-NN − A2 tree": r"Revised $k$-NN",
    "TabM − A2 tree": "TabM", "A2 k-NN − A2 tree": r"Baseline $k$-NN",
}
fig, ax = plt.subplots(figsize=(3.45, 2.35))
fig.subplots_adjust(left=0.3, right=0.97, bottom=0.19, top=0.86)
for position, (comparison, label) in enumerate(ROWS.items()):
    folds = paired.loc[comparison, [f"fold_{fold}" for fold in FOLDS]].to_numpy(dtype=float)
    row = len(ROWS) - 1 - position
    ax.scatter(folds[:4], [row] * 4, s=16, color=BLUE, zorder=3, label="Folds 1 to 4" if position == 0 else None)
    ax.scatter(folds[4], row, s=22, marker="D", facecolor="white", edgecolor=ORANGE, linewidth=1.1, zorder=3,
               label="Fold 5" if position == 0 else None)
    ax.vlines(folds.mean(), row - 0.3, row + 0.3, color="black", linewidth=1.4, zorder=4,
              label="Mean" if position == 0 else None)
ax.axvline(0, color="0.5", linewidth=0.8)
ax.set_yticks(range(len(ROWS)), list(ROWS.values())[::-1])
ax.set_xlabel("Macro F-score minus baseline tree")
ax.grid(axis="x", alpha=0.25, linewidth=0.5)
ax.legend(loc="lower left", bbox_to_anchor=(-0.05, 1.0), ncol=3, frameon=False, handletextpad=0.3, columnspacing=1.0)
save(fig, "paired_differences")

# Row-normalised confusion matrices, one PDF per sub-figure.
for name, matrix in matrices.items():
    fig, ax = plt.subplots(figsize=(3.45, 3.55))
    fig.subplots_adjust(left=0.33, right=0.98, bottom=0.34, top=0.98)
    ax.imshow(matrix, cmap="Blues", vmin=0, vmax=1)
    for (row, col), value in np.ndenumerate(matrix.to_numpy()):
        if round(100 * value) >= 1:
            ax.text(col, row, f"{100 * value:.0f}", ha="center", va="center", fontsize=8,
                    color="white" if value > 0.5 else "black")
    ax.set_xticks(range(len(CLASSES)), CLASSES, rotation=90)
    ax.set_yticks(range(len(CLASSES)), CLASSES)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.spines[:].set_visible(True)
    ax.tick_params(length=2)
    save(fig, f"confusion_{name}")

# TabM inner-validation macro F-score against ensemble size.
sweep = pd.read_csv(OUTPUT / "ass4_q1_k_sweep.csv", header=[0, 1], index_col=0)
sizes = sweep.index.astype(int)
fig, ax = plt.subplots(figsize=(3.45, 2.2))
fig.subplots_adjust(left=0.2, right=0.97, bottom=0.2, top=0.84)
for column, label, color, style, marker, dodge in (
    ("val_macro_f1", "Ensemble", BLUE, "-", "o", 0.95),
    ("mean_member_f1", "Mean single member", ORANGE, "--", "s", 1.05),
):
    ax.errorbar(sizes * dodge, sweep[column, "mean"], yerr=sweep[column, "std"], color=color, linestyle=style,
                marker=marker, markersize=4, linewidth=1.3, elinewidth=0.8, capsize=2, label=label)
ax.set_xscale("log", base=2)
ax.set_xticks(sizes, [str(size) for size in sizes])
ax.minorticks_off()
ax.set_xlabel(r"Ensemble size $M$")
ax.set_ylabel("Inner macro F-score")
ax.grid(alpha=0.25, linewidth=0.5)
ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, frameon=False)
save(fig, "tabm_ensemble_size")

# Own-class coefficients of the S2 meta-learner (positive values only, so a sequential scale).
coefficients = pd.read_csv(OUTPUT / "ass4_q7_meta_coefficients.csv", index_col=0)
assert (coefficients.to_numpy() > 0).all()
fig, ax = plt.subplots(figsize=(3.45, 2.9))
fig.subplots_adjust(left=0.3, right=0.86, bottom=0.17, top=0.98)
image = ax.imshow(coefficients, cmap="Blues", vmin=0, vmax=2, aspect="auto")
for (row, col), value in np.ndenumerate(coefficients.to_numpy()):
    ax.text(col, row, f"{value:.2f}", ha="center", va="center", fontsize=9, color="white" if value > 1.2 else "black")
ax.set_xticks(range(3), [r"$k$-NN", "Tree", "TabM"])
ax.set_xlabel("Base learner of stack S2")
ax.set_yticks(range(len(CLASSES)), CLASSES)
ax.spines[:].set_visible(True)
ax.tick_params(length=2)
colorbar = fig.colorbar(image, ax=ax, fraction=0.08, pad=0.04)
colorbar.set_label("Coefficient")
save(fig, "meta_coefficients")
print("Figures written to", FIGURES)
