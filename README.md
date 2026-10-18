# ML441 Assignment 4

Ensemble learning for network intrusion classification on UNSW-NB15. Compares TabM, a
neural ensemble with shared weights, and stacking against the nearest neighbour and
decision tree models from Assignment 2.

- [Notebook](assignment4.ipynb): experiments, tuning and analysis.
- [Report](report/27233162RW441assignment4.pdf): methods, results and discussion.
- [Results](output/): cached predictions, scores and figures.
- [Scripts](scripts/): neighbour-search diagnostics and seed checks.

Open `assignment4.ipynb` in Jupyter from the repository root and run the cells in order.
The experiments used Python 3.12 and an NVIDIA GPU. Package versions are recorded in
[ass4_environment.csv](output/ass4_environment.csv).

The notebook expects the Assignment 2 data and saved outputs in a sibling directory:

```text
../Assignment2/networkTraffic.csv
../Assignment2/features.csv
../Assignment2/output/
```

Cached results are reused from `output/`. Set `ML441_EXPERIMENT_MODE=smoke` before
starting the notebook for a shorter run on 10% of predictor groups.
