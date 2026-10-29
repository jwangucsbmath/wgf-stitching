# wgf-stitching
Clean-room PyTorch reproduction of learning Wasserstein gradient flows from snapshot data via particle stitching and Wasserstein residuals (arXiv:2607.04738).

## Layout

```
wgf/                         shared implementation: simulator, KDE, forces, residual, model, training,
                             metrics, datasets, saved-dataset store (datastore.py), run records (runs.py)
experiments/
  paper_reproduction/        the paper's experiments (§5.1–5.5); none reproduced yet
  validation/                additional validation on simulated data (not paper experiments)
    component_checks/        simulator and residual against exact answers
    ou_quadratic/            the method on the Gaussian (OU) flow
    double_well/             1-D double-well: saved datasets, held-out splits, train / eval / plot, notebook
scripts/show_run.py          list run folders, or print one run's configuration
notebooks/                   00_stitching_from_scratch.ipynb, the from-scratch teaching walkthrough
tests/                       pytest suite
PROGRESS.md                  what was built, step by step, with the recorded results
```

Set up and test:

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python -m pytest -q
```

## Double-well validation: data, training, evaluation, plots

All commands run from the project root. See [`experiments/validation/double_well/README.md`](experiments/validation/double_well/README.md) for the protocol and results.

Load the saved data. This is the default: `dw-001` is committed, and nothing regenerates it.

```bash
.venv/bin/python experiments/validation/double_well/data.py list
.venv/bin/python experiments/validation/double_well/data.py show dw-001
```

```python
from data import load        # with experiments/validation/double_well on sys.path
ds = load("dw-001")          # samples, times, trajectory IDs, settings, seeds, splits; checks hashes
train = ds.train("interleaved", "dense")      # {t: cells}, training times and trajectories only
test = ds.held_out("interleaved", "test")     # {t: cells}, held-out times and trajectories
```

To generate another dataset, ask for it explicitly. It gets a new ID (`dw-002`, …) and never overwrites one that exists:

```bash
.venv/bin/python experiments/validation/double_well/data.py generate --seed 1
```

Train on a saved dataset and split. This creates a new run folder recording the dataset and split IDs, the configuration, the seeds and the source commit, along with the checkpoints and the training histories:

```bash
.venv/bin/python experiments/validation/double_well/run.py train --split interleaved --dataset dw-001 --epochs 2000
```

Evaluate a trained run. This loads the checkpoints and the saved dataset; it does not train or regenerate data:

```bash
.venv/bin/python experiments/validation/double_well/run.py eval experiments/validation/double_well/runs/<run>
```

Plot from the saved metrics only:

```bash
.venv/bin/python experiments/validation/double_well/run.py plot experiments/validation/double_well/runs/<run>
```

Inspect runs:

```bash
.venv/bin/python scripts/show_run.py
```

The notebook [`experiments/validation/double_well/double_well.ipynb`](experiments/validation/double_well/double_well.ipynb) reads the same saved dataset and a finished run.

## Other validation scripts

```bash
.venv/bin/python experiments/validation/component_checks/forward.py
.venv/bin/python experiments/validation/component_checks/residual.py
.venv/bin/python experiments/validation/ou_quadratic/fit_quadratic.py
.venv/bin/python experiments/validation/ou_quadratic/fit_nn.py
.venv/bin/python experiments/validation/ou_quadratic/rollout.py train
```

`rollout.py` has the same `train`, `eval <run>` and `plot <run>` commands as the double-well `run.py`.
