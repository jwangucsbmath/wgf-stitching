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
scripts/show_run.py          list run folders, or print one run's configuration and progress
notebooks/                   00_stitching_from_scratch.ipynb, the from-scratch teaching walkthrough
tests/                       pytest suite
PROGRESS.md                  what was built, step by step, with the recorded results
```

Set up and test:

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python -m pytest -q
```

## Data (double-well)

The double-well data are saved once and loaded by ID. `dw-001` is committed, and nothing regenerates it implicitly.

```bash
.venv/bin/python experiments/validation/double_well/data.py list
.venv/bin/python experiments/validation/double_well/data.py show dw-001
.venv/bin/python experiments/validation/double_well/data.py verify dw-001
```

`verify` re-simulates the dataset from its recorded seed and settings and compares the result with the saved files; it writes nothing. To make another dataset, ask for it explicitly. It gets a new ID (`dw-002`, …) and never overwrites an existing one:

```bash
.venv/bin/python experiments/validation/double_well/data.py generate --seed 1
```

In Python:

```python
from data import load        # with experiments/validation/double_well on sys.path
ds = load("dw-001")          # samples, times, trajectory IDs, settings, seeds, splits; checks hashes
train = ds.train("interleaved", "dense")      # {t: cells}, training times and trajectories only
test = ds.held_out("interleaved", "test")     # {t: cells}, held-out times and trajectories
```

## Training, resuming, evaluating, plotting

The double-well commands are below. `experiments/validation/ou_quadratic/rollout.py` takes the same `train`, `resume`, `eval` and `plot` commands; its `train` has no `--split` or `--dataset`. Run everything from the project root.

Train a new run, which goes into a new folder. Every `--checkpoint-every` epochs (default 100) it writes a training checkpoint:

```bash
.venv/bin/python experiments/validation/double_well/run.py train --split interleaved --dataset dw-001 --epochs 2000
```

Resume an interrupted run (Ctrl-C, a killed process, a crash). Finished models are kept, the unfinished one continues from its last checkpoint on the original learning-rate schedule, and the models not yet started begin from their seeded initialisation. The saved dataset is loaded and checked against its hashes:

```bash
.venv/bin/python experiments/validation/double_well/run.py resume experiments/validation/double_well/runs/<run>
```

Evaluate a finished run from its saved models and data. This never trains, and by default it never regenerates the data:

```bash
.venv/bin/python experiments/validation/double_well/run.py eval experiments/validation/double_well/runs/<run>
```

Evaluate and also re-simulate the dataset from its recorded seed, comparing it with the saved one. A mismatch is listed and stops the evaluation:

```bash
.venv/bin/python experiments/validation/double_well/run.py eval experiments/validation/double_well/runs/<run> --verify-data
```

Plot tables and figures from the saved metrics and histories only:

```bash
.venv/bin/python experiments/validation/double_well/run.py plot experiments/validation/double_well/runs/<run>
```

List every saved run, with its experiment, dataset, status and source commit:

```bash
.venv/bin/python scripts/show_run.py
```

Show one run's configuration, seeds, sessions, per-model progress, models, checkpoints and evaluations:

```bash
.venv/bin/python scripts/show_run.py experiments/validation/double_well/runs/<run>
```

To try out resuming, `train` and `resume` accept `--stop-after N`, which stops after N epochs in that session exactly as an interruption would (exit status 3).

## Run folders

Every `train` creates `<experiment>/runs/<YYYYmmdd-HHMMSS>_<name>/`, for example `experiments/validation/double_well/runs/20261028-173759_full-dw-001-interleaved/`. **Run folders stay on your machine.** `runs/` is in `.gitignore`, so they are never committed or pushed; back them up yourself if you need them elsewhere. Each `eval` adds a new `evals/` folder; `plot` (re)writes the tables and figures inside the evaluation it is given.

```
<run>/
  config.json                  configuration, seeds, dataset ID + split + hashes, source commit,
                               package versions, sessions (train / resume), per-model progress, status
  source.diff                  uncommitted changes at train time (only if the tree was dirty);
                               source.session<k>.diff for a later dirty resume
  data.pt                      OU runs only: the run's training and test data (hash in config.json)
  checkpoints/<tag>.pt         training checkpoint: parameters + Adam m, v and step counts, scheduler,
                               completed epochs, history so far, RNG states (resume continues from it)
  models/<tag>.pt              finished model: spec + parameters (what eval loads)
  history/<tag>.json           finished model's history, one value per epoch
  evals/<YYYYmmdd-HHMMSS>_seed<k>/
    eval.json                  evaluation settings and seeds, data check, source commit, runtime
    metrics.json               per-time W1/W2 for each method, force errors, curves for the figures
    predictions.pt             every predicted cloud, by method and time
    tables.md, *.png           written by `plot` from metrics.json and history/ alone
```

| what | where |
|---|---|
| data | double-well: the saved dataset `experiments/validation/double_well/data/<id>/` (committed), named by `config.json` → `config.dataset` with its hashes; OU: `data.pt` in the run |
| configuration | `config.json`: `config` (data, split, model, training, seeds), `provenance`, `sessions`, `progress` |
| model | `models/<tag>.pt` once finished (what `eval` loads); `checkpoints/<tag>.pt` while training (`wgf.runs.load_model` reads both) |
| training history | `history/<tag>.json`: `loss`, `nll`, `R`, `lamR` (λR), `lr`, `h`, `c_H` per epoch, `seconds`, and `resumed_at_epoch` when resumed |
| evaluation results | `evals/<…>/metrics.json` (per-time numbers), `tables.md` and figures after `plot` |

A double-well run has six models (tags `dense_V_theta`, `dense_V_star`, `terminal_…`, `trickle_…`); an OU run has one (`6a_quadratic`). Every file is written to a temporary name and then renamed over the target, so a process killed mid-write leaves the previous version intact. A kill loses at most `--checkpoint-every` epochs.

Runs made before checkpoints existed have only `models/` and `history/`. They evaluate and plot as before, and `resume` reports that they finished. A model-only file cannot be continued exactly: it lacks Adam's moments and step counts and the scheduler's position. If `resume` meets one for an unfinished model, it explains this and stops instead of guessing.

## Notebook and other scripts

[`experiments/validation/double_well/double_well.ipynb`](experiments/validation/double_well/double_well.ipynb) reads the saved dataset and a finished run. It never trains or simulates. See [`experiments/validation/double_well/README.md`](experiments/validation/double_well/README.md) for the protocol and results.

```bash
.venv/bin/python experiments/validation/component_checks/forward.py
.venv/bin/python experiments/validation/component_checks/residual.py
.venv/bin/python experiments/validation/ou_quadratic/fit_quadratic.py
.venv/bin/python experiments/validation/ou_quadratic/fit_nn.py
```
