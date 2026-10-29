# Double-well: additional validation on simulated data

**This is not a paper experiment.** It is an extra check of the stitching method on a 1-D system whose truth is known exactly, so we can measure whether the method recovers the force and reproduces held-out distributions. The paper's own experiments are listed in [`../../paper_reproduction/`](../../paper_reproduction/README.md).

System: $dX = -\nabla V(X)\,dt + \sqrt{2\beta}\,dB$ with $V(x) = \tfrac12(|x|-7)^2$, $\beta = 1$ and $X_0 \sim \mathcal N(0,1)$. It is integrated by Euler–Maruyama with step ≤ 0.01, as 2000 paired trajectories observed at 20 times on [0, 4].

| file | what it is |
|---|---|
| [`double_well.ipynb`](double_well.ipynb) | Covers how the data were generated, the true potential and force, simulated trajectories, the distributions over time, observation times versus the model's internal grid, the splits, and the comparison of distributions and forces. It loads saved files only. |
| [`data.py`](data.py) | `list`, `show`, `verify`, `generate` saved datasets |
| [`run.py`](run.py) | `train`, `resume`, `eval`, `plot` on a saved dataset and split |
| `data/dw-001/` | the saved dataset used everywhere (committed) |
| `runs/` | run folders (local, git-ignored) |
| `results/legacy_2026-10-26/` | figures of the earlier results, kept unchanged |

## Data: generate once, then load by ID

```bash
.venv/bin/python experiments/validation/double_well/data.py list
.venv/bin/python experiments/validation/double_well/data.py show dw-001
.venv/bin/python experiments/validation/double_well/data.py verify dw-001      # re-simulate and compare; writes nothing
.venv/bin/python experiments/validation/double_well/data.py generate --seed 1     # only if you want ANOTHER dataset
```

Loading is the default, and nothing ever regenerates data implicitly. `generate` is the only command that simulates. It always creates a new ID (`dw-002`, `dw-003`, …) and never touches an existing dataset. A dataset folder holds:

- `samples.pt`: the actual samples, `X[j, i]` for trajectory `traj_ids[i]` at `times[j]`, plus 12 trajectories at every integration step (used only for the figures).
- `dataset.json`: the potential, initial distribution, diffusion coefficient, integration method and step size, the data-generation seed, the sha256 hashes, and the source commit.
- `splits.json`: for each split, the time indices and trajectory IDs of the train, validation and test parts, and the trajectory IDs each regime observes at each training time.

`dw-001` is seed 0 with the defaults. Its samples, every split and every regime are bit-for-bit the data behind the 2026-10-26 results, and `tests/test_datastore.py` checks this.

**Splits** (`wgf.datasets.SPLITS`) are defined on the 20 observation times. Both endpoints always train. All splits share one partition of the trajectories into 1000 train and 1000 test, which are disjoint.

| split | training times | test times |
|---|---|---|
| `interleaved` | 10, alternating | 10 |
| `midpoint` | 11 (every other + the last) | 9, each midway between two training times |
| `coarse` | 6 (0, 4, 8, 12, 16, 19) | 14 |

There is no validation part, because nothing is tuned or selected on held-out data. The regimes are `dense` (every training time, all training trajectories), `terminal` (first and last training time only) and `trickle` (10 training trajectories per training time). Held-out samples never reach training or initialisation. The paths start from the first and last training snapshots, and the bandwidth comes from the first. A test replaces every held-out sample with NaN and gets bit-identical models.

## Train, evaluate, plot

```bash
.venv/bin/python experiments/validation/double_well/run.py train  --split interleaved [--dataset dw-001] [--epochs 2000] [--init-seed 0] [--checkpoint-every 100]
.venv/bin/python experiments/validation/double_well/run.py resume experiments/validation/double_well/runs/<run>
.venv/bin/python experiments/validation/double_well/run.py eval   experiments/validation/double_well/runs/<run> [--eval-seed 0] [--verify-data]
.venv/bin/python experiments/validation/double_well/run.py plot   experiments/validation/double_well/runs/<run>
.venv/bin/python scripts/show_run.py experiments/validation/double_well/runs/<run>
```

- **`resume`** continues an interrupted run. It keeps finished models, continues the unfinished one from its checkpoint (Adam state, scheduler and history included), and starts the rest. The result is the same as an uninterrupted run with the same planned epochs.
- **`eval --verify-data`** also re-simulates the dataset from its seed and stops with a list of the differences if it doesn't match.
- The run-folder layout is described in the [main README](../../../README.md#run-folders).

- **`train`** loads the dataset by ID. It records the dataset ID, split ID and hashes, the full configuration, the seeds (data generation, initialisation; evaluation seeds go in each `eval.json`) and the source commit. It saves one checkpoint and one training history per model: V_θ and the V\* reference, for each regime.
- **`eval`** loads the checkpoints and the saved dataset, checks the hashes, and writes `metrics.json` with per-time results. It neither trains nor regenerates the data.
- **`plot`** reads `metrics.json` and the histories only, and writes `tables.md`, `w_per_time_<split>.png`, `force.png`, `distributions.png` and `training_history.png`.

The learned model is compared with the simulation through **distributions** (W1/W2 at each held-out time; the KDE against a histogram of the test cells) and through **forces** ($c_V\nabla V_\theta$ against $\nabla V^*$ on the test cells and on a grid). Stochastic paths are not pointwise targets for the stitching paths. The methods in every table are:

- OT interp: a no-physics baseline.
- V_θ KDE: the stitching prediction.
- V_θ SDE: an independent rollout of the learned SDE.
- V\* KDE and V\* SDE: true-dynamics references.

## Results

### Full-length check: `dw-001:interleaved`, 2000 epochs (reproduces the 2026-10-26 numbers)

Run `20261028-173759_full-dw-001-interleaved`, init seed 0, eval seed 0. This is a check that the saved dataset reproduces the earlier held-out results exactly. It is not a paper benchmark. The values are means of W1 over the 10 test times.

| regime | OT interp | V_θ KDE | V_θ SDE | V\* KDE | V\* SDE | c_H (true 1) | force rel. L2, test cells |
|---|---|---|---|---|---|---|---|
| dense | 0.236 | 0.386 | 0.461 | 0.418 | 0.268 | 0.810 | 0.668 |
| terminal | 1.771 | 1.547 | 1.378 | 0.505 | 0.268 | 0.598 | 0.612 |
| trickle | 1.996 | 0.805 | 1.243 | 0.620 | 2.083 | 0.630 | 1.611 |

Every value, including W2 and c_H, equals the table recorded on 2026-10-26 in [`PROGRESS.md`](../../../PROGRESS.md) ("Held-out protocol for the double-well").

### Diagnostic checks: all three splits, 200 epochs

**These are diagnostic checks, not results.** At 200 epochs the models are far from converged. These runs only show that every split works end to end on the saved dataset: train, then eval, then plot. The values are means of W1 over each split's test times.

| split (test times) | regime | OT interp | V_θ KDE | V_θ SDE | V\* KDE | V\* SDE |
|---|---|---|---|---|---|---|
| interleaved (10) | dense | 0.236 | 1.560 | 1.491 | 1.365 | 0.268 |
| | terminal | 1.771 | 1.689 | 1.612 | 1.428 | 0.268 |
| | trickle | 1.996 | 1.758 | 1.597 | 1.652 | 2.083 |
| midpoint (9) | dense | 0.235 | 1.593 | 1.504 | 1.389 | 0.264 |
| | terminal | 1.824 | 1.736 | 1.621 | 1.458 | 0.264 |
| | trickle | 1.813 | 2.752 | 2.606 | 2.721 | 2.037 |
| coarse (14) | dense | 0.282 | 1.618 | 1.515 | 1.379 | 0.264 |
| | terminal | 1.775 | 1.691 | 1.611 | 1.425 | 0.264 |
| | trickle | 1.459 | 3.191 | 2.798 | 3.176 | 2.066 |

### Earlier results (2026-10-26), kept as they were

`results/legacy_2026-10-26/` holds the figures that were in `figures/` before the reorganisation:

- `demo_split_interleaved.png`: the held-out protocol above, reproduced exactly.
- `demo_double_well.png`: step 6b, from the earlier "all times" protocol.
- `demo_rollout.png`: the earlier rollout test. Its 6a quadratic case now lives in `../ou_quadratic/rollout.py`.
- `train_double_well_loo.png`: from the pre-plan prototype, which used the LOO score and 5 snapshots.

The scripts behind the two "all times" figures, `scripts/demo_double_well.py` and `scripts/demo_rollout.py`, are in git at commit `c1d8680`, and their numbers are in `PROGRESS.md`. Their protocol tested held-out *trajectories* at the training times themselves. The split protocol here supersedes it by also holding out the times.
