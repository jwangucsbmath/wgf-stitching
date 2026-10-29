# Experiments

The experiments are grouped by purpose. Both groups use simulated data, as the paper does too, so the data source does not decide the group. All of them use the shared implementation in [`wgf/`](../wgf).

| folder | purpose | contents |
|---|---|---|
| [`paper_reproduction/`](paper_reproduction/README.md) | reproduce what the paper reports | §5.1–5.5, none started yet |
| [`validation/component_checks/`](validation/component_checks) | check single components against exact answers | `forward.py` (step 1, SDE simulator), `residual.py` (step 4, velocity residual) |
| [`validation/ou_quadratic/`](validation/ou_quadratic) | the method on the Gaussian (OU) flow, where everything is closed-form | `fit_quadratic.py` (step 5, V_a), `fit_nn.py` (step 6a, V_θ), `rollout.py` (6a rollout and force recovery; train / eval / plot) |
| [`validation/double_well/`](validation/double_well/README.md) | additional validation on a known 1-D double-well: held-out times and trajectories, three regimes | saved datasets, `run.py` (train / eval / plot), notebook |

Each script runs from the project root, for example `.venv/bin/python experiments/validation/component_checks/forward.py`, and writes figures to its own `figures/` folder. Run folders go to the experiment's `runs/` folder (git-ignored), and `scripts/show_run.py` lists them all.
