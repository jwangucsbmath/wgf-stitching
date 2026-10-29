# Progress — clean-room Wasserstein residuals ("stitching"), arXiv:2607.04738

Following `wgf_implementation_plan.pdf` (steps 1–7). PyTorch, float64 in tests, every random draw from a seeded `torch.Generator`.
No code from BasisResearch/wasserstein-residuals.

Paths in this log were updated on 2026-10-28, when `scripts/demo_*.py` and `figures/` moved into `experiments/` (see the last section).

Run everything:
```bash
.venv/bin/python -m pytest -q
```

Earlier work (before the plan, 2026-10-26): a first end-to-end prototype in `notebooks/00_stitching_from_scratch.ipynb`. It was a teaching copy with its own inline code; we found the learned double-well barrier at x ≈ −0.43. The package code from that phase was replaced step by step by the plan's modules.

---

## Step 1 - SDE simulator (2026-10-26)
Built: `wgf/sim.py` with `drift`, `em_step(X, dt, beta, grad_V, grad_W=None, w=None, gen=None)`, `heun_step`, and `simulate(X0, t_obs, dt, beta, grad_V, grad_W=None, w=None, paired=True, gen=None, scheme="euler")`, which returns `[(t, X_t)]`. `wgf/potentials.py` holds V (quadratic, double-well, wavy valley) and W (quadratic, Gaussian) with closed-form gradients, plus `autograd_grad`. `experiments/validation/component_checks/forward.py` produces `experiments/validation/component_checks/figures/forward.png`.
Tests: `tests/test_sim.py`, 27 passing. The six plan tests:
1. Noise scale: at β = 0.3 the variance is 0.6 at t = 1, within 5%.
2. Gaussian flow: halving Δt halves the bias of both the mean and the variance (ratio between 1.6 and 2.5).
3. Double-well: |x| has mean 7 ± 0.05 and variance 1 ± 0.05.
4. Centre of mass: it drifts by less than 1e-12.
5. Determinism, and same seed gives the same output.
6. Paired vs unpaired, for shapes and for particle identity.

Extras: vectorised = frozen loop, the in-place loop is order-dependent, Euler order 1 / Heun order 2 on an exact interacting flow, observation times are hit exactly, and a learned (autograd) gradient works inside the `no_grad` stepper. The demo histograms lie on the exact Gaussian densities: sample moments are within 2.5% at t = 0…4.
Choices:
- `X0` can be a tensor or a sampler `gen -> X0`. For `paired=False` with a fixed tensor, each snapshot is a fresh run with fresh noise and permuted rows.
- `grad_V=None` means V = 0.
- An optional `scheme="heun"` is available.
- Wavy-valley tilt is 0.3 (paper App. E).

Open: none.

## Step 2 - KDE, score, NW velocity (2026-10-26)
Built: `wgf/kde.py` with:
- `log_density(y, centres, h, w)` (Eq. 8)
- `score` (Eq. 36)
- `score_at_centres(centres, h, w, loo=False)` (Eq. 37 when `loo=True`; also takes `(K, N, D)` to do all slices at once)
- `nw_velocity` (Eq. 9)
- `silverman(x)` ($h_d=\hat\sigma_d(4/((D+2)n))^{1/(D+4)}$)
- `sample` (systematic resampling)

Tests: `tests/test_kde.py`, 12 passing.
- ∫ρ = 1 to within 1e-6.
- The score equals autograd to 1e-10, for the full score and for leave-one-out.
- The notes' worked example (full −0.64, LOO −1.83).
- **Gaussian slope −1.716** (target −1.724 ± 5%; the plan measured −1.754), which is clearly not −2.041.
- NW velocity: constant velocities are returned exactly, and v(x_k) → ẋ_k as h → 0.
- The KDE satisfies the continuity equation with v_NW.
- Batched slices agree with slice-by-slice results; Silverman; sampling weights and dtype.

Choices: weights `w` are passed as probabilities, not log-weights, and the log is taken internally. The default is the full score (`loo=False`), as in the plan's interface. The earlier prototype defaulted to LOO.
Open: none.

## Step 3 - The force g (2026-10-26)
Built: `wgf/energy.py` with `interaction_force(X, w, W_net)`, which symmetrises $W_s=\tfrac12(W(r)+W(-r))$ and differentiates it by autograd with `create_graph=True`, and `force(X, h, w, grad_V, c_H, c_V=1, c_W=0, W_net=None, loo=False)` (Eqs. 7, 14).
Tests: `tests/test_energy.py`, 6 passing.
- Quadratic W gives exactly $x_i-\sum_j w_jx_j$.
- The non-even $W=r_1+r_1^2$ reduces to $W_s=r_1^2$, with zero self-force.
- With V = x²/2 and c_H = β = 0.3, the slope matches $1-\beta/(s^2+h^2)=0.483$ to within 0.03; the unsmoothed value would be 0.388.
- `gradcheck` w.r.t. X passes (double backprop through ∇V and ∇W), with full and LOO scores.
- Batched slices equal slice-by-slice results, also under `no_grad`.

Choices: `force` accepts one slice `(N, D)` or all slices `(K, N, D)` at once. Each slice only sees its own particles. The test uses N = 5000, not 20,000: the full score is O(N²) in memory.
Open: none.

## Step 4 - The residual (2026-10-26)
Built: `wgf/residual.py` with `residual(X, t_grid, w, force_fn=None, scheme="trapezoid", g=None)` (Eq. 13 / App. C). `wgf/potentials.gaussian_flow_path` builds the exact (and perturbed) Gaussian-flow paths with their force. `experiments/validation/component_checks/residual.py` produces `experiments/validation/component_checks/figures/residual.png`: R_vel vs Δt on log–log axes, plus the SDE-path demonstration.
Tests: `tests/test_residual.py`, 7 passing.
1. Exact path, exact g:
   - Euler, K = 21 → 41: 5.92e-3 → 1.45e-3 (ratio 4.1; plan 6.1e-3 → 1.5e-3).
   - Trapezoid: 3.97e-6 → 2.52e-7 (ratio 15.8; plan 4.1e-6 → 2.6e-7).
2. Perturbed path (r = 0.5): 0.8648 at K = 41 and 0.8647 at K = 161, against the limit 0.8647.
3. KDE force (c_H = β, h = 0.3, N = 2000, seeds 0–2): exact path < perturbed/10, both inside the plan's ranges.
4. EDE identity: F(T) − F(0) = −2.0116 from the closed form. Quadrature of ‖g‖²_ρ matches to 1e-6, and the particle estimate matches to within 3%.
5. Demonstration on SDE sample paths: R_vel = 8.9, 18.9, 38.9, 78.6, 158.3 for K = 11…161, close to 2βD(K−1) = 10, 20, 40, 80, 160. The plan quoted 21.3 at K = 21; we use the exact density force on the paths, which may explain the small gap.

Choices: `residual` also accepts precomputed forces `g` for all slices, so the model can evaluate the force batched over time.
Open: none.

## Step 5 - First model: learned X and one parameter a (2026-10-26)
Built:
- `wgf/init.py`: `ot_straight_lines`, which does Hungarian OT between the first and last snapshots, then straight lines plus a per-particle offset (1% of the range, the same at every time), seeded.
- `wgf/model.py`: `QuadraticV` (a, init 0), `Stitching` (X, h via softplus, optional learned c_H, c_V = 1, optional W), `centres_at` (exact on grid nodes), and `moments_at` (KDE mean and variance = particle variance + h²).
- `wgf/train.py`: `nll` (per snapshot, or `pooled=True`), `loss`, and `fit` (Adam, cosine decay to 0.05×, full batch).
- `wgf/energy.grad_x`, and `wgf/datasets.py` (`quadratic_snapshots`, `double_well_snapshots`).
- `experiments/validation/ou_quadratic/fit_quadratic.py`, which produces `experiments/validation/ou_quadratic/figures/fit_quadratic.png`.

Tests: `tests/test_model.py`, 9 passing, including the plan's done-when targets on a 2000-epoch fit (~10 s).
Results (seed 0, 2000 epochs, K = 41, N = 100, trapezoid, λ = 1, c_H = 0.5 fixed):

| run | a | c_H | mean at t = 1 | var at t = 1 |
|---|---|---|---|---|
| main | **0.923** | 0.5 | **0.776** | **0.4662** |
| λ = 0 | **0.0000** (exactly) | 0.5 | 1.220 | 0.341 |
| learn c_H | 0.939 | 0.536 | 0.770 | 0.478 |
| Euler | 0.907 | 0.5 | 0.774 | 0.470 |

Targets: |a − 1| < 0.1, mean within 0.05 of 0.7358, variance within 10% of 0.4662. All are met; the mean is off by 0.041. The held-out data cloud itself has mean 0.726 and variance 0.412 (1000 cells).
Choices:
- `wgf/datasets.py` is a module not in the plan's layout. It keeps the data recipe identical in tests and demos.
- `grad_x` lives in `energy.py` so steps 5 and 6 share it.
- With N = 100 ≤ 1000 OT pairs, particles are drawn without replacement.

Open:
- a is still rising at epoch 2000 (0.92), so more epochs would probably bring it closer to 1.
- The held-out mean sits 0.04–0.05 above the truth in all runs, within the target but close to its edge.

## Step 6 - Swap in the network V_theta (2026-10-26)
Built: `wgf/nn.py` `MLP` (D → 64 → 64 → 1, SiLU, residual skip, zero-initialised head without bias). `wgf/metrics.py` gains `w2_1d`. Scripts: `experiments/validation/ou_quadratic/fit_nn.py` (6a), which produces `experiments/validation/ou_quadratic/figures/fit_nn.png`, and `scripts/demo_double_well.py` (6b; removed on 2026-10-28, still at commit c1d8680), whose figure is kept as `experiments/validation/double_well/results/legacy_2026-10-26/demo_double_well.png`.
Tests: `tests/test_nn.py`, 4 passing: zero init and the skip, double backprop through `grad_x`, metrics, and the 6a targets. The full suite is **65 passing** (~50 s).

**6a** (same data as step 5, V_a → V_θ, 2000 epochs, 20 s): potential **R² = 0.992**; held-out t = 1 mean 0.771 (target 0.7358 ± 0.05) and variance 0.473 (0.4662 ± 10%). All targets are met.

**6b** (double-well; seed 0; 2000 epochs; N = 100, K = 50; c_H learned from 1.0; full score). X0 ~ N(0, 1) is our choice; it is recorded in `wgf/datasets.double_well_snapshots`. 1000 train and 1000 test cells come from one paired run with 20 snapshots on [0, 4].

| regime | R²(V) | c_H | h | W2 mean (20 times) | W2 at unseen times | learned barrier (true 0) |
|---|---|---|---|---|---|---|
| dense | 0.850 | 0.920 | 0.505 | 1.177 | – | +0.48 |
| terminal | 0.753 | 0.598 | 0.307 | 1.769 | 1.869 | −0.06 |
| trickle (10 cells) | −1.174 | 0.913 | 0.751 | 2.025 | – | −0.22 |

Choices:
- c_H is learned from 1.0, as in the paper preset (step 5 kept it fixed).
- The score is the full score, the plan's default. The earlier prototype used LOO with 6000 epochs and got R² 0.96.
- W2 compares 1000 KDE samples with the 1000 test cells.

Open (for review; no thresholds set):
- W2 ≈ 1.2–2.0 is large. With wells 14 apart, it is dominated by how the model's 100 particles split between the wells (each 1% of mass ≈ 0.14 in W1). The barrier again sits off 0.
- The trickle regime fails on V (R² < 0). 10 cells per snapshot looks too few for 2000 epochs.
- Next candidates: more epochs (a was still rising in step 5), LOO vs full score, several seeds.

## Check after step 6 - forward rollout and force recovery (2026-10-26)
Built: `scripts/demo_rollout.py` (now `experiments/validation/ou_quadratic/rollout.py`, 6a only; the 6b cases are covered by `experiments/validation/double_well/run.py`), whose figure is kept as `experiments/validation/double_well/results/legacy_2026-10-26/demo_rollout.png`. Each step-6 model is retrained (same seeds), then V_θ and c_H are frozen.

**Test 1, forward simulation.** The step-1 simulator runs dX = −c_V∇V_θ dt + √(2c_H) dB from the observed t = 0 cells (6a: the 1000 training cells; 6b: the 1000 test cells). At every snapshot time, the result is compared with held-out cells by exact 1-D W1. The rollout never uses the learned paths X.
- Floor: the true SDE (V\*, β), from the same start with independent noise.
- Also listed: the stitching KDE's own W1.

| run | W1 learned SDE | W1 true SDE (floor) | W1 stitching KDE | right-well mass at t = 4, rollout / data |
|---|---|---|---|---|
| 6a quadratic | 0.044 | 0.032 | 0.031 | – |
| 6b dense | 1.541 | 0.132 | 0.340 | 0.339 / 0.492 |
| 6b terminal | 1.284 | 0.132 | 1.444 | 0.490 / 0.492 |
| 6b trickle | 2.613 | 0.132 | 0.845 | 0.605 / 0.492 |

(W1 values are means over the snapshot times.) At the held-out time t = 1, 6a's rollout scores W1 = 0.034, below the floor's 0.042.

**Test 2, force recovery.** The relative L2 error ‖c_V∇V_θ − ∇V\*‖ / ‖∇V\*‖ is measured on all test cells and on a uniform grid (6a: the data range; 6b: [−10, 10]). Gradients have no free constant, so nothing is centred. ∇V\* of the double-well is x − 7·sign(x), which jumps from −7 to +7 at x = 0, so no smooth network can match it near the barrier.

| run | error on cells | error on grid | c_H (true β) |
|---|---|---|---|
| 6a quadratic | 0.122 | 0.181 | 0.500 (0.5, fixed) |
| 6b dense | 0.464 | 0.513 | 0.920 (1.0) |
| 6b terminal | 0.631 | 0.683 | 0.598 (1.0) |
| 6b trickle | 1.979 | 1.945 | 0.913 (1.0) |

Reading:
- 6a passes both tests. Its rollout is at the noise floor, and its force error is 12%, larger than R² = 0.992 suggests.
- In 6b, the learned SDE generalises much worse than the stitching KDE fits. In dense, the rollout puts only 34% of the mass in the right well against 49% in the data, consistent with the barrier at +0.48.
- In terminal, the well split is right, but the transit is wrong in between: W1 peaks at 2.15 near t = 1.5.
- The force errors of 46–200% in 6b show that V_θ is not yet recovered there.

## Held-out protocol for the double-well (2026-10-26)
Built:
- `wgf/datasets.py`: `SPLITS` (`interleaved` 10/10 is the default; `midpoint` 11/9; `coarse` every 4th + 19), `split_snapshots` (splits by time *and* by trajectory, with both endpoints always in train), and `double_well_regimes`. All three regimes use this one split and are scored on the same 10 test times and the same 1000 test cells.
- `wgf/baselines.py`: 1-D OT (quantile) interpolation of the two training snapshots on either side of a test time.
- `wgf/model.FixedV`.
- `scripts/demo_split.py` (now `experiments/validation/double_well/run.py`), whose figure is kept as `experiments/validation/double_well/results/legacy_2026-10-26/demo_split_interleaved.png`.

Tests: `tests/test_split.py`, 8 passing. The full suite is **73 passing**.

Protocol: paper presets only, and nothing is tuned on the test times.
- Ceiling: OT interpolation, which uses no physics.
- Floor: the same model with V = V\* and c_H = β fixed ("V\* KDE"), and the true SDE rolled out from the training cloud at t = 0 ("V\* SDE").

Results (seed 0, 2000 epochs). Values are means over the 10 test times.

W1:

| regime | OT interp | V_θ KDE | V_θ SDE | V\* KDE | V\* SDE |
|---|---|---|---|---|---|
| dense | **0.236** | 0.386 | 0.461 | 0.418 | 0.268 |
| terminal | 1.771 | 1.547 | 1.378 | 0.505 | 0.268 |
| trickle | 1.996 | 0.805 | 1.243 | 0.620 | 2.083 |

W2:

| regime | OT interp | V_θ KDE | V_θ SDE | V\* KDE | V\* SDE |
|---|---|---|---|---|---|
| dense | **0.897** | 1.191 | 1.325 | 1.201 | 0.963 |
| terminal | 1.947 | 1.846 | 1.566 | 1.293 | 0.963 |
| trickle | 2.990 | 2.016 | 1.618 | 1.437 | 3.944 |

V_θ's learned c_H and force error (relative L2 on the test cells), by regime:
- dense: c_H 0.81, error 0.67
- terminal: c_H 0.60, error 0.61
- trickle: c_H 0.63, error 1.61

Reading:
- **Dense:** OT interpolation wins, even against the model with V\* fixed (0.418). With short gaps, the test shows no learned dynamics. The stitching pipeline (N = 100 particles, the smoothing h) costs more than the potential: V_θ and V\* score the same here.
- **Terminal** (a long gap): V_θ beats interpolation (1.38–1.55 vs 1.77) but stays far from the V\* floor (0.27–0.51). Physics helps, but V_θ captures only part of it.
- **Trickle:** the V_θ KDE (0.81) beats interpolation (2.00) and comes close to the V\* KDE (0.62). V\* SDE is poor here because it starts from only 10 cells, so the true-SDE floor is not informative for trickle.
- W2 is dominated by how mass splits between the wells. Even the true SDE gives 0.96 on dense, so W1 is the clearer metric here.

Next: run the `coarse` split, which tests the long gaps where physics should matter. Then use several seeds.

## Run records; separate train / eval / plot (2026-10-28)
Built: `wgf/runs.py`, the one place that saves and loads results. `experiments/validation/double_well/run.py` and `experiments/validation/ou_quadratic/rollout.py` (then `scripts/demo_split.py` and `scripts/demo_rollout.py`) use it through three commands:

```bash
.venv/bin/python experiments/validation/double_well/run.py train --split interleaved [--dataset dw-001] [--epochs 2000] [--loo] [--pooled] [--fix-c-H] ...
.venv/bin/python experiments/validation/double_well/run.py eval <run folder> [--eval-seed 0]
.venv/bin/python experiments/validation/double_well/run.py plot <run folder> [--eval <evals folder>]
.venv/bin/python scripts/show_run.py [<run folder>]
```

The same three commands exist for `ou_quadratic/rollout.py`. `train` makes a new folder, `<experiment>/runs/<YYYYmmdd-HHMMSS>_<name>/`, and never reuses one. `eval` loads the saved data and checkpoints and writes a new `evals/<timestamp>_seed<k>/` folder; it never trains. `plot` reads only `metrics.json` and the training histories; it never trains or evaluates. Missing files, a run of another experiment, an unfinished run, or data whose hash changed each stop with one `error: …` line (exit 1), and `eval` stops before it creates any folder. `show_run.py` lists the runs, or prints one run's whole configuration, provenance, models and evaluations.

What a run folder holds:
- `config.json` — data settings, split (name and train indices), regimes, N, K, the V_θ architecture, epochs, lr and schedule, λ, full/LOO score, c_H init and whether it is learned, the NLL convention (per snapshot vs pooled), and the data and init seeds with what each one drives. It also records the source commit, whether the tree was dirty, the Python, torch, numpy, scipy and matplotlib versions, the command line, the start and end time, the runtime, the status, and the sha256 of `data.pt`.
- `source.diff` — written when the tree was dirty: `git diff HEAD` plus untracked files.
- `data.pt` — every training and test set the run used. `eval` also regenerates them from the seed and records in `eval.json` whether the hash matches.
- `models/<tag>.pt` — the spec and `state_dict`, with V_θ, X, h, c_H, w, t_grid, and W_θ and c_W when present. The same values are also stored in natural units. `wgf.runs.load_model(path)` rebuilds the model in a fresh process and restores float64.
- `history/<tag>.json` — the return value of `fit()`, which now also records `lamR` (λR) and `lr` for every epoch, next to loss, NLL, R, h and c_H.
- `evals/…/eval.json` — eval settings and every eval seed (start resampling and KDE samples, the V_θ rollout, the V\* rollout).
- `evals/…/predictions.pt` — every predicted cloud.
- `evals/…/metrics.json` — W1 and W2 for each method at each test time, force errors, potential R², c_H and h. Every method is labelled with its kind (stitching KDE, SDE rollout, OT baseline) and its dynamics (learned, true, none).
- `evals/…/tables.md`, `*.png` — written by `plot`.

Choices:
- `runs/` is in `.gitignore`. The checkpoints and data stay local.
- Resuming training is not supported, so no optimiser, scheduler or RNG state is saved. `fit()` is full batch and draws no random numbers after initialisation, so the init seed and the config reproduce training exactly.
- `double_well_regimes` passes simulation settings through to `double_well_snapshots`. `build_nn_model` and `train_regime` take `loo`, and `train_regime` also takes the `fit` settings, so the config drives what runs.
- The other scripts write their figures to their experiment's `figures/` folder.

Tests: `tests/test_runs.py`, 9 passing. They cover rebuilding in a fresh process (MLP, LOO with fixed c_H, FixedV, with W), folder collisions and status, the data-hash check, the clear errors, the `lr`/`lamR` history, and an end-to-end `train → eval → plot` that checks eval and plot leave the trained files untouched.

## Experiments reorganised; saved double-well datasets (2026-10-28)
Layout: `experiments/paper_reproduction/` is for the paper's §5.1–5.5, none reproduced yet. `experiments/validation/` holds the additional validation on simulated data: `component_checks/` (steps 1 and 4), `ou_quadratic/` (steps 5 and 6a) and `double_well/`. Experiments are grouped by purpose rather than by data source, since the paper also uses simulated data. The shared code stays in `wgf/`, and `scripts/run_plots.py` became `wgf/report.py`. The learning algorithm is unchanged.

Built:
- `wgf/datastore.py`: saved datasets with IDs. `create` always takes a new `<prefix>-NNN`; `load` checks the sha256 of the samples and of the splits.
- `wgf/datasets.double_well_dataset`: one seeded simulation that also keeps 12 trajectories at every integration step, followed by every split drawn from the same generator.
- `experiments/validation/double_well/`:
  - `data.py` with `list`, `show` and `generate`.
  - `data/dw-001/`: seed 0, committed.
  - `run.py` (the old `demo_split.py`): it trains from a dataset ID and split, records the dataset and split IDs and hashes, and evaluates without regenerating data. It also saves force curves, KDE densities, test histograms, right-well fractions and the barrier, so `plot` can draw force and distribution figures from `metrics.json` alone.
  - `double_well.ipynb`, the notebook.
  - `README.md`.
- `ou_quadratic/rollout.py` keeps only the 6a case. `demo_double_well.py` was removed: its "all times" protocol is superseded by the held-out splits, the script is still at commit `c1d8680`, and its figure was moved to `double_well/results/legacy_2026-10-26/`.

Checks:
- `dw-001` equals `double_well_regimes(Generator(0), split)` bit for bit, for every split, regime and test set.
- A 2000-epoch `interleaved` run on `dw-001` reproduces the held-out tables above exactly: all W1/W2 means, c_H, and force errors.
- Short 200-epoch **diagnostic** runs of `interleaved`, `midpoint` and `coarse` all train, evaluate and plot. Their numbers are unconverged and labelled as diagnostics in the run config, the figures and `double_well/README.md`.
- Tests: `tests/test_datastore.py`, 8 passing. They cover equality with the old data, fine paths matching the snapshots, disjoint splits whose regimes use only training cells, training that is unchanged when every held-out sample is NaN, and `create` never overwriting while corrupt files are detected. `tests/test_runs.py` now runs the double-well `train → eval → plot` end to end on `dw-001`.

## Not done
- Step 7 (interaction network W_θ, CIS data): later, per the plan. `energy.interaction_force` is already in place and tested.
- `notebooks/00_stitching_from_scratch.ipynb` still calls the old `simulate(..., generator=)` API and will fail at its data cell until it is updated.
