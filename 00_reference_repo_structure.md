# 00 — Reference implementation: structure analysis

Repo: [BasisResearch/wasserstein-residuals](https://github.com/BasisResearch/wasserstein-residuals) (package name `stitching`), commit `70e098c` (2026-07-08), Apache-2.0.
Paper: Heinonen et al., *Wasserstein Residuals: Learning Gradient Flows from Population Dynamics*, arXiv:2607.04738.

**Clean-room rule for this project.** We read their repo only to learn its *structure* and *effective settings*. Our code is written from the paper and our own derivations (`derivations/01_physics_loss_restrictions.md`), with no copied code. We can run their repo as a black box to get reference numbers to check ours against. Datasets are data, not code: we may load them, citing the sources (the RNA set is third-party, see §5).

---

## 1. Layout at a glance

~13.4k lines of Python (docstrings included), built on JAX + Equinox (modules), Optax (Adam), Diffrax (SDE), POT (OT metrics). **The core math (KDE, velocity, residual, functional) is ~1.1k lines; adding the model, the MLP and the optimiser brings the method to ~1.9k.** The rest is experiment harness, plotting, persistence and tests.

```
stitching/                     ← library
  _kde/        kernel.py        IsotropicGaussian kernel, Silverman bandwidth          [MATH]
               distributions.py KernelDensity: log_prob, analytic score, NW velocity  [MATH]
               curves.py        InterpolationCurve: (K,N,D) centres → ρ_t at any t     [MATH]
  velocity.py                   ∇ δF/δρ closed form  +  NLL action                     [MATH / LOSS]
  residual.py                   discretised velocity residual R_vel (2 estimators)     [LOSS]
  models/      stitching.py     Stitching model (params, init) + StitchingLoss          [NN / LOSS]
               lightspeed.py    JKOnet* baseline model + loss                           [BASELINE]
               _functional.py   assembles F = c_H·H + c_V·V_θ + c_W·W_θ                 [NN]
  utils/       nn.py            MLP (SiLU, residual skips, zero-init head)              [NN]
               optim.py         Adam + cosine schedule, frozen-leaf filter, loop        [GD]
               ot.py            Hungarian matching (scipy)                              [INIT]
               solvers.py       one explicit Euler step (JKO "chord" predictor)         [EVAL]
               metrics.py       W1/W2, MMD, Bures-W2, UVPs, potential R², one-step      [EVAL]
               plotting, paper_plots, persistence, runners, cli, …     (harness, ~2.4k lines)
  synthetic/   potentials.py    15 2-D test potentials + 1-D double-well               [DATA]
               sde.py, simulate.py  Euler–Maruyama for the WGF SDE                      [DATA]
               dataset.py       observation protocols, coupled vs. uncoupled sims       [DATA]
  data.py, config/, build.py, train.py, evaluate.py   config → model → fit → metrics (harness)
configs/{data,method,experiment}/*.json   presets, merged in order: defaults < data < method < experiment
experiments/   one spec per paper experiment, stage engine (data/train/eval/plot)
data/          rna (5 PCs), cis, chiral (committed .npy/.npz)
tests/         velocity parity vs. autodiff, residual properties, config, regression
```

## 2. Paper ↔ code map

| Paper | Object | Where it lives | Notes |
|---|---|---|---|
| Eq. 6 | $\mathcal L=\lambda\mathcal R_{\rm vel}+\sum_t\mathcal D(\rho_t,q_t)$ | `StitchingLoss.__call__` | `total = w_nll·NLL + w_consistency·R`; $\lambda$ = `stitching_coeff` |
| Eq. 8 | $\rho^\theta_t=\sum_k w_k\,\phi_h(x-x_{t,k})$ | `InterpolationCurve` → `KernelDensity` + `IsotropicGaussian` | diagonal Gaussian, one bandwidth $h\in\mathbb R^D$ shared by all particles and times; off-grid $t$ → linear interpolation of centres |
| $\nabla\frac{\delta\mathcal F}{\delta\rho}$ | $c_H\nabla\log\rho+c_V\nabla V_\theta+c_W\sum_j w_j\nabla W_\theta(y-x_j)$ | `velocity.stitching_velocity` | entropy term = analytic KDE score; $V,W$ terms by autodiff |
| Eq. 9 / 35 | Nadaraya–Watson $v^\theta_t(x)$ | `KernelDensity.nadaraya_watson` | used only by the KDE-MC residual ($\alpha>0$) |
| Eq. 13 | $\sum_j\sum_k w_k\frac1{\Delta t_j}\lVert\Delta x_{jk}+\Delta t_j\,g_{jk}\rVert^2$ | `residual.wgf_residual` | code default is **trapezoidal**, not the forward Euler of Eq. 13 (see §9) |
| App. C, $\alpha$ | KDE-Monte-Carlo residual | `residual.wgf_kde_sampled_residual` | blended as $(1-\alpha)R_{\rm centres}+\alpha R_{\rm KDE}$ |
| Eq. 37 | leave-one-out score at centres | **not used** | `kde_score` keeps the self-term (see §9) |
| Eq. 26 | Lemma 1, the energy-dissipation (EDE) identity | theory only | — |
| Eqs. 27–28 | KL / NLL divergence | `velocity.nll` | the only $\mathcal D$ implemented; score-matching variants (Eqs. 30–34) are absent |
| App. D.1 | JKOnet* | `models/lightspeed.py` | in-repo reimplementation, named "Lightspeed" |
| data model | $dx=-\nabla\frac{\delta\mathcal F}{\delta\rho}dt+\sigma\,dW$ | `synthetic/simulate.py`, `sde.py` | see §9 on the $\sigma$ convention |

## 3. The three blocks we must mark: NN · LOSS · GD

### 3.1 [NN] What is learned (Stitching)

| Parameter | Shape | Constraint | Initial value | Trained? |
|---|---|---|---|---|
| `trajectories` $x_{t_j,k}$ | $(K,N,D)$ | none | OT-matched straight lines + jitter (§4) | yes |
| `t_grid` | $(K,)$ | — | `linspace(t_first_obs, t_last_obs, K)` | **no** (frozen by name) |
| `potential_net` $V_\theta$ | MLP $D(+1)\to64\to64\to1$ | — | output layer zeroed ⇒ $V_\theta\equiv0$ | yes |
| `interaction_net` $W_\theta$ | MLP $1\to64\to64\to1$ on $\lVert r\rVert^2$ ("radial") or $D\to\dots$ ("full") | — | zeroed head | yes, only built if `interaction_coeff>0` |
| `raw_bandwidth` → $h$ | $(D,)$ | softplus | Silverman on the $t_0$ snapshot, or a given scalar | yes |
| `raw_entropy_coeff` → $c_H$ | scalar | softplus | from data preset (0.03 / 0.05 / 1.0) | yes, unless `freeze_entropy` |
| `raw_potential_coeff` → $c_V$ | scalar | softplus | 1.0 (floored at 1e-4) | yes, unless `freeze_potential` |
| `raw_interaction_coeff` → $c_W$ | scalar | softplus | preset value (floored at 1e-4) | yes, unless `freeze_interaction` |
| `raw_weights` → $w_k$ | $(N,)$ | softmax | zeros ⇒ uniform $1/N$ | **no** by default (`freeze_particle_weights=True`) |

MLP block: `Linear → SiLU`, with a residual skip `y ← y + SiLU(Ly)` whenever the in and out widths match (so from the 2nd hidden layer on), then a final `Linear`. With hidden=(64,64) on $D=5$ that is ≈4.6k weights, while the trajectories hold $50\cdot100\cdot5=25$k numbers. **Most of the parameters are particle positions, not network weights.**
Time-conditioned variant: $V_\theta(x,t)$ takes `concat([x, t])` as input (used in the RNA "tcond" runs).

### 3.2 Forward pass (one loss evaluation)

```
data {(x_i,t_i)} ────────────────────────────────────────────┐
                                                             ▼
X[K,N,D] ──► ρ_t = KDE(centres X(t), bandwidth h, weights w) ──► NLL = −mean_i log ρ_{t_i}(x_i)   ──┐
   │                         │                                                                      │
   │                         └─► score s = ∇log ρ at each centre (closed form) ─┐                    │
   │                                                                            ▼                    ├─► L = w_nll·NLL + λ·R
   │                  V_θ, W_θ (autodiff) ─► g = c_H·s + c_V·∇V_θ + c_W·Σ_j w_j ∇W_θ(x−x_j)      │
   │                                                                            │                    │
   └─ ΔX_j = X[j+1] − X[j] ──► R = Σ_j (1/Δt_j) Σ_k w_k ‖ΔX_jk + Δt_j · ḡ_jk / κ‖² ◄──────────────┘
                                   ḡ = ½(g_j + g_{j+1})  (trapez; euler/implicit/midpoint also available)
```

### 3.3 [LOSS] The two terms

- **Data term.** $\mathrm{NLL}=-\frac1{M}\sum_{i=1}^M\log\rho^\theta_{t_i}(x_i)$, averaged over *all* observed points. $\rho^\theta_{t_i}$ is the KDE at the (interpolated) observation time.
- **Physics term, centres estimator ($\alpha=0$, the default).**
  $R=\sum_{j=0}^{K-2}\frac1{\Delta t_j}\sum_k w_k\big\lVert\Delta x_{jk}+\frac{\Delta t_j}{\kappa}\bar g_{jk}\big\rVert^2$, where $\kappa$ = `stitching_kinetic` (default 1).
  This equals $\sum_j\Delta t_j\sum_k w_k\lVert\dot x_{jk}+g_{jk}/\kappa\rVert^2$, i.e. the left-Riemann sum of $\int\!\lVert v+\nabla\frac{\delta\mathcal F}{\delta\rho}\rVert^2_\rho\,dt$ with $v(x_k)=\dot x_k$ (Claim 2).
  `norm="relative"` / `"relative-sg"` instead divides each term by $\lVert\Delta x\rVert^2+\lVert\text{force}\rVert^2$ and takes the mean over intervals. These options are not used in the paper presets.
- **Physics term, KDE-MC estimator ($\alpha=1$).** At each snapshot $j<K-1$ draw $y_k=x_{jk}+h\odot\varepsilon_k$, one sample per particle. Evaluate the NW velocity $v(y_k)$ built from all particles' $\dot x_{jk}=\Delta x_{jk}/\Delta t_j$, and $g(y_k)$ from the snapshot KDE. Then $R_{\rm KDE}=\sum_j\Delta t_j\sum_k w_k\lVert v(y_k)+g(y_k)/\kappa\rVert^2$. It needs a PRNG key per step and does not support a time-conditioned $V$.

### 3.4 [GD] Optimiser

- Adam (Optax defaults) on a cosine-decay learning rate: from `lr` down to `0.05·lr` over all steps.
- **Full batch, one step per epoch.** `fit()` passes `batch_size = #train points`, so `cfg.batch_size` is ignored.
- Trainable mask: every array leaf except `t_grid`, and `raw_weights` when frozen.
- One JIT warm-up step whose result is thrown away, then `epochs` steps. A fresh PRNG key every step (only the KDE-MC residual uses it).
- Gradients flow jointly into the particles (pulled toward the data by NLL, and toward a gradient flow by $R$), into $V_\theta,W_\theta$ (pulled to explain $\Delta x$), and into $h,c_H,c_V,c_W$.

## 4. Initialisation of the trajectories

1. Take the first and last observed snapshots and match them with Hungarian OT (squared-Euclidean cost, `scipy.optimize.linear_sum_assignment`).
2. Pick $N$ matched pairs at random, with replacement if there are fewer than $N$ pairs.
3. Interpolate each pair linearly across the $K$ grid times.
4. Add one Gaussian offset per particle (std = 1 % of the per-dimension data range). The same offset is applied at every time.

The intermediate snapshots play no part in the initialisation; they enter only through the NLL. With a single snapshot, the particles start constant in time.

## 5. Data & simulator

| Dataset | Raw shape | Times | Split | Origin |
|---|---|---|---|---|
| `rna` (EB scRNA, first 5 PCs, standardised) | 16,819 × 5 | labels 0–4 | random particle split with `train_fraction` 0.6, or timepoint split (gappy: train {0,2,4}, eval {1,3}) | **third-party**: Moon et al., *Nat. Biotech.* 2019; TrajectoryNet (Tong et al., ICML 2020) — cite both |
| `cis` | trajectories (201, 310, 2) | 0…200 | split by particle, 50/50 | McKean–Vlasov sim (double-well + Gaussian interaction) |
| `chiral` | 1,750 train + 1,750 test, 2-D | {0, 50, 100, 200, 400, 600, 800} | pre-split | non-gradient flow (rotated interaction force) |
| synthetic (live) | 2,000 simulated (1,000 train) | $T=5$ intervals, $\Delta t=0.01$ | split by particle, 50/50 | 15 JKOnet* potentials, $U[-4,4]^2$ init, $\beta=0$ |
| doublewell 1-D (live) | 1,000 | 20 snapshots on [0, 4] | dense / terminal / trickle (10 cells per snapshot) | $V=\tfrac12(\lvert x\rvert-7)^2$, $\beta=1$ |
| wavy_valley (live) | 80 simulated (40 train) | {0, 5, 10, 20, 30} | 50/50 | $V=0.6(x_2-\sin\frac{\pi x_1}{2})^2-0.5x_1$, init $\mathcal N([-3,1],0.1^2)$ |

- **Simulator:** Euler–Maruyama via Diffrax. The drift $-(c_V\nabla V+c_W\sum_j w_j\nabla W(x-x_j))$ is evaluated on the empirical cloud; the entropy term enters only as noise.
- **Coupled vs. uncoupled:** "paired" tracks the same particles across snapshots. "Unpaired" runs a fresh simulation from $t=0$ for every snapshot.
- **Observation protocols:** `dense`, `terminal` (first and last only), `irregular` (random interior subset), `trickle` (few cells per snapshot). They are applied to the training data only.

## 6. Evaluation

- **Sampling.** Stitching takes the grid node nearest each test time and draws `num_samples` points from that KDE (default = training-set size). This is repeated `eval_reps` times and averaged.
- **Distribution metrics per test time:** $W_1$ (EMD), $W_2$, $W_2^2$ (POT, exact), squared Bures–$W_2$ of Gaussian fits, an RBF MMD² (median-heuristic bandwidth × {¼, ½, 1, 2, 4}), and a DMSB-style MMD.
- **Synthetic-only metrics:**
  - potential $R^2$ / relative RMSE / pattern correlation of centred $c_V V_\theta$ vs. $V^*$, on 500 uniform points in the data bounding box ±0.5;
  - $L^2$-UVP on gradients (iJKOnet Eq. 19);
  - Bd²W₂-UVP on marginals (iJKOnet Eq. 20).
- **One-step-ahead.** Push the test particles at $t_k$ through one explicit Euler step of the learned $\mathcal F$ (KDE at bandwidth $h$), then compare with $t_{k+1}$ (JKOnet* protocol).

## 7. Baseline "Lightspeed" (JKOnet*)

- **Parameters:** $V_\theta$, and $c_H,c_V$ (softplus). The bandwidth is fixed at Silverman on $t_0$.
- **Pairs:** Hungarian OT between consecutive observed snapshots, each truncated to the smallest per-snapshot count (the first rows, not a random subset). The `natural` coupling uses particle identity instead.
- **Loss:** $\sum_j\operatorname{mean}_k\lVert\tau_j\nabla\frac{\delta\mathcal F}{\delta\rho}(y_k)+(y_k-x_k)\rVert^2$, with the gradient evaluated at the *target* cloud (the implicit JKO condition). There is no $1/\tau$ weight.
- **Prediction:** start from the observed $t_0$ cloud and take one explicit Euler step per interval. This straight "chord" step is the bias that the paper's Fig. 1 illustrates.

## 8. Effective hyperparameters per paper experiment (Stitching)

The common settings come from `configs/method/stitching.json`: lr 5e-3, Adam + cosine, `potential_hidden=(64,64)`, trapezoidal scheme, `norm="raw"`, $\kappa=1$, bandwidth from Silverman, weights frozen. These override the dataclass defaults, which are (64,64,64), N=50, K=20.

| Experiment | N | K | epochs | $\lambda$ | $\alpha$ | $c_H$ init | $c_V$ init | $c_W$ | bandwidth init | other |
|---|---|---|---|---|---|---|---|---|---|---|
| §5.1 wavy valley (Fig. 1) | 20 | 50 | 10,000 | **0.05** | **0.5** | 0.05 | 1 | – | **0.5** | data: 40 train cells, irregular times, $\beta=0.00625$ |
| §5.2 synthetic (Tab. 1/4) | **1000** | 50 | 2,000 | 1 | 0 | 1.0 | 1 | – | Silverman | 15 potentials × {paired, unpaired} |
| §5.3 RNA (Fig. 3, Tab. 2/5) | 100 | 50 | 10,000 | 1 | 0 | 0.03 | 1 | – | Silverman | static vs. time-conditioned $V$; gappy variant; paper averages 5 seeds |
| §5.4 CIS (Fig. 4, Tab. 3) | 100 | 50 | 1,000 | 1 | 0 | 0.03 | 1 | 1 (radial) | Silverman | — |
| §5.5 chiral (Fig. 5) | 100 | 50 | 1,000 | 1 | 0 | 0.03 | ≈1e-4 | 1 (radial) | Silverman | "V-free", but see §9.4 |
| double-well panel | 100 | 50 | 2,000 | 1 | 0 | 1.0 | 1 | – | Silverman | dense / terminal / trickle |

## 9. Things to watch: paper ↔ code differences and pitfalls

Each item was checked against the source.

1. **Score at the centres keeps the self-term.** The code's `kde_score` evaluates $\nabla\log\rho^\theta$ at $x_{t,j}$ including $k=j$. That is the exact score of the model density. Paper Eq. 37 prescribes leave-one-out.
   Our quick check on Gaussian clouds: the full score has about half the magnitude of the LOO score at $N=100$, $D=5$ (the RNA setting), and the two agree at $N=1000$, $D=2$. Neither wins everywhere: the full score is closer to the kernel-smoothed density's score, the LOO closer to the raw density's.
   → For us: implement both and treat it as an ablation.
2. **Time scheme.** Eq. 13 in the main text is forward Euler. The code default is trapezoidal, which matches App. C.
3. **SDE noise convention.** `sde.diffusion` returns $\sigma=\sqrt2\,c_H$, whereas Fokker–Planck with $\beta\Delta\rho$ needs $\sigma=\sqrt{2\beta}$. The two agree only for $\beta\in\{0,1\}$. Their own `SyntheticSystem` docstring and the McKean–Vlasov loader assume $\sqrt{2\beta}$.
   - Unaffected: double-well ($\beta=1$) and the 2-D synthetic panel ($\beta=0$).
   - Wavy valley ($\beta=0.00625$): this gives $\sigma\approx0.0088$ instead of $\approx0.112$.
   → For us: use $\sigma=\sqrt{2\beta}$. If we want to match their Fig. 1 data, note the difference explicitly.
4. **Coefficients are learned.** $h,c_H,c_V,c_W$ are all trainable unless frozen, so the "V-free" chiral run only *initialises* $c_V\approx10^{-4}$. Also $c_V\cdot V_\theta$ has a scale redundancy.
5. **Evaluation inconsistency.** The potential $R^2$ uses $c_V V_\theta$, but the $L^2$-UVP uses $\nabla V_\theta$ *without* $c_V$ (`evaluate.run_synthetic_evaluation`).
6. **NLL weighting.** NLL is a mean over all points, so each snapshot is weighted by its cell count. The paper writes $\sum_t\mathcal D(\rho_t,q_t)$.
7. **Init ignores the interior snapshots** (§4). This matters for the terminal and gappy protocols.
8. **RNA split.** With the default `train_fraction=0.6` the loader looks for `split_60_40_seed0.npz`, which does not exist, so it draws a random 60/40 split. The committed `split_70_30_seed0.npz` is never used.
9. **Full batch only.** `batch_size` is ignored in `fit`.
10. **KDE-MC residual** ($\alpha>0$) raises `NotImplementedError` for a time-conditioned $V$.

## 10. Blueprint for our clean-room version

**Notebook series.** Every notebook uses the same banner cells, each preceded by a markdown cell that states the formula being implemented:
`# ═══ [DATA] ═══` · `# ═══ [NN] ═══` · `# ═══ [LOSS] ═══` · `# ═══ [OPTIMISER / GD] ═══` · `# ═══ [EVAL] ═══`

| # | Notebook | Content | Built-in check |
|---|---|---|---|
| 00 | toy Gaussian flow | the 1-D Gaussian example from `derivations/01` with a closed-form flow | residual = 0 on the exact flow, > 0 when perturbed; $\dot{\mathcal F}=-\lVert g\rVert^2_\rho$ |
| 01 | KDE, score, NW velocity | Eq. 8, analytic score (full and LOO), Eq. 9 | analytic score = autograd score; NW velocity → $\dot x_k$ as $h\to0$ |
| 02 | energy & Wasserstein gradient | $V_\theta$ / $W_\theta$ MLPs, entropy, $\nabla\frac{\delta\mathcal F}{\delta\rho}$ | autograd of the first variation $\Phi(y)$ |
| 03 | residual loss | Euler / trapezoid / midpoint, centres and KDE-MC estimators | vanishes on an exact Euler flow |
| 04 | Stitching on the 1-D double-well | init, parameters, NLL + $\lambda R$, Adam loop | dense / terminal / trickle regimes |
| 05 | wavy valley vs. our JKOnet* | Fig. 1 | chord bias visible |
| 06 | synthetic potentials + metrics | Table 1/4 style | potential $R^2$, UVPs |
| 07 | RNA (static / time-conditioned, gappy) | Fig. 3, Tables 2 and 5 | $W_2$ at held-out times |
| 08+ | CIS interaction, chiral | §5.4–5.5 | — |

**Shared module.** Once the notebooks settle, move the code into a small package: `wgf/{kde,energy,residual,model,train,sim,metrics}.py`.

**Choices we make deliberately:**
- LOO vs. full score, as an ablation (§9.1);
- $\sigma=\sqrt{2\beta}$ in the simulator (§9.3);
- per-snapshot NLL weighting, $\sum_t$ as in the paper, vs. the pooled mean (§9.6);
- the $c_V$ factor applied consistently in the metrics (§9.5).

**Open decision:** framework. PyTorch keeps us clearly separate from their JAX code and is easy to read. Gradients $\nabla_x V_\theta$ inside the loss need `torch.autograd.grad(..., create_graph=True)`. JAX would make numerical parity with their outputs easier.
