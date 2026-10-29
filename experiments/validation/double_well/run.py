"""Double-well validation (additional to the paper): held-out protocol on a saved simulated dataset.

Data: a saved dataset (data.py; dw-001 by default), never simulated here. One split names the training
and test observation times and the training and test trajectories (disjoint); three regimes say what
the model sees of the training part:
  dense    - every training time, all training trajectories
  terminal - first and last training time only
  trickle  - every training time, 10 training trajectories (a different subset per time)
All regimes are scored on the same test times and test trajectories. Nothing is tuned on held-out data.

At each test time, against the test cells (W1 and W2, exact 1-D):
  OT interp   - quantile average of the two bracketing training snapshots: no physics, the ceiling
  V_theta     - stitching KDE of the learned model, and a forward rollout of its SDE
                (c_V grad V_theta, c_H) from the training cloud at t = 0
  V*          - the same stitching model with V fixed to the truth and c_H = beta (learns only X, h),
                and the true SDE rolled out from the same training cloud: the floor
Forces are compared as c_V grad V_theta against grad V* on the test cells and on a grid. Stochastic
paths are never compared pointwise with the stitching paths: only distributions and forces are.

Commands, from the project root (run folders: wgf/runs.py; datasets: wgf/datastore.py):
  .venv/bin/python experiments/validation/double_well/run.py train  --split interleaved [--dataset dw-001] [--epochs 2000] [--checkpoint-every 100] ...
  .venv/bin/python experiments/validation/double_well/run.py resume <run folder>
  .venv/bin/python experiments/validation/double_well/run.py eval   <run folder> [--eval-seed 0] [--verify-data]
  .venv/bin/python experiments/validation/double_well/run.py plot   <run folder> [--eval <evals folder>]
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
from data import DEFAULT_DATASET, load, verify_dataset  # noqa: E402
from wgf import potentials as P  # noqa: E402
from wgf.baselines import ot_interpolate_snapshots  # noqa: E402
from wgf.datasets import SPLITS  # noqa: E402
from wgf.init import ot_straight_lines  # noqa: E402
from wgf.kde import log_density, sample, silverman  # noqa: E402
from wgf.metrics import potential_r2, w1_1d, w2_1d  # noqa: E402
from wgf.model import FixedV, Stitching  # noqa: E402
from wgf.nn import MLP  # noqa: E402
from wgf.report import history_figure, md_table, mean, method_notes, write_tables  # noqa: E402
from wgf.runs import Run, RunMismatch, cli  # noqa: E402
from wgf.sim import simulate  # noqa: E402
from wgf.train import fit  # noqa: E402

torch.set_default_dtype(torch.float64)
EXPERIMENT = "double_well_validation"
CATEGORY = "additional validation on simulated data (not a paper experiment)"
REGIMES = ("dense", "terminal", "trickle")
PLAN = [(f"{n}_{s}", n, s == "V_star") for n in REGIMES for s in ("V_theta", "V_star")]  # training order
# every column of the comparison: how it predicts, and with which dynamics
METHODS = {
    "ot_interp": dict(label="OT interp", kind="OT baseline", dynamics="none",
                      description="1-D OT (quantile) interpolation of the two training snapshots that bracket t"),
    "theta_kde": dict(label="V_theta KDE", kind="stitching KDE", dynamics="learned",
                      description="samples of the learned model's KDE at t (its fitted particle paths X)"),
    "theta_sde": dict(label="V_theta SDE", kind="SDE rollout", dynamics="learned",
                      description="independent rollout of dX = -c_V grad V_theta dt + sqrt(2 c_H) dB from the "
                                  "training cloud at t0; never uses the paths X"),
    "true_kde": dict(label="V* KDE", kind="stitching KDE", dynamics="true",
                     description="true-dynamics reference: stitching model with V = V*, c_H = beta fixed "
                                 "(learns X and h only), KDE samples"),
    "true_sde": dict(label="V* SDE", kind="SDE rollout", dynamics="true",
                     description="true-dynamics reference: the true SDE rolled out from the same start"),
}
STYLES = {"ot_interp": "k:", "theta_kde": "o-", "theta_sde": "s-", "true_kde": "^--", "true_sde": "k--"}
FORCE_GRID = (-10.0, 10.0, 401)
DENSITY_GRID = (-12.0, 12.0, 241)
HIST_BINS = (-12.0, 12.0, 61)


# ── train ───────────────────────────────────────────────────────────────────────────────────────


def add_train_args(p):
    p.add_argument("--dataset", default=DEFAULT_DATASET, help="saved dataset ID (data.py list)")
    p.add_argument("--split", required=True, choices=list(SPLITS))
    p.add_argument("--epochs", type=int, default=2000)
    p.add_argument("--init-seed", type=int, default=0, help="model initialisation seed")
    p.add_argument("--N", type=int, default=100)
    p.add_argument("--K", type=int, default=50)
    p.add_argument("--lr", type=float, default=5e-3)
    p.add_argument("--lam", type=float, default=1.0)
    p.add_argument("--loo", action="store_true", help="leave-one-out score (default: full score)")
    p.add_argument("--pooled", action="store_true", help="pooled NLL (default: per-snapshot mean, summed)")
    p.add_argument("--c-H-init", type=float, default=1.0)
    p.add_argument("--fix-c-H", action="store_true", help="keep c_H at its initial value")
    p.add_argument("--purpose", default="diagnostic check (short run; not a paper benchmark result)")


def make_config(a, ds):
    split = ds.split(a.split)
    regimes = {}
    for r in REGIMES:
        idx, ids = ds.regime_cells(a.split, r)
        regimes[r] = {"time_idx": idx, "cells_per_time": sorted(set(map(len, ids)))}
    beta = ds.meta["settings"]["diffusion"]["beta"]
    return {
        "category": CATEGORY, "purpose": a.purpose,
        "dataset": {"id": ds.id, "root": str(ds.path.parent.relative_to(ROOT)), "split": a.split,
                    "split_id": ds.split_id(a.split), "samples_sha256": ds.meta["samples_sha256"],
                    "splits_sha256": ds.meta["splits_sha256"],
                    "observation_times": ds.times.tolist(),
                    "train_time_idx": split["train_time_idx"], "val_time_idx": split["val_time_idx"],
                    "test_time_idx": split["test_time_idx"],
                    "trajectories": {"train": len(split["train_traj_ids"]), "val": len(split["val_traj_ids"]),
                                     "test": len(split["test_traj_ids"])},
                    "regimes": regimes},
        "data": ds.meta["settings"],
        "model": {"N": a.N, "K": a.K,
                  "t_grid": "internal grid, linspace(first, last training time, K); not the observation times",
                  "X_init": "wgf.init.ot_straight_lines(first, last training snapshot), jitter 0.01",
                  "h_init": "silverman(first training snapshot); learned via softplus", "weights": "uniform 1/N",
                  "score": "loo" if a.loo else "full",
                  "V_theta": {"type": "mlp", "d_in": 1, "hidden": [64, 64], "activation": "silu",
                              "residual_skip": True, "head": "linear, zero init, no bias"},
                  "c_H": {"init": a.c_H_init, "learned": not a.fix_c_H}, "c_V": 1.0, "W": None},
        "reference_model": {"V": "wgf.potentials.double_well_1d (true, fixed)",
                            "c_H": {"init": beta, "learned": False}, "otherwise": "as model"},
        "train": {"optimizer": "Adam, full batch, one step per epoch", "epochs": a.epochs, "lr": a.lr,
                  "lr_schedule": "cosine to 0.05 lr", "lam": a.lam, "residual_scheme": "trapezoid",
                  "nll": {"pooled": a.pooled,
                          "convention": "one mean over all cells" if a.pooled
                          else "mean NLL per snapshot, summed over snapshots"},
                  "checkpoint_every": a.checkpoint_every,
                  "models": [tag for tag, _, _ in PLAN]},
        "seeds": {"data_generation": ds.meta["seeds"]["data_generation"], "init": a.init_seed,
                  "init_uses": "Generator(init) for ot_straight_lines, then torch.manual_seed(init) "
                               "before the network is built; fit() draws no random numbers",
                  "evaluation": "set per evaluation (eval --eval-seed), recorded in eval.json"},
        "dtype": "float64",
    }


def build_model(train, cfg, true_V=False):
    """A fresh model for `train` only: the paths start from its first and last snapshots, the bandwidth
    from its first snapshot; no held-out cell is passed in. Seeded, so it is rebuilt identically."""
    m, seed = cfg["model"], cfg["seeds"]["init"]
    times = sorted(train)
    t_grid = torch.linspace(times[0], times[-1], m["K"])
    X = ot_straight_lines(train[times[0]], train[times[-1]], t_grid, m["N"], torch.Generator().manual_seed(seed))
    torch.manual_seed(seed)
    loo = m["score"] == "loo"
    if true_V:
        model = Stitching(t_grid, X, silverman(train[times[0]]), FixedV(P.double_well_1d),
                          c_H=cfg["data"]["diffusion"]["beta"], loo=loo)
    else:
        V = MLP(m["V_theta"]["d_in"], tuple(m["V_theta"]["hidden"]))
        model = Stitching(t_grid, X, silverman(train[times[0]]), V, c_H=m["c_H"]["init"],
                          learn_c_H=m["c_H"]["learned"], loo=loo)
    return model


def fit_settings(cfg):
    tr = cfg["train"]
    return dict(lr=tr["lr"], lam=tr["lam"], scheme=tr["residual_scheme"], pooled=tr["nll"]["pooled"])


def train_model(train, cfg, true_V=False):
    """Build and fit one model in memory (no run folder, no checkpoints)."""
    model = build_model(train, cfg, true_V)
    return model, fit(model, train, epochs=cfg["train"]["epochs"], log_every=0, **fit_settings(cfg))


def train_all(run, ds, budget=None):
    """Every model of the run in PLAN order: finished ones are kept, a partial one continues from its
    checkpoint, the rest start from their seeded initialisation. budget: epochs before stopping."""
    cfg = run.config
    tr, split = cfg["train"], cfg["dataset"]["split"]
    for tag, regime, true_V in PLAN:
        observed = ds.train(split, regime)
        model, hist, used = run.train_or_resume(tag, lambda: build_model(observed, cfg, true_V), observed,
                                                tr["epochs"], fit_settings(cfg), tr.get("checkpoint_every", 100),
                                                budget)
        budget = None if budget is None else budget - used
        note = "kept (finished earlier)" if used == 0 else f"{hist['seconds']:5.0f} s"
        if "resumed_at_epoch" in hist and used:
            note += f", resumed at epoch {hist['resumed_at_epoch'][-1]}"
        print(f"{tag:17} {note}   c_H {model.c_H.item():.3f}   h {model.h.item():.3f}", flush=True)


def train(args):
    ds = load(args.dataset)
    cfg = make_config(args, ds)
    tag = "diagnostic" if cfg["purpose"].startswith("diagnostic") else "run"
    with Run.create(args.name or f"{tag}-{ds.id}-{args.split}", EXPERIMENT, cfg, args.runs_dir) as run:
        print(f"run {run.path}\ndataset {cfg['dataset']['split_id']}; {cfg['purpose']}")
        train_all(run, ds, args.stop_after)
    return run


def resume(args):
    """Continue an interrupted run: loads its saved dataset (hash-checked) and its checkpoints."""
    run = Run.open(args.run, EXPERIMENT, finished=False)
    if run.manifest["status"] == "finished":
        print(f"{run.path} finished training; nothing to resume")
        return run
    ds, _ = dataset_of(run)
    with run.begin_resume():
        print(f"resuming {run.path}\ndataset {run.config['dataset']['split_id']}")
        train_all(run, ds, args.stop_after)
    return run


# ── eval ────────────────────────────────────────────────────────────────────────────────────────


def dataset_of(run):
    """The run's saved dataset, checked against the hashes recorded at training time."""
    d = run.config["dataset"]
    ds = load(d["id"])
    for k in ("samples_sha256", "splits_sha256"):
        if ds.meta[k] != d[k]:
            raise RunMismatch(f"dataset {d['id']} changed since {run.path.name} was trained ({k} differs)")
    return ds, d["split"]


@torch.no_grad()
def predictions(train, test, m_theta, m_true, beta, seed=0):
    """{method: {t: (M, 1) predicted cloud}} at every test time."""
    times, M = sorted(test), len(next(iter(test.values())))
    g = torch.Generator().manual_seed(seed + 10)
    t0 = min(train)
    X0 = train[t0][torch.randint(len(train[t0]), (M,), generator=g)] if len(train[t0]) != M else train[t0]
    learned = lambda x: m_theta.c_V * m_theta.grad_V(x)
    roll_theta = dict(simulate(X0, [t0] + times, 1e-2, float(m_theta.c_H), learned,
                               gen=torch.Generator().manual_seed(seed + 1)))
    roll_true = dict(simulate(X0, [t0] + times, 1e-2, beta, P.grad_double_well_1d,
                              gen=torch.Generator().manual_seed(seed + 2)))
    kde = lambda m, t: sample(m.centres_at(t), m.h, m.w, M, g)
    return {
        "ot_interp": {t: ot_interpolate_snapshots(train, t, M) for t in times},
        "theta_kde": {t: kde(m_theta, t) for t in times},
        "theta_sde": {t: roll_theta[t] for t in times},
        "true_kde": {t: kde(m_true, t) for t in times},
        "true_sde": {t: roll_true[t] for t in times},
    }


@torch.no_grad()
def kde_density(model, t, xs):
    return log_density(xs, model.centres_at(t), model.h, model.w).exp().tolist()


def evaluate(args):
    run = Run.open(args.run, EXPERIMENT)
    cfg, s = run.config, args.eval_seed
    ds, split = dataset_of(run)
    verification = verify_dataset(ds) if args.verify_data else None
    if verification is not None and not verification["match"]:
        raise RunMismatch("--verify-data: regenerating the dataset from its recorded seed and settings does not "
                          f"reproduce the saved {ds.id}:\n  " + "\n  ".join(verification["differences"])
                          + "\nThe saved data (which the models were trained on) are unchanged; no evaluation was run.")
    regimes = {n: ds.train(split, n) for n in REGIMES}
    test = ds.held_out(split, "test")
    # load every checkpoint first, so a missing one fails before an evals/ folder is made
    models = {n: (run.load_model(f"{n}_V_theta"), run.load_model(f"{n}_V_star")) for n in REGIMES}
    ecfg = {"dataset": cfg["dataset"]["split_id"],
            "data_check": verification or {"regenerated": False,
                                           "checked": "saved samples and splits against their sha256 hashes"},
            "rollout": {"dt": 1e-2, "scheme": "euler",
                        "start": "training cloud at t0, resampled with replacement to the test-cell count "
                                 "when the sizes differ"},
            "kde_samples": "wgf.kde.sample (systematic resampling), one per test cell",
            "metrics": {"W1": "exact 1-D, predicted cloud vs test cells", "W2": "exact 1-D",
                        "force_rel_l2": "||c_V grad V_theta - grad V*|| / ||grad V*|| on all test cells and on "
                                        f"linspace{FORCE_GRID}",
                        "potential_r2_test_cells": "centred R^2 of c_V V_theta vs V* on all test cells",
                        "right_well_fraction": "share of x > 0 per test time, for every method and the test cells",
                        "barrier": "x in [-3, 3] where c_V grad V_theta changes sign from + to - (true: 0)",
                        "curves": "force on the force grid; KDE densities and test-cell histograms per test time"},
            "seeds": {"eval": s, "start_resample_and_kde_samples": s + 10, "rollout_V_theta": s + 1,
                      "rollout_V_star": s + 2},
            "methods": METHODS}
    times = sorted(test)
    Yt = torch.cat([test[t] for t in times])
    beta = cfg["data"]["diffusion"]["beta"]
    xf = torch.linspace(*FORCE_GRID)[:, None]
    xd = torch.linspace(*DENSITY_GRID)[:, None]
    edges = torch.linspace(*HIST_BINS)
    hist = lambda Y: (torch.histc(Y[:, 0], bins=len(edges) - 1, min=edges[0].item(), max=edges[-1].item())
                      / (len(Y) * (edges[1] - edges[0]))).tolist()
    metrics = {"experiment": EXPERIMENT, "category": CATEGORY, "purpose": cfg["purpose"], "run": run.path.name,
               "dataset": cfg["dataset"]["split_id"], "split": split, "eval_seed": s, "times": times,
               "train_times": sorted(regimes["dense"]), "methods": METHODS,
               "test_cells": {"right_well_fraction": [(test[t] > 0).double().mean().item() for t in times],
                              "hist_edges": edges.tolist(), "hist": [hist(test[t]) for t in times]},
               "true_force": {"x": xf[:, 0].tolist(), "grad_V": P.grad_double_well_1d(xf)[:, 0].tolist()},
               "density_x": xd[:, 0].tolist(), "regimes": {}}
    preds = {}
    with run.new_eval(s, ecfg) as ev:
        for n, (m_theta, m_true) in models.items():
            preds[n] = predictions(regimes[n], test, m_theta, m_true, beta, s)
            err = {}
            for k, x in (("test_cells", Yt), ("grid", xf)):
                gl, gt = (m_theta.c_V * m_theta.grad_V(x)).detach(), P.grad_double_well_1d(x)
                err[k] = ((gl - gt).norm() / gt.norm()).item()
            xb = torch.linspace(-3, 3, 601)[:, None]
            gb = (m_theta.c_V * m_theta.grad_V(xb)).detach()[:, 0]
            metrics["regimes"][n] = {
                "train_times": sorted(regimes[n]),
                "train_cells_per_time": sorted({len(Y) for Y in regimes[n].values()}),
                "V_theta": {"model": f"models/{n}_V_theta.pt", "c_H": m_theta.c_H.item(), "h": m_theta.h.tolist(),
                            "force_rel_l2_test_cells": err["test_cells"], "force_rel_l2_grid": err["grid"],
                            "potential_r2_test_cells": potential_r2(lambda x: m_theta.c_V * m_theta.V(x),
                                                                    P.double_well_1d, Yt),
                            "barrier": xb[:-1, 0][(gb[:-1] > 0) & (gb[1:] <= 0)].tolist(),
                            "force_curve": (m_theta.c_V * m_theta.grad_V(xf)).detach()[:, 0].tolist(),
                            "density": [kde_density(m_theta, t, xd) for t in times]},
                "V_star": {"model": f"models/{n}_V_star.pt", "c_H": m_true.c_H.item(), "h": m_true.h.tolist(),
                           "density": [kde_density(m_true, t, xd) for t in times]},
                "scores": {k: {"W1": [w1_1d(p[t], test[t]) for t in times],
                               "W2": [w2_1d(p[t], test[t]) for t in times],
                               "right_well_fraction": [(p[t] > 0).double().mean().item() for t in times]}
                           for k, p in preds[n].items()},
            }
            r = metrics["regimes"][n]["V_theta"]
            print(f"{n}: c_H {r['c_H']:.3f}, force rel. L2 error on test cells {r['force_rel_l2_test_cells']:.3f}",
                  flush=True)
        ev.save_tensors("predictions.pt", {"times": times, "predictions": preds})
        ev.save_json("metrics.json", metrics)
    return ev


# ── plot ────────────────────────────────────────────────────────────────────────────────────────


def plot(ev, args):
    m = ev.load_json("metrics.json", "this evaluation did not finish; rerun `eval`")
    times, methods, regimes = m["times"], m["methods"], m["regimes"]
    head = f"{m['dataset']}, {m['purpose']}"
    lines = [f"# Double-well validation: run {m['run']}, eval {ev.path.name}", "",
             f"{m['category']}. Dataset and split: {head}.", ""]
    for metric in ("W2", "W1"):
        lines += [f"## {metric}, mean over the {len(times)} test times", ""]
        lines += md_table(["regime"] + [v["label"] for v in methods.values()],
                          [[n] + [mean(r["scores"][k][metric]) for k in methods] for n, r in regimes.items()])
        lines.append("")
    lines += ["## V_theta: forces and potential", ""]
    lines += md_table(["regime", "c_H (true 1)", "force rel. L2, test cells", "force rel. L2, grid",
                       "potential R^2", "barrier (true 0)"],
                      [[n, r["V_theta"]["c_H"], r["V_theta"]["force_rel_l2_test_cells"],
                        r["V_theta"]["force_rel_l2_grid"], r["V_theta"]["potential_r2_test_cells"],
                        ", ".join(f"{b:+.2f}" for b in r["V_theta"]["barrier"]) or "none"]
                       for n, r in regimes.items()])
    lines += ["", f"## Mass in the right well at the last test time (t = {times[-1]:.2f})", ""]
    lines += md_table(["regime"] + [v["label"] for v in methods.values()] + ["test cells"],
                      [[n] + [r["scores"][k]["right_well_fraction"][-1] for k in methods]
                       + [m["test_cells"]["right_well_fraction"][-1]] for n, r in regimes.items()])
    lines += ["", "## Methods", ""] + method_notes(methods)
    write_tables(ev, lines)

    fig, ax = plt.subplots(2, len(regimes), figsize=(4.5 * len(regimes), 6.6), sharey="row", squeeze=False)
    for j, (n, r) in enumerate(regimes.items()):
        for i, metric in enumerate(("W2", "W1")):
            for k, v in methods.items():
                ax[i, j].plot(times, r["scores"][k][metric], STYLES.get(k, "-"), ms=3, label=v["label"])
            for t in m["train_times"]:
                ax[i, j].axvline(t, color="0.85", lw=0.8, zorder=0)
            ax[i, j].set(title=f"{n} ({m['split']})" if i == 0 else None, xlabel="test time t" if i == 1 else None)
    ax[0, 0].set_ylabel("$W_2$ to test cells")
    ax[1, 0].set_ylabel("$W_1$ to test cells")
    ax[0, 0].legend(fontsize=8)
    fig.suptitle(f"{head}   (grey lines: training times)", fontsize=9)
    fig.tight_layout()
    save(fig, ev, f"w_per_time_{m['split']}.png")

    fig, ax = plt.subplots(1, len(regimes), figsize=(4.5 * len(regimes), 3.4), sharey=True, squeeze=False)
    for a, (n, r) in zip(ax[0], regimes.items()):
        a.plot(m["true_force"]["x"], m["true_force"]["grad_V"], "k--", label="$\\nabla V^*$")
        a.plot(m["true_force"]["x"], r["V_theta"]["force_curve"], label="$c_V\\nabla V_\\theta$")
        a.set(title=f"{n}: force error {r['V_theta']['force_rel_l2_grid']:.2f} (grid)", xlabel="x", ylim=(-12, 12))
    ax[0, 0].legend(fontsize=8)
    fig.suptitle(head, fontsize=9)
    fig.tight_layout()
    save(fig, ev, "force.png")

    pick = sorted({0, len(times) // 2, len(times) - 1})
    fig, ax = plt.subplots(len(regimes), len(pick), figsize=(4 * len(pick), 2.6 * len(regimes)),
                           sharex=True, squeeze=False)
    e = m["test_cells"]["hist_edges"]
    for i, (n, r) in enumerate(regimes.items()):
        for j, k in enumerate(pick):
            a = ax[i, j]
            a.stairs(m["test_cells"]["hist"][k], e, fill=True, color="0.85", label="test cells")
            a.plot(m["density_x"], r["V_theta"]["density"][k], label="$V_\\theta$ model KDE")
            a.plot(m["density_x"], r["V_star"]["density"][k], "--", label="$V^*$ model KDE")
            a.set(title=f"{n}, test time t = {times[k]:.2f}")
    ax[0, 0].legend(fontsize=7)
    fig.suptitle(f"distributions at held-out times; {head}", fontsize=9)
    fig.tight_layout()
    save(fig, ev, "distributions.png")
    history_figure(ev, [f"{n}_{s}" for n in regimes for s in ("V_theta", "V_star")])


def save(fig, ev, name):
    out = ev.path / name
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"saved {out}")


if __name__ == "__main__":
    cli(__doc__.splitlines()[0], add_train_args, train, evaluate, plot, resume, runs_dir=HERE / "runs")
