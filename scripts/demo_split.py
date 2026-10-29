"""Held-out protocol for the double-well (after step 6b): one split, three regimes, three references.

Split (wgf.datasets.split_snapshots): 20 snapshots; train times interleaved with test times, both
endpoints in train; train and test cells come from disjoint trajectories. Every regime is scored
on the same test times and test cells. Nothing is tuned on the test times (paper presets only).

At each test time, against the test cells (W1 and W2, exact 1-D):
  OT interp   - quantile average of the two bracketing training snapshots: no physics, the ceiling
  V_theta     - stitching KDE of the learned model, and a forward rollout of its SDE
                (c_V grad V_theta, c_H) from the training cloud at t = 0
  V*          - the same stitching model with V fixed to the truth and c_H = beta (learns only X, h),
                and the true SDE rolled out from the same training cloud: the floor
A useful V_theta sits clearly between the OT ceiling and the V* floor.

Commands, from the project root (folders and files: wgf/runs.py):
  .venv/bin/python scripts/demo_split.py train [--epochs 2000] [--split interleaved] [--seed 0] [--loo] ...
  .venv/bin/python scripts/demo_split.py eval runs/<run> [--eval-seed 0]
  .venv/bin/python scripts/demo_split.py plot runs/<run> [--eval <evals folder>]
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from run_plots import history_figure, md_table, mean, method_notes, write_tables  # noqa: E402
from wgf import potentials as P  # noqa: E402
from wgf.baselines import ot_interpolate_snapshots  # noqa: E402
from wgf.datasets import SPLITS, double_well_regimes  # noqa: E402
from wgf.init import ot_straight_lines  # noqa: E402
from wgf.kde import sample, silverman  # noqa: E402
from wgf.metrics import potential_r2, w1_1d, w2_1d  # noqa: E402
from wgf.model import FixedV, Stitching  # noqa: E402
from wgf.nn import MLP  # noqa: E402
from wgf.runs import Run, cli, data_hash  # noqa: E402
from wgf.sim import simulate  # noqa: E402
from wgf.train import fit  # noqa: E402

torch.set_default_dtype(torch.float64)
EXPERIMENT = "double_well_heldout"
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


# ── train ───────────────────────────────────────────────────────────────────────────────────────


def add_train_args(p):
    p.add_argument("--epochs", type=int, default=2000)
    p.add_argument("--split", default="interleaved", choices=list(SPLITS))
    p.add_argument("--seed", type=int, default=0, help="default for --data-seed and --init-seed")
    p.add_argument("--data-seed", type=int, default=None)
    p.add_argument("--init-seed", type=int, default=None)
    p.add_argument("--N", type=int, default=100)
    p.add_argument("--K", type=int, default=50)
    p.add_argument("--lr", type=float, default=5e-3)
    p.add_argument("--lam", type=float, default=1.0)
    p.add_argument("--loo", action="store_true", help="leave-one-out score (default: full score)")
    p.add_argument("--pooled", action="store_true", help="pooled NLL (default: per-snapshot mean, summed)")
    p.add_argument("--c-H-init", type=float, default=1.0)
    p.add_argument("--fix-c-H", action="store_true", help="keep c_H at its initial value")
    p.add_argument("--M", type=int, default=2000, help="cells per snapshot before the train/test split")
    p.add_argument("--trickle-cells", type=int, default=10)


def make_config(a):
    beta = 1.0
    return {
        "data": {"system": "double_well_1d", "V": "(|x| - 7)^2 / 2", "beta": beta, "X0": "N(0, 1)",
                 "M": a.M, "n_snap": 20, "T": 4.0, "sim_dt": 1e-2, "sim_scheme": "euler", "paired": True,
                 "builder": "wgf.datasets.double_well_regimes"},
        "split": {"name": a.split, "train_idx": SPLITS[a.split], "train_frac": 0.5,
                  "by": "time and trajectory (test cells never share a trajectory with training cells)"},
        "regimes": {"names": ["dense", "terminal", "trickle"], "trickle_cells": a.trickle_cells},
        "model": {"N": a.N, "K": a.K, "t_grid": "linspace(first, last training time, K)",
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
                          else "mean NLL per snapshot, summed over snapshots"}},
        "seeds": {"data": a.seed if a.data_seed is None else a.data_seed,
                  "init": a.seed if a.init_seed is None else a.init_seed,
                  "init_uses": "Generator(init) for ot_straight_lines, then torch.manual_seed(init) "
                               "before the network is built; fit() draws no random numbers"},
        "dtype": "float64",
    }


def make_data(cfg):
    """({regime: train dict}, test dict), regenerated exactly from the config and data seed."""
    d = cfg["data"]
    return double_well_regimes(torch.Generator().manual_seed(cfg["seeds"]["data"]), cfg["split"]["train_idx"],
                               M=d["M"], trickle_cells=cfg["regimes"]["trickle_cells"],
                               beta=d["beta"], n_snap=d["n_snap"], T=d["T"], dt=d["sim_dt"])


def train_model(train, cfg, true_V=False):
    m, tr, seed = cfg["model"], cfg["train"], cfg["seeds"]["init"]
    times = sorted(train)
    t_grid = torch.linspace(times[0], times[-1], m["K"])
    X = ot_straight_lines(train[times[0]], train[times[-1]], t_grid, m["N"], torch.Generator().manual_seed(seed))
    torch.manual_seed(seed)
    loo = m["score"] == "loo"
    if true_V:
        model = Stitching(t_grid, X, silverman(train[times[0]]), FixedV(P.double_well_1d),
                          c_H=cfg["data"]["beta"], loo=loo)
    else:
        V = MLP(m["V_theta"]["d_in"], tuple(m["V_theta"]["hidden"]))
        model = Stitching(t_grid, X, silverman(train[times[0]]), V, c_H=m["c_H"]["init"],
                          learn_c_H=m["c_H"]["learned"], loo=loo)
    hist = fit(model, train, epochs=tr["epochs"], lr=tr["lr"], lam=tr["lam"], scheme=tr["residual_scheme"],
               pooled=tr["nll"]["pooled"], log_every=0)
    return model, hist


def train(args):
    cfg = make_config(args)
    regimes, test = make_data(cfg)
    with Run.create(args.name or f"split-{args.split}", EXPERIMENT, cfg, args.runs_dir) as run:
        print(f"run {run.path}")
        run.save_data({"regimes": regimes, "test": test})
        for name in cfg["regimes"]["names"]:
            for tag, true_V in ((f"{name}_V_theta", False), (f"{name}_V_star", True)):
                model, hist = train_model(regimes[name], cfg, true_V)
                run.save_model(tag, model)
                run.save_history(tag, hist)
                print(f"{tag:17} {hist['seconds']:5.0f} s   c_H {model.c_H.item():.3f}   h {model.h.item():.3f}",
                      flush=True)
    return run


# ── eval ────────────────────────────────────────────────────────────────────────────────────────


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


def evaluate(args):
    run = Run.open(args.run, EXPERIMENT)
    cfg, s = run.config, args.eval_seed
    data = run.load_data()
    regimes, test = data["regimes"], data["test"]
    # load every checkpoint first, so a missing one fails before an evals/ folder is made
    models = {n: (run.load_model(f"{n}_V_theta"), run.load_model(f"{n}_V_star")) for n in cfg["regimes"]["names"]}
    regenerated = make_data(cfg)
    ecfg = {"rollout": {"dt": 1e-2, "scheme": "euler",
                        "start": "training cloud at t0, resampled with replacement to the test-cell count "
                                 "when the sizes differ"},
            "kde_samples": "wgf.kde.sample (systematic resampling), one per test cell",
            "metrics": {"W1": "exact 1-D, predicted cloud vs test cells", "W2": "exact 1-D",
                        "force_rel_l2_test_cells": "||c_V grad V_theta - grad V*|| / ||grad V*|| on all test cells",
                        "potential_r2_test_cells": "centred R^2 of c_V V_theta vs V* on all test cells"},
            "seeds": {"eval": s, "start_resample_and_kde_samples": s + 10, "rollout_V_theta": s + 1,
                      "rollout_V_star": s + 2},
            "data_regenerated_from_seed_matches": data_hash(dict(zip(("regimes", "test"), regenerated)))
            == run.manifest["data_sha256"],
            "methods": METHODS}
    times = sorted(test)
    Yt = torch.cat([test[t] for t in times])
    beta = cfg["data"]["beta"]
    metrics = {"experiment": EXPERIMENT, "run": run.path.name, "split": cfg["split"]["name"], "eval_seed": s,
               "times": times, "methods": METHODS, "regimes": {}}
    preds = {}
    with run.new_eval(s, ecfg) as ev:
        for n, (m_theta, m_true) in models.items():
            preds[n] = predictions(regimes[n], test, m_theta, m_true, beta, s)
            gl, gt = (m_theta.c_V * m_theta.grad_V(Yt)).detach(), P.grad_double_well_1d(Yt)
            metrics["regimes"][n] = {
                "train_times": sorted(regimes[n]),
                "V_theta": {"model": f"models/{n}_V_theta.pt", "c_H": m_theta.c_H.item(), "h": m_theta.h.tolist(),
                            "force_rel_l2_test_cells": ((gl - gt).norm() / gt.norm()).item(),
                            "potential_r2_test_cells": potential_r2(lambda x: m_theta.c_V * m_theta.V(x),
                                                                    P.double_well_1d, Yt)},
                "V_star": {"model": f"models/{n}_V_star.pt", "c_H": m_true.c_H.item(), "h": m_true.h.tolist()},
                "scores": {k: {"W1": [w1_1d(p[t], test[t]) for t in times],
                               "W2": [w2_1d(p[t], test[t]) for t in times]} for k, p in preds[n].items()},
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
    lines = [f"# {m['experiment']}: run {m['run']}, eval {ev.path.name}, split {m['split']}", ""]
    for metric in ("W2", "W1"):
        lines += [f"## {metric}, mean over the {len(times)} test times", ""]
        lines += md_table(["regime"] + [v["label"] for v in methods.values()],
                          [[n] + [mean(r["scores"][k][metric]) for k in methods] for n, r in regimes.items()])
        lines.append("")
    lines += ["## V_theta", ""]
    lines += md_table(["regime", "c_H", "force rel. L2 (test cells)", "potential R^2 (test cells)"],
                      [[n, r["V_theta"]["c_H"], r["V_theta"]["force_rel_l2_test_cells"],
                        r["V_theta"]["potential_r2_test_cells"]] for n, r in regimes.items()])
    lines += ["", "## Methods", ""]
    lines += method_notes(methods)
    write_tables(ev, lines)

    fig, ax = plt.subplots(2, len(regimes), figsize=(4.5 * len(regimes), 6.6), sharey="row", squeeze=False)
    for j, (n, r) in enumerate(regimes.items()):
        for i, metric in enumerate(("W2", "W1")):
            for k, v in methods.items():
                ax[i, j].plot(times, r["scores"][k][metric], STYLES.get(k, "-"), ms=3, label=v["label"])
            ax[i, j].set(title=f"{n} ({m['split']})" if i == 0 else None, xlabel="test time t" if i == 1 else None)
        ax[0, 0].set_ylabel("$W_2$ to test cells")
        ax[1, 0].set_ylabel("$W_1$ to test cells")
    ax[0, 0].legend(fontsize=8)
    fig.tight_layout()
    out = ev.path / f"w_per_time_{m['split']}.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"saved {out}")
    history_figure(ev, [f"{n}_{s}" for n in regimes for s in ("V_theta", "V_star")])


if __name__ == "__main__":
    cli(__doc__.splitlines()[0], add_train_args, train, evaluate, plot)
