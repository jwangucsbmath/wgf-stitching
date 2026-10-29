"""OU / quadratic validation (step 6a): forward rollout and force recovery of the learned V_theta.

Train V_theta as in step 6a (quadratic V = x^2/2, beta = 0.5, t = 1 held out), freeze it, then run the
step-1 simulator with the learned SDE
    dX = -c_V grad V_theta(X) dt + sqrt(2 c_H) dB
from the training cells at t = 0 and compare the simulated cloud with an independent test dataset at
every snapshot time by exact 1-D W1 and W2. References per time:
  true SDE  - the same simulator with the true V* and beta, same start, independent noise: the
              finite-sample floor;
  stitching - samples from the model's own KDE at that time.
The rollout never sees the learned paths X. Second test, force recovery: relative L2 error of
c_V grad V_theta against grad V*. (The double-well cases moved to experiments/validation/double_well.)

Commands, from the project root (folders and files: wgf/runs.py):
  .venv/bin/python experiments/validation/ou_quadratic/rollout.py train [--epochs 2000] [--seed 0] [--loo] ...
  .venv/bin/python experiments/validation/ou_quadratic/rollout.py eval <run folder> [--eval-seed 0]
  .venv/bin/python experiments/validation/ou_quadratic/rollout.py plot <run folder> [--eval <evals folder>]
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
from fit_nn import build_nn_model  # noqa: E402
from wgf import potentials as P  # noqa: E402
from wgf.datasets import quadratic_snapshots  # noqa: E402
from wgf.kde import sample  # noqa: E402
from wgf.metrics import w1_1d, w2_1d  # noqa: E402
from wgf.report import history_figure, md_table, mean, method_notes, write_tables  # noqa: E402
from wgf.runs import Run, cli  # noqa: E402
from wgf.sim import simulate  # noqa: E402
from wgf.train import fit  # noqa: E402

torch.set_default_dtype(torch.float64)
EXPERIMENT = "ou_quadratic_rollout"
TAG = "6a_quadratic"
METHODS = {
    "theta_sde": dict(label="learned SDE", kind="SDE rollout", dynamics="learned",
                      description="independent rollout of dX = -c_V grad V_theta dt + sqrt(2 c_H) dB from the "
                                  "training cells at t = 0; never uses the paths X"),
    "true_sde": dict(label="true SDE", kind="SDE rollout", dynamics="true",
                     description="true-dynamics reference (finite-sample floor): the true SDE (V*, beta) from "
                                 "the same start with independent noise"),
    "theta_kde": dict(label="stitching KDE", kind="stitching KDE", dynamics="learned",
                      description="samples of the learned model's KDE at t (its fitted particle paths X)"),
}
STYLES = {"theta_sde": "o-", "true_sde": "k--", "theta_kde": "s:"}


# ── train ───────────────────────────────────────────────────────────────────────────────────────


def add_train_args(p):
    p.add_argument("--epochs", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0, help="default for --data-seed and --init-seed")
    p.add_argument("--data-seed", type=int, default=None)
    p.add_argument("--init-seed", type=int, default=None)
    p.add_argument("--lr", type=float, default=5e-3)
    p.add_argument("--lam", type=float, default=1.0)
    p.add_argument("--loo", action="store_true", help="leave-one-out score (default: full score)")
    p.add_argument("--pooled", action="store_true", help="pooled NLL (default: per-snapshot mean, summed)")


def make_config(a):
    return {
        "category": "additional validation on simulated data (not a paper experiment)",
        "data": {"system": "quadratic", "V": "x^2 / 2", "beta": 0.5, "X0": "N(2, 0.5^2)", "M": 1000,
                 "times": [0.0, 0.5, 1.0, 1.5, 2.0], "sim_dt": 1e-3, "sim_scheme": "euler",
                 "paired": False, "builder": "wgf.datasets.quadratic_snapshots"},
        "split": {"train": "dataset from Generator(data seed) without the held-out times",
                  "test": "independent dataset from Generator(data seed + 1), every time",
                  "held_out_times": [1.0]},
        "model": {"builder": "experiments/validation/ou_quadratic/fit_nn.build_nn_model", "N": 100, "K": 41,
                  "t_grid": "linspace(0, 2, K)", "c_H": {"init": 0.5, "learned": False},
                  "X_init": "wgf.init.ot_straight_lines(first, last training snapshot), jitter 0.01",
                  "h_init": "silverman(first training snapshot); learned via softplus", "weights": "uniform 1/N",
                  "score": "loo" if a.loo else "full",
                  "V_theta": {"type": "mlp", "d_in": 1, "hidden": [64, 64], "activation": "silu",
                              "residual_skip": True, "head": "linear, zero init, no bias"},
                  "c_V": 1.0, "W": None},
        "eval": {"rollout_start": "training cells at t = 0", "rollout_dt": 1e-3, "rollout_scheme": "euler",
                 "force_grid": "linspace(min, max of the test cells, 401)"},
        "train": {"optimizer": "Adam, full batch, one step per epoch", "epochs": a.epochs, "lr": a.lr,
                  "lr_schedule": "cosine to 0.05 lr", "lam": a.lam, "residual_scheme": "trapezoid",
                  "nll": {"pooled": a.pooled,
                          "convention": "one mean over all cells" if a.pooled
                          else "mean NLL per snapshot, summed over snapshots"}},
        "seeds": {"data": a.seed if a.data_seed is None else a.data_seed,
                  "init": a.seed if a.init_seed is None else a.init_seed,
                  "data_uses": "train Generator(data), test Generator(data + 1)",
                  "init_uses": "Generator(init) for ot_straight_lines, then torch.manual_seed(init) "
                               "before the network is built; fit() draws no random numbers",
                  "evaluation": "set per evaluation (eval --eval-seed), recorded in eval.json"},
        "dtype": "float64",
    }


def make_data(cfg):
    """Training and test datasets, regenerated exactly from the config and data seed (small: 5 x 1000 cells)."""
    s, d = cfg["seeds"]["data"], cfg["data"]
    quad = lambda seed: quadratic_snapshots(torch.Generator().manual_seed(seed), M=d["M"], beta=d["beta"],
                                            times=tuple(d["times"]), dt=d["sim_dt"])
    full = quad(s)
    return {"train": {t: Y for t, Y in full.items() if t not in cfg["split"]["held_out_times"]}, "test": quad(s + 1)}


def train(args):
    cfg = make_config(args)
    data = make_data(cfg)
    tr, m = cfg["train"], cfg["model"]
    with Run.create(args.name or "rollout", EXPERIMENT, cfg, args.runs_dir) as run:
        print(f"run {run.path}")
        run.save_data(data)
        model = build_nn_model(data["train"], cfg["seeds"]["init"], m["K"], m["N"], m["score"] == "loo")
        hist = fit(model, data["train"], epochs=tr["epochs"], lr=tr["lr"], lam=tr["lam"],
                   scheme=tr["residual_scheme"], pooled=tr["nll"]["pooled"], log_every=0)
        run.save_model(TAG, model)
        run.save_history(TAG, hist)
        print(f"{TAG}  {hist['seconds']:.0f} s   c_H {model.c_H.item():.3f}", flush=True)
    return run


# ── eval ────────────────────────────────────────────────────────────────────────────────────────


@torch.no_grad()
def rollout_predictions(model, X0, test, dt, beta_true, grad_V_true, seed=0):
    """{method: {t: cloud}} at every test time: learned-SDE rollout, true-SDE rollout, stitching KDE."""
    times = sorted(test)
    learned_grad = lambda x: model.c_V * model.grad_V(x)
    learned = dict(simulate(X0, times, dt, float(model.c_H), learned_grad,
                            gen=torch.Generator().manual_seed(seed + 1)))
    true = dict(simulate(X0, times, dt, beta_true, grad_V_true, gen=torch.Generator().manual_seed(seed + 2)))
    g = torch.Generator().manual_seed(seed + 3)
    kde = {t: sample(model.centres_at(t), model.h, model.w, len(test[t]), g) for t in times}
    return {"theta_sde": {t: learned[t] for t in times}, "true_sde": {t: true[t] for t in times},
            "theta_kde": kde}


def force_error(model, grad_V_true, Y, xs):
    """Force recovery: relative L2 error of c_V grad V_theta against grad V*,
        err = ||c_V grad V_theta - grad V*|| / ||grad V*||,
    on the test cells Y (weighted by where the data are) and on a uniform grid xs (the whole range,
    including sparsely visited regions). Gradients have no free constant, so nothing is centred."""
    out = {}
    for k, x in (("test_cells", Y), ("grid", xs)):
        gl = (model.c_V * model.grad_V(x)).detach()
        gt = grad_V_true(x)
        out[k] = ((gl - gt).norm() / gt.norm()).item()
    return out


def evaluate(args):
    run = Run.open(args.run, EXPERIMENT)
    cfg, s = run.config, args.eval_seed
    data = run.load_data()  # the run's own saved copy; nothing is re-simulated
    model = run.load_model(TAG)
    train_, test = data["train"], data["test"]
    times = sorted(test)
    Yt = torch.cat([test[t] for t in times])
    grid = torch.linspace(Yt.min().item(), Yt.max().item(), 401)[:, None]
    beta = cfg["data"]["beta"]
    ecfg = {"rollout": cfg["eval"],
            "kde_samples": "wgf.kde.sample (systematic resampling), one per test cell",
            "metrics": {"W1": "exact 1-D, predicted cloud vs test cells", "W2": "exact 1-D",
                        "force_rel_l2": "||c_V grad V_theta - grad V*|| / ||grad V*||, on all test cells and on a grid"},
            "seeds": {"eval": s, "rollout_V_theta": s + 1, "rollout_true": s + 2, "kde_samples": s + 3},
            "methods": METHODS}
    with run.new_eval(s, ecfg) as ev:
        p = rollout_predictions(model, train_[0.0], test, cfg["eval"]["rollout_dt"], beta, P.grad_quadratic, s)
        r = {"model": f"models/{TAG}.pt", "times": times, "train_times": sorted(train_),
             "held_out_times": [t for t in times if t not in train_], "beta_true": beta,
             "c_H": model.c_H.item(), "h": model.h.tolist(),
             "force_rel_l2": force_error(model, P.grad_quadratic, Yt, grid),
             "scores": {k: {"W1": [w1_1d(c[t], test[t]) for t in times],
                            "W2": [w2_1d(c[t], test[t]) for t in times]} for k, c in p.items()}}
        print(f"{TAG} mean W1: " + "  ".join(f"{METHODS[k]['label']} {mean(r['scores'][k]['W1']):.3f}"
                                            for k in METHODS), flush=True)
        ev.save_tensors("predictions.pt", {TAG: p})
        ev.save_json("metrics.json", {"experiment": EXPERIMENT, "run": run.path.name, "eval_seed": s,
                                      "methods": METHODS, "models": {TAG: r}})
    return ev


# ── plot ────────────────────────────────────────────────────────────────────────────────────────


def plot(ev, args):
    m = ev.load_json("metrics.json", "this evaluation did not finish; rerun `eval`")
    methods, r = m["methods"], m["models"][TAG]
    labels = [v["label"] for v in methods.values()]
    lines = [f"# {m['experiment']}: run {m['run']}, eval {ev.path.name}", ""]
    lines += md_table(["metric"] + labels, [[f"mean {k}"] + [mean(r["scores"][c][k]) for c in methods]
                                            for k in ("W1", "W2")])
    lines += ["", f"Force rel. L2 error: test cells {r['force_rel_l2']['test_cells']:.3f}, grid "
                  f"{r['force_rel_l2']['grid']:.3f}; c_H {r['c_H']:.3f} (true beta {r['beta_true']})", "",
              "## W1 per time", ""]
    lines += md_table(["t"] + labels + ["held out"],
                      [[f"{t:.2f}"] + [r["scores"][c]["W1"][i] for c in methods]
                       + ["yes" if t in r["held_out_times"] else ""] for i, t in enumerate(r["times"])])
    lines += ["", "## Methods", ""] + method_notes(methods)
    write_tables(ev, lines)

    fig, ax = plt.subplots(1, 2, figsize=(8, 3.2))
    for a, metric in zip(ax, ("W1", "W2")):
        for k, v in methods.items():
            a.plot(r["times"], r["scores"][k][metric], STYLES.get(k, "-"), ms=3, label=v["label"])
        a.set(title=f"{TAG}: {metric} to test cells", xlabel="t")
    ax[0].legend(fontsize=8)
    fig.tight_layout()
    out = ev.path / "rollout.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"saved {out}")
    history_figure(ev, [TAG])


if __name__ == "__main__":
    cli(__doc__.splitlines()[0], add_train_args, train, evaluate, plot, runs_dir=HERE / "runs")
