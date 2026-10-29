"""Forward-simulation test of the learned potential (after steps 6a and 6b).

Train V_theta as in step 6, freeze it, then run the step-1 simulator with the learned SDE
    dX = -c_V grad V_theta(X) dt + sqrt(2 c_H) dB
from the observed cells at t = 0 and compare the simulated cloud with held-out cells at every
snapshot time by exact 1-D W1 and W2. Two references per time:
  true SDE  - the same simulator with the true V* and beta, same start, independent noise: the
              finite-sample floor (W1 a perfect V would get);
  stitching - samples from the model's own KDE at that time (what steps 5-6 already evaluated).
The rollout never sees the learned paths X; it uses only V_theta and c_H.
Second test, force recovery: relative L2 error of c_V grad V_theta against grad V*.

Commands, from the project root (folders and files: wgf/runs.py):
  .venv/bin/python scripts/demo_rollout.py train [--epochs 2000] [--seed 0] [--loo] ...
  .venv/bin/python scripts/demo_rollout.py eval runs/<run> [--eval-seed 0]
  .venv/bin/python scripts/demo_rollout.py plot runs/<run> [--eval <evals folder>]
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
from demo_double_well import make_regimes, train_regime  # noqa: E402
from demo_fit_nn import build_nn_model  # noqa: E402
from run_plots import history_figure, md_table, mean, method_notes, write_tables  # noqa: E402
from wgf import potentials as P  # noqa: E402
from wgf.datasets import quadratic_snapshots  # noqa: E402
from wgf.kde import sample  # noqa: E402
from wgf.metrics import w1_1d, w2_1d  # noqa: E402
from wgf.runs import Run, cli, data_hash  # noqa: E402
from wgf.sim import simulate  # noqa: E402
from wgf.train import fit  # noqa: E402

torch.set_default_dtype(torch.float64)
EXPERIMENT = "rollout_force"
METHODS = {
    "theta_sde": dict(label="learned SDE", kind="SDE rollout", dynamics="learned",
                      description="independent rollout of dX = -c_V grad V_theta dt + sqrt(2 c_H) dB from the "
                                  "observed cells at t = 0; never uses the paths X"),
    "true_sde": dict(label="true SDE", kind="SDE rollout", dynamics="true",
                     description="true-dynamics reference (finite-sample floor): the true SDE (V*, beta) from "
                                 "the same start with independent noise"),
    "theta_kde": dict(label="stitching KDE", kind="stitching KDE", dynamics="learned",
                      description="samples of the learned model's KDE at t (its fitted particle paths X)"),
}
STYLES = {"theta_sde": "o-", "true_sde": "k--", "theta_kde": "s:"}
V_THETA = {"type": "mlp", "d_in": 1, "hidden": [64, 64], "activation": "silu", "residual_skip": True,
           "head": "linear, zero init, no bias"}


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
    score = "loo" if a.loo else "full"
    common = {"X_init": "wgf.init.ot_straight_lines(first, last training snapshot), jitter 0.01",
              "h_init": "silverman(first training snapshot); learned via softplus", "weights": "uniform 1/N",
              "score": score, "V_theta": V_THETA, "c_V": 1.0, "W": None}
    return {
        "experiments": {
            "6a_quadratic": {
                "data": {"system": "quadratic", "V": "x^2 / 2", "beta": 0.5, "X0": "N(2, 0.5^2)", "M": 1000,
                         "times": [0.0, 0.5, 1.0, 1.5, 2.0], "sim_dt": 1e-3, "sim_scheme": "euler",
                         "paired": False, "builder": "wgf.datasets.quadratic_snapshots"},
                "split": {"train": "dataset from Generator(data seed) without the held-out times",
                          "test": "independent dataset from Generator(data seed + 1), every time",
                          "held_out_times": [1.0]},
                "model": {"builder": "scripts/demo_fit_nn.build_nn_model", "N": 100, "K": 41,
                          "t_grid": "linspace(0, 2, K)", "c_H": {"init": 0.5, "learned": False}, **common},
                "eval": {"rollout_start": "training cells at t = 0", "rollout_dt": 1e-3, "rollout_scheme": "euler",
                         "force_grid": "linspace(min, max of the test cells, 401)"}},
            "6b_double_well": {
                "data": {"system": "double_well_1d", "V": "(|x| - 7)^2 / 2", "beta": 1.0, "X0": "N(0, 1)",
                         "M": 2000, "n_snap": 20, "T": 4.0, "sim_dt": 1e-2, "sim_scheme": "euler", "paired": True,
                         "builder": "scripts/demo_double_well.make_regimes"},
                "split": {"cells": "random 50/50 by trajectory", "test": "the test cells at all 20 times"},
                "regimes": {"dense": "all 20 times, 1000 cells each", "terminal": "first and last time only",
                            "trickle": "all 20 times, 10 cells each, a fresh subset per time"},
                "model": {"builder": "scripts/demo_double_well.train_regime", "N": 100, "K": 50,
                          "t_grid": "linspace(first, last training time, K)",
                          "c_H": {"init": 1.0, "learned": True}, **common},
                "eval": {"rollout_start": "test cells at t = 0", "rollout_dt": 1e-2, "rollout_scheme": "euler",
                         "force_grid": "linspace(-10, 10, 401)"}},
        },
        "train": {"optimizer": "Adam, full batch, one step per epoch", "epochs": a.epochs, "lr": a.lr,
                  "lr_schedule": "cosine to 0.05 lr", "lam": a.lam, "residual_scheme": "trapezoid",
                  "nll": {"pooled": a.pooled,
                          "convention": "one mean over all cells" if a.pooled
                          else "mean NLL per snapshot, summed over snapshots"}},
        "seeds": {"data": a.seed if a.data_seed is None else a.data_seed,
                  "init": a.seed if a.init_seed is None else a.init_seed,
                  "data_uses": "6a train Generator(data), 6a test Generator(data + 1); 6b Generator(data) for "
                               "the snapshots, the cell split and the trickle subsets",
                  "init_uses": "Generator(init) for ot_straight_lines, then torch.manual_seed(init) "
                               "before the network is built; fit() draws no random numbers"},
        "dtype": "float64",
    }


def make_data(cfg):
    """Every dataset of the run, regenerated exactly from the config and data seed."""
    s, e = cfg["seeds"]["data"], cfg["experiments"]["6a_quadratic"]
    d = e["data"]
    quad = lambda seed: quadratic_snapshots(torch.Generator().manual_seed(seed), M=d["M"], beta=d["beta"],
                                            times=tuple(d["times"]), dt=d["sim_dt"])
    full = quad(s)
    regimes, test = make_regimes(s)
    return {"6a_quadratic": {"train": {t: Y for t, Y in full.items() if t not in e["split"]["held_out_times"]},
                             "test": quad(s + 1)},
            "6b_double_well": {"regimes": regimes, "test": test}}


def train(args):
    cfg = make_config(args)
    data = make_data(cfg)
    tr, seed = cfg["train"], cfg["seeds"]["init"]
    fit_kw = dict(lr=tr["lr"], lam=tr["lam"], scheme=tr["residual_scheme"], pooled=tr["nll"]["pooled"])
    with Run.create(args.name or "rollout", EXPERIMENT, cfg, args.runs_dir) as run:
        print(f"run {run.path}")
        run.save_data(data)
        m = cfg["experiments"]["6a_quadratic"]["model"]
        model = build_nn_model(data["6a_quadratic"]["train"], seed, m["K"], m["N"], m["score"] == "loo")
        trained = {"6a_quadratic": (model, fit(model, data["6a_quadratic"]["train"], epochs=tr["epochs"],
                                                log_every=0, **fit_kw))}
        m = cfg["experiments"]["6b_double_well"]["model"]
        for n, train_n in data["6b_double_well"]["regimes"].items():
            trained[f"6b_{n}"] = train_regime(train_n, tr["epochs"], seed, m["K"], m["N"], m["score"] == "loo",
                                              **fit_kw)
        for tag, (model, hist) in trained.items():
            run.save_model(tag, model)
            run.save_history(tag, hist)
            print(f"{tag:13} {hist['seconds']:5.0f} s   c_H {model.c_H.item():.3f}", flush=True)
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


def eval_setups(cfg, data):
    """Per model tag: its training and test data, rollout start, true dynamics and force grid."""
    out = {}
    for exp, d in data.items():
        e = cfg["experiments"][exp]
        beta, grad_true = e["data"]["beta"], getattr(P, "grad_" + e["data"]["system"])
        test = d["test"]
        Yt = torch.cat([test[t] for t in sorted(test)])
        if exp == "6a_quadratic":
            trains = {exp: d["train"]}
            X0 = d["train"][0.0]
            grid = torch.linspace(Yt.min().item(), Yt.max().item(), 401)[:, None]
        else:
            trains = {f"6b_{n}": tr for n, tr in d["regimes"].items()}
            X0 = test[0.0]
            grid = torch.linspace(-10, 10, 401)[:, None]
        for tag, tr in trains.items():
            out[tag] = dict(experiment=exp, train=tr, test=test, X0=X0, dt=e["eval"]["rollout_dt"], beta=beta, grad=grad_true,
                            Yt=Yt, grid=grid)
    return out


def evaluate(args):
    run = Run.open(args.run, EXPERIMENT)
    cfg, s = run.config, args.eval_seed
    data = run.load_data()
    setups = eval_setups(cfg, data)
    models = {tag: run.load_model(tag) for tag in setups}  # all checkpoints before any eval folder exists
    ecfg = {"rollouts": {tag: cfg["experiments"][su["experiment"]]["eval"] for tag, su in setups.items()},
            "kde_samples": "wgf.kde.sample (systematic resampling), one per test cell",
            "metrics": {"W1": "exact 1-D, predicted cloud vs test cells", "W2": "exact 1-D",
                        "force_rel_l2": "||c_V grad V_theta - grad V*|| / ||grad V*||, on all test cells and on a grid",
                        "right_well_fraction": "6b: share of x > 0 in the learned rollout and in the test cells"},
            "seeds": {"eval": s, "rollout_V_theta": s + 1, "rollout_true": s + 2, "kde_samples": s + 3},
            "data_regenerated_from_seed_matches": data_hash(make_data(cfg)) == run.manifest["data_sha256"],
            "methods": METHODS}
    metrics = {"experiment": EXPERIMENT, "run": run.path.name, "eval_seed": s, "methods": METHODS, "models": {}}
    preds = {}
    with run.new_eval(s, ecfg) as ev:
        for tag, su in setups.items():
            model, test = models[tag], su["test"]
            times = sorted(test)
            preds[tag] = p = rollout_predictions(model, su["X0"], test, su["dt"], su["beta"], su["grad"], s)
            r = {"model": f"models/{tag}.pt", "times": times, "train_times": sorted(su["train"]),
                 "held_out_times": [t for t in times if t not in su["train"]], "beta_true": su["beta"],
                 "c_H": model.c_H.item(), "h": model.h.tolist(),
                 "force_rel_l2": force_error(model, su["grad"], su["Yt"], su["grid"]),
                 "scores": {k: {"W1": [w1_1d(c[t], test[t]) for t in times],
                                "W2": [w2_1d(c[t], test[t]) for t in times]} for k, c in p.items()}}
            if tag.startswith("6b"):
                r["right_well_fraction"] = {"theta_sde": [(p["theta_sde"][t] > 0).double().mean().item() for t in times],
                                            "test_cells": [(test[t] > 0).double().mean().item() for t in times]}
            metrics["models"][tag] = r
            print(f"{tag:13} mean W1: " + "  ".join(f"{METHODS[k]['label']} {mean(r['scores'][k]['W1']):.3f}"
                                                    for k in METHODS), flush=True)
        ev.save_tensors("predictions.pt", preds)
        ev.save_json("metrics.json", metrics)
    return ev


# ── plot ────────────────────────────────────────────────────────────────────────────────────────


def plot(ev, args):
    m = ev.load_json("metrics.json", "this evaluation did not finish; rerun `eval`")
    methods, models = m["methods"], m["models"]
    labels = [v["label"] for v in methods.values()]
    lines = [f"# {m['experiment']}: run {m['run']}, eval {ev.path.name}", ""]
    for metric in ("W1", "W2"):
        lines += [f"## {metric}, mean over the snapshot times", ""]
        lines += md_table(["run"] + labels, [[tag] + [mean(r["scores"][k][metric]) for k in methods]
                                              for tag, r in models.items()])
        lines.append("")
    lines += ["## Force recovery", ""]
    lines += md_table(["run", "rel. L2 on test cells", "rel. L2 on grid", "c_H", "true beta"],
                      [[tag, r["force_rel_l2"]["test_cells"], r["force_rel_l2"]["grid"], r["c_H"], r["beta_true"]]
                       for tag, r in models.items()])
    lines.append("")
    for tag, r in models.items():
        lines += [f"## {tag}: W1 per time", ""]
        rows = []
        for i, t in enumerate(r["times"]):
            row = [f"{t:.2f}"] + [r["scores"][k]["W1"][i] for k in methods] + ["yes" if t in r["held_out_times"] else ""]
            if "right_well_fraction" in r:
                row += [r["right_well_fraction"]["theta_sde"][i], r["right_well_fraction"]["test_cells"][i]]
            rows.append(row)
        extra = ["right well, rollout", "right well, test"] if "right_well_fraction" in r else []
        lines += md_table(["t"] + labels + ["held out"] + extra, rows) + [""]
    lines += ["## Methods", ""] + method_notes(methods)
    write_tables(ev, lines)

    fig, ax = plt.subplots(2, len(models), figsize=(4 * len(models), 6.4), squeeze=False)
    for j, (tag, r) in enumerate(models.items()):
        for i, metric in enumerate(("W1", "W2")):
            for k, v in methods.items():
                ax[i, j].plot(r["times"], r["scores"][k][metric], STYLES.get(k, "-"), ms=3, label=v["label"])
            ax[i, j].set(title=tag if i == 0 else None, xlabel="t" if i == 1 else None)
        ax[0, j].set_ylabel("$W_1$ to test cells")
        ax[1, j].set_ylabel("$W_2$ to test cells")
    ax[0, 0].legend(fontsize=8)
    fig.tight_layout()
    out = ev.path / "rollout.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"saved {out}")
    history_figure(ev, list(models))


if __name__ == "__main__":
    cli(__doc__.splitlines()[0], add_train_args, train, evaluate, plot)
