"""Forward-simulation test of the learned potential (after steps 6a and 6b).

Train V_theta as in step 6, freeze it, then run the step-1 simulator with the learned SDE
    dX = -c_V grad V_theta(X) dt + sqrt(2 c_H) dB
from the observed cells at t = 0 and compare the simulated cloud with held-out cells at every
snapshot time by exact 1-D W1. Two references per time:
  true SDE  - the same simulator with the true V* and beta, same start, independent noise: the
              finite-sample floor (W1 a perfect V would get);
  stitching - samples from the model's own KDE at that time (what steps 5-6 already evaluated).
The rollout never sees the learned paths X; it uses only V_theta and c_H.

Run from the project root:  .venv/bin/python scripts/demo_rollout.py [--epochs 2000]
"""

import argparse
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
from wgf import potentials as P  # noqa: E402
from wgf.datasets import quadratic_snapshots  # noqa: E402
from wgf.kde import sample  # noqa: E402
from wgf.metrics import w1_1d  # noqa: E402
from wgf.sim import simulate  # noqa: E402
from wgf.train import fit  # noqa: E402

torch.set_default_dtype(torch.float64)


@torch.no_grad()
def rollout_w1(model, X0, test, dt, beta_true, grad_V_true, seed=0):
    """W1 per test time for: learned-SDE rollout, true-SDE rollout, stitching KDE samples."""
    times = sorted(test)
    learned_grad = lambda x: model.c_V * model.grad_V(x)
    learned = dict(simulate(X0, times, dt, float(model.c_H), learned_grad,
                            gen=torch.Generator().manual_seed(seed + 1)))
    true = dict(simulate(X0, times, dt, beta_true, grad_V_true, gen=torch.Generator().manual_seed(seed + 2)))
    g = torch.Generator().manual_seed(seed + 3)
    out = {}
    for t in times:
        Y = test[t]
        kde = sample(model.centres_at(t), model.h, model.w, len(Y), g)
        out[t] = dict(learned=w1_1d(learned[t], Y), true=w1_1d(true[t], Y), stitch=w1_1d(kde, Y),
                      right=((learned[t] > 0).double().mean().item(), (Y > 0).double().mean().item()))
    return out


def force_error(model, grad_V_true, Y, xs):
    """Force recovery: relative L2 error of c_V grad V_theta against grad V*,
        err = ||c_V grad V_theta - grad V*|| / ||grad V*||,
    on the test cells Y (weighted by where the data are) and on a uniform grid xs (the whole range,
    including sparsely visited regions). Gradients have no free constant, so nothing is centred."""
    out = {}
    for k, x in (("cells", Y), ("grid", xs)):
        gl = (model.c_V * model.grad_V(x)).detach()
        gt = grad_V_true(x)
        out[k] = ((gl - gt).norm() / gt.norm()).item()
    out["c_H"] = float(model.c_H)
    return out


def report_force(name, f, beta):
    print(f"{name:15} force rel. L2 error: on test cells {f['cells']:.3f}, on grid {f['grid']:.3f};"
          f"  c_H {f['c_H']:.3f} (true beta {beta})")


def report(name, res, unseen=()):
    print(f"\n{name}:   {'t':>5} {'learned SDE':>12} {'true SDE':>9} {'stitching':>10}")
    for t, r in res.items():
        mark = "  (held-out time)" if t in unseen else ""
        print(f"{'':{len(name) + 4}}{t:5.2f} {r['learned']:12.3f} {r['true']:9.3f} {r['stitch']:10.3f}{mark}")
    mean = {k: sum(r[k] for r in res.values()) / len(res) for k in ("learned", "true", "stitch")}
    t_end = list(res)[-1]
    if name.startswith("6b"):
        print(f"{'':{len(name) + 4}}fraction in right well at t={t_end:.2f}: rollout {res[t_end]['right'][0]:.3f}, "
              f"test cells {res[t_end]['right'][1]:.3f}")
    print(f"{'':{len(name) + 4}}{'mean':>5} {mean['learned']:12.3f} {mean['true']:9.3f} {mean['stitch']:10.3f}")
    return mean


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=2000)
    args = ap.parse_args()
    results, means, forces = {}, {}, {}

    # 6a: quadratic V, beta = 0.5; train without t = 1; test on a fresh independent dataset
    data = quadratic_snapshots(torch.Generator().manual_seed(0))
    train = {t: Y for t, Y in data.items() if t != 1.0}
    test = quadratic_snapshots(torch.Generator().manual_seed(1))
    model = build_nn_model(train)
    fit(model, train, epochs=args.epochs, log_every=0)
    results["6a quadratic"] = rollout_w1(model, train[0.0], test, 1e-3, 0.5, P.grad_quadratic)
    means["6a quadratic"] = report("6a quadratic", results["6a quadratic"], unseen=(1.0,))
    Yt = torch.cat(list(test.values()))
    forces["6a quadratic"] = force_error(model, P.grad_quadratic, Yt,
                                         torch.linspace(Yt.min().item(), Yt.max().item(), 401)[:, None])

    # 6b: double-well, beta = 1; start from the test cells at t = 0, compare with test cells
    regimes, test = make_regimes()
    for name, tr in regimes.items():
        model, _ = train_regime(tr, args.epochs)
        key = f"6b {name}"
        results[key] = rollout_w1(model, test[0.0], test, 1e-2, 1.0, P.grad_double_well_1d)
        means[key] = report(key, results[key], unseen=[t for t in test if t not in tr])
        Yt = torch.cat(list(test.values()))
        forces[key] = force_error(model, P.grad_double_well_1d, Yt, torch.linspace(-10, 10, 401)[:, None])

    print(f"\n{'run':15} {'W1 learned SDE':>15} {'W1 true SDE':>12} {'W1 stitching':>13}   (means over times)")
    for k, m in means.items():
        print(f"{k:15} {m['learned']:15.3f} {m['true']:12.3f} {m['stitch']:13.3f}")

    print("\nForce recovery (second test):")
    for k, f in forces.items():
        report_force(k, f, 0.5 if k.startswith("6a") else 1.0)

    fig, ax = plt.subplots(1, len(results), figsize=(4 * len(results), 3.5))
    for a, (k, res) in zip(ax, results.items()):
        ts = list(res)
        for col, lab, st in (("learned", "learned SDE ($V_\\theta$, $c_H$)", "o-"),
                             ("true", "true SDE (floor)", "k--"), ("stitch", "stitching KDE", "s:")):
            a.plot(ts, [res[t][col] for t in ts], st, ms=3, label=lab)
        a.set(title=k, xlabel="t", ylabel="$W_1$ to test cells")
    ax[0].legend(fontsize=8)
    fig.tight_layout()
    out = ROOT / "figures" / "demo_rollout.png"
    fig.savefig(out, dpi=130)
    print(f"saved {out}")
