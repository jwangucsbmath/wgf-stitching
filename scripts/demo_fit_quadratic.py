"""Step 5 demo: the full pipeline with V_a(x) = a x^2/2 (no network). Recover a = 1.

Data: V = x^2/2, beta = 0.5, X0 ~ N(2, 0.5^2), unpaired snapshots at {0, 0.5, 1, 1.5, 2} with 1000
cells; train on {0, 0.5, 1.5, 2}, hold out t = 1. Model: K = 41 on [0, 2], N = 100, c_H = 0.5 fixed.
Also runs the plan's ablations: lambda = 0, learned c_H, Euler vs trapezoid.

Run from the project root:  .venv/bin/python scripts/demo_fit_quadratic.py [--epochs 2000] [--no-ablations]
"""

import argparse
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from wgf.datasets import quadratic_snapshots  # noqa: E402
from wgf.init import ot_straight_lines  # noqa: E402
from wgf.kde import silverman  # noqa: E402
from wgf.model import QuadraticV, Stitching  # noqa: E402
from wgf.train import fit  # noqa: E402

torch.set_default_dtype(torch.float64)
BETA, M0, S0 = 0.5, 2.0, 0.5
M1, V1 = M0 * math.exp(-1), BETA + (S0**2 - BETA) * math.exp(-2)  # exact held-out moments at t = 1


def build(train, seed=0, learn_c_H=False, K=41, N=100):
    gen = torch.Generator().manual_seed(seed)
    t_grid = torch.linspace(0.0, 2.0, K)
    X = ot_straight_lines(train[0.0], train[2.0], t_grid, N, gen)
    return Stitching(t_grid, X, silverman(train[0.0]), QuadraticV(), c_H=BETA, learn_c_H=learn_c_H)


def run(train, epochs, lam=1.0, scheme="trapezoid", learn_c_H=False, log_every=100):
    model = build(train, learn_c_H=learn_c_H)
    hist = fit(model, train, epochs=epochs, lam=lam, scheme=scheme, log_every=log_every,
               track={"a": lambda m: m.V.a.item()})
    mean, var = model.moments_at(1.0)
    return model, hist, mean.item(), var.item()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=2000)
    ap.add_argument("--no-ablations", action="store_true")
    args = ap.parse_args()

    data = quadratic_snapshots(torch.Generator().manual_seed(0))
    train = {t: Y for t, Y in data.items() if t != 1.0}
    held = data[1.0]

    model, hist, mean, var = run(train, args.epochs)
    a = model.V.a.item()
    ok = abs(a - 1) < 0.1 and abs(mean - M1) < 0.05 and abs(var / V1 - 1) < 0.1
    print(f"\nmain run ({hist['seconds']:.0f} s): a = {a:.4f}   held-out t=1: mean {mean:.4f} (exact {M1:.4f}, "
          f"data {held.mean().item():.4f})   var {var:.4f} (exact {V1:.4f}, data {held.var().item():.4f})")
    print("targets |a-1|<0.1, |mean-m(1)|<0.05, var within 10%:", "MET" if ok else "MISSED")

    rows = [("trapezoid, lambda=1, c_H fixed", a, mean, var, float(model.c_H))]
    if not args.no_ablations:
        for label, kw in [("lambda = 0", dict(lam=0.0)), ("learn c_H", dict(learn_c_H=True)),
                          ("Euler scheme", dict(scheme="euler"))]:
            m_, h_, mu_, v_ = run(train, args.epochs, log_every=0, **kw)
            rows.append((label, m_.V.a.item(), mu_, v_, float(m_.c_H)))
        print(f"\n{'run':32} {'a':>8} {'c_H':>7} {'mean(1)':>8} {'var(1)':>8}")
        for r in rows:
            print(f"{r[0]:32} {r[1]:8.4f} {r[4]:7.4f} {r[2]:8.4f} {r[3]:8.4f}")

    with torch.no_grad():
        fig, ax = plt.subplots(1, 3, figsize=(16, 4))
        tg = model.t_grid.numpy()
        for k in range(model.X.shape[1]):
            ax[0].plot(tg, model.X[:, k, 0].numpy(), lw=0.5, c="C0", alpha=0.5)
        for t, Y in data.items():
            ax[0].scatter([t] * len(Y), Y[:, 0].numpy(), s=1, c="C1" if t == 1.0 else "C3", alpha=0.2)
        tt = torch.linspace(0, 2, 100)
        ax[0].plot(tt, M0 * torch.exp(-tt), "k--", lw=1, label="exact mean")
        ax[0].set(title="particle paths (blue), train (red), held-out t=1 (orange)", xlabel="t"); ax[0].legend()
        ax[1].plot(hist["nll"], label="NLL"); ax[1].plot(hist["R"], label="$R_{vel}$")
        ax[1].set(yscale="log", title="loss terms", xlabel="epoch"); ax[1].legend()
        ax[2].plot(hist["a"]); ax[2].axhline(1, c="k", ls="--", lw=0.8)
        ax[2].set(title="a vs epoch (truth 1)", xlabel="epoch")
        fig.tight_layout()
        out = ROOT / "figures" / "demo_fit_quadratic.png"
        fig.savefig(out, dpi=130)
        print(f"saved {out}")
