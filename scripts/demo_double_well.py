"""Step 6b: V_theta on the 1-D double-well in three observation regimes.

Data: V = (|x| - 7)^2 / 2, beta = 1, X0 ~ N(0, 1) (our choice: mass splits into both wells),
one paired run of 2000 cells, 20 snapshots on [0, 4]. Cells are split 50/50 into train and test.
Regimes (training data only):
  dense    - all 20 snapshots, 1000 cells each
  terminal - first and last snapshot only
  trickle  - all 20 snapshots, 10 cells each (a different random subset at every time)
Model (paper preset): N = 100, K = 50 on [0, 4], 2000 epochs, lambda = 1, trapezoid, MLP V_theta,
c_H learned from 1.0, bandwidth from Silverman on the first training snapshot, full score.
Report: potential R^2 (centred, on the test cells at all times) and W2 between KDE samples and
the 1000 test cells per snapshot.

Run from the project root:  .venv/bin/python scripts/demo_double_well.py [--epochs 2000]
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
from wgf import potentials as P  # noqa: E402
from wgf.datasets import double_well_snapshots  # noqa: E402
from wgf.init import ot_straight_lines  # noqa: E402
from wgf.kde import sample, silverman  # noqa: E402
from wgf.metrics import potential_r2, w2_1d  # noqa: E402
from wgf.model import Stitching  # noqa: E402
from wgf.nn import MLP  # noqa: E402
from wgf.train import fit  # noqa: E402

torch.set_default_dtype(torch.float64)


def make_regimes(seed=0):
    g = torch.Generator().manual_seed(seed)
    snaps = double_well_snapshots(g, M=2000)
    perm = torch.randperm(2000, generator=g)
    tr, te = perm[:1000], perm[1000:]
    times = list(snaps)
    test = {t: snaps[t][te] for t in times}
    dense = {t: snaps[t][tr] for t in times}
    terminal = {times[0]: dense[times[0]], times[-1]: dense[times[-1]]}
    trickle = {t: dense[t][torch.randperm(1000, generator=g)[:10]] for t in times}
    return {"dense": dense, "terminal": terminal, "trickle": trickle}, test


def train_regime(train, epochs, seed=0, K=50, N=100, loo=False, **fit_kw):
    """fit_kw: lr, lam, scheme, pooled for wgf.train.fit."""
    times = sorted(train)
    gen = torch.Generator().manual_seed(seed)
    t_grid = torch.linspace(times[0], times[-1], K)
    X = ot_straight_lines(train[times[0]], train[times[-1]], t_grid, N, gen)
    torch.manual_seed(seed)
    model = Stitching(t_grid, X, silverman(train[times[0]]), MLP(1), c_H=1.0, learn_c_H=True, loo=loo)
    hist = fit(model, train, epochs=epochs, log_every=0, **fit_kw)
    return model, hist


@torch.no_grad()
def evaluate(model, train, test, seed=0):
    g = torch.Generator().manual_seed(seed + 100)
    w2 = {t: w2_1d(sample(model.centres_at(t), model.h, model.w, len(Y), g), Y) for t, Y in test.items()}
    Y_all = torch.cat(list(test.values()))
    r2 = potential_r2(lambda x: model.c_V * model.V(x), P.double_well_1d, Y_all)
    xb = torch.linspace(-3, 3, 601)[:, None]
    gV = model.c_V * model.grad_V(xb)[:, 0]
    barrier = xb[:-1, 0][(gV[:-1] > 0) & (gV[1:] <= 0)].tolist()
    unseen = [t for t in test if t not in train]
    return dict(r2=r2, w2=w2, w2_mean=sum(w2.values()) / len(w2),
                w2_unseen=(sum(w2[t] for t in unseen) / len(unseen)) if unseen else float("nan"),
                c_H=float(model.c_H), h=model.h.item(), barrier=barrier)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=2000)
    args = ap.parse_args()
    regimes, test = make_regimes()
    results, models = {}, {}
    for name, train in regimes.items():
        models[name], hist = train_regime(train, args.epochs)
        results[name] = evaluate(models[name], train, test)
        print(f"{name:9s} done in {hist['seconds']:.0f} s")

    print(f"\n{'regime':9} {'R2(V)':>7} {'c_H':>6} {'h':>6} {'W2 mean':>8} {'W2 unseen':>10}  barrier (true 0)")
    for name, r in results.items():
        print(f"{name:9} {r['r2']:7.3f} {r['c_H']:6.3f} {r['h']:6.3f} {r['w2_mean']:8.3f} {r['w2_unseen']:10.3f}  "
              f"{[round(b, 2) for b in r['barrier']]}")

    with torch.no_grad():
        fig, ax = plt.subplots(1, 2, figsize=(13, 4))
        xs = torch.linspace(-11, 11, 400)[:, None]
        Y_all = torch.cat(list(test.values()))
        ax[0].plot(xs, P.double_well_1d(xs), "k--", label="$V^\\star$")
        for name, m in models.items():
            v = m.c_V * m.V(xs)
            ax[0].plot(xs, v - m.V(Y_all).mean() + P.double_well_1d(Y_all).mean(),
                       label=f"{name}, $R^2$={results[name]['r2']:.3f}")
        ax[0].set(ylim=(-1, 30), xlabel="x", title="learned potential (shifted to match on the data)"); ax[0].legend()
        for name, r in results.items():
            ax[1].plot(list(r["w2"]), list(r["w2"].values()), "o-", ms=3, label=name)
        ax[1].set(xlabel="t", ylabel="$W_2$(model KDE, test cells)", title="held-out $W_2$ per snapshot"); ax[1].legend()
        fig.tight_layout()
        out = ROOT / "figures" / "demo_double_well.png"
        fig.savefig(out, dpi=130)
        print(f"saved {out}")
