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

Run from the project root:  .venv/bin/python scripts/demo_split.py [--epochs 2000] [--split interleaved]
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
from wgf.baselines import ot_interpolate_snapshots  # noqa: E402
from wgf.datasets import SPLITS, double_well_regimes  # noqa: E402
from wgf.init import ot_straight_lines  # noqa: E402
from wgf.kde import sample, silverman  # noqa: E402
from wgf.metrics import w1_1d, w2_1d  # noqa: E402
from wgf.model import FixedV, Stitching  # noqa: E402
from wgf.nn import MLP  # noqa: E402
from wgf.sim import simulate  # noqa: E402
from wgf.train import fit  # noqa: E402

torch.set_default_dtype(torch.float64)
BETA = 1.0
COLS = ["OT interp", "V_theta KDE", "V_theta SDE", "V* KDE", "V* SDE"]


def train_model(train, epochs, true_V=False, seed=0, K=50, N=100):
    times = sorted(train)
    t_grid = torch.linspace(times[0], times[-1], K)
    X = ot_straight_lines(train[times[0]], train[times[-1]], t_grid, N, torch.Generator().manual_seed(seed))
    torch.manual_seed(seed)
    if true_V:
        model = Stitching(t_grid, X, silverman(train[times[0]]), FixedV(P.double_well_1d), c_H=BETA)
    else:
        model = Stitching(t_grid, X, silverman(train[times[0]]), MLP(1), c_H=1.0, learn_c_H=True)
    fit(model, train, epochs=epochs, log_every=0)
    return model


@torch.no_grad()
def predictions(train, test, m_theta, m_true, seed=0):
    """{column: {t: (M, 1) predicted cloud}} at every test time."""
    times, M = sorted(test), len(next(iter(test.values())))
    g = torch.Generator().manual_seed(seed + 10)
    t0 = min(train)
    X0 = train[t0][torch.randint(len(train[t0]), (M,), generator=g)] if len(train[t0]) != M else train[t0]
    learned = lambda x: m_theta.c_V * m_theta.grad_V(x)
    roll_theta = dict(simulate(X0, [t0] + times, 1e-2, float(m_theta.c_H), learned,
                               gen=torch.Generator().manual_seed(seed + 1)))
    roll_true = dict(simulate(X0, [t0] + times, 1e-2, BETA, P.grad_double_well_1d,
                              gen=torch.Generator().manual_seed(seed + 2)))
    kde = lambda m, t: sample(m.centres_at(t), m.h, m.w, M, g)
    return {
        "OT interp": {t: ot_interpolate_snapshots(train, t, M) for t in times},
        "V_theta KDE": {t: kde(m_theta, t) for t in times},
        "V_theta SDE": {t: roll_theta[t] for t in times},
        "V* KDE": {t: kde(m_true, t) for t in times},
        "V* SDE": {t: roll_true[t] for t in times},
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=2000)
    ap.add_argument("--split", default="interleaved", choices=list(SPLITS))
    args = ap.parse_args()
    regimes, test = double_well_regimes(torch.Generator().manual_seed(0), split=args.split)
    print(f"split {args.split}: train idx {SPLITS[args.split]}, {len(test)} test times, "
          f"{len(next(iter(test.values())))} test cells (disjoint trajectories)")

    scores = {}
    for name, train in regimes.items():
        m_theta = train_model(train, args.epochs)
        m_true = train_model(train, args.epochs, true_V=True)
        pred = predictions(train, test, m_theta, m_true)
        scores[name] = {c: {t: (w1_1d(p[t], test[t]), w2_1d(p[t], test[t])) for t in test} for c, p in pred.items()}
        Yt = torch.cat(list(test.values()))
        gl, gt = (m_theta.c_V * m_theta.grad_V(Yt)).detach(), P.grad_double_well_1d(Yt)
        print(f"{name}: c_H {float(m_theta.c_H):.3f}, force rel. L2 error on test cells "
              f"{((gl - gt).norm() / gt.norm()).item():.3f}", flush=True)

    for k, label in ((1, "W2"), (0, "W1")):
        print(f"\n{label}, mean over the {len(test)} test times")
        print(f"{'regime':9}" + "".join(f"{c:>13}" for c in COLS))
        for name, sc in scores.items():
            print(f"{name:9}" + "".join(f"{sum(v[k] for v in sc[c].values()) / len(test):13.3f}" for c in COLS))

    fig, ax = plt.subplots(1, len(scores), figsize=(4.5 * len(scores), 3.6), sharey=True)
    for a, (name, sc) in zip(ax, scores.items()):
        ts = sorted(test)
        for c, st in zip(COLS, ("k:", "o-", "s-", "^--", "k--")):
            a.plot(ts, [sc[c][t][1] for t in ts], st, ms=3, label=c)
        a.set(title=f"{name} ({args.split})", xlabel="test time t")
    ax[0].set_ylabel("$W_2$ to test cells")
    ax[0].legend(fontsize=8)
    fig.tight_layout()
    out = ROOT / "figures" / f"demo_split_{args.split}.png"
    fig.savefig(out, dpi=130)
    print(f"saved {out}")
