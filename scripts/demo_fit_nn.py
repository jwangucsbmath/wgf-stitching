"""Step 6a demo: same data and pipeline as step 5, with V_a replaced by the network V_theta.

Run from the project root:  .venv/bin/python scripts/demo_fit_nn.py [--epochs 2000]
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
from wgf import potentials as P  # noqa: E402
from wgf.datasets import quadratic_snapshots  # noqa: E402
from wgf.init import ot_straight_lines  # noqa: E402
from wgf.kde import silverman  # noqa: E402
from wgf.metrics import potential_r2  # noqa: E402
from wgf.model import Stitching  # noqa: E402
from wgf.nn import MLP  # noqa: E402
from wgf.train import fit  # noqa: E402

torch.set_default_dtype(torch.float64)
BETA, M0, S0 = 0.5, 2.0, 0.5
M1, V1 = M0 * math.exp(-1), BETA + (S0**2 - BETA) * math.exp(-2)


def build_nn_model(train, seed=0, K=41, N=100):
    gen = torch.Generator().manual_seed(seed)
    t_grid = torch.linspace(0.0, 2.0, K)
    X = ot_straight_lines(train[0.0], train[2.0], t_grid, N, gen)
    torch.manual_seed(seed)  # network init
    return Stitching(t_grid, X, silverman(train[0.0]), MLP(1), c_H=BETA)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=2000)
    args = ap.parse_args()
    data = quadratic_snapshots(torch.Generator().manual_seed(0))
    train = {t: Y for t, Y in data.items() if t != 1.0}
    model = build_nn_model(train)
    hist = fit(model, train, epochs=args.epochs, log_every=250)
    Y = torch.cat(list(train.values()))
    r2 = potential_r2(lambda x: model.c_V * model.V(x), P.quadratic, Y)
    mean, var = model.moments_at(1.0)
    ok = r2 > 0.95 and abs(mean.item() - M1) < 0.05 and abs(var.item() / V1 - 1) < 0.1
    print(f"\n({hist['seconds']:.0f} s) potential R^2 = {r2:.4f}   held-out t=1: mean {mean.item():.4f} "
          f"(exact {M1:.4f})   var {var.item():.4f} (exact {V1:.4f})")
    print("targets R^2>0.95, mean within 0.05, var within 10%:", "MET" if ok else "MISSED")

    with torch.no_grad():
        xs = torch.linspace(Y.min().item(), Y.max().item(), 200)[:, None]
        vt, vs = model.c_V * model.V(xs), P.quadratic(xs)
        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        ax[0].plot(xs, vs - vs.mean(), "k--", label="$x^2/2$ (centred)")
        ax[0].plot(xs, vt - vt.mean(), label=f"$c_V V_\\theta$ (centred), $R^2$={r2:.3f}")
        ax[0].set(title="learned potential on the data range", xlabel="x"); ax[0].legend()
        ax[1].plot(hist["nll"], label="NLL"); ax[1].plot(hist["R"], label="$R_{vel}$")
        ax[1].set(yscale="log", xlabel="epoch", title="loss terms"); ax[1].legend()
        fig.tight_layout()
        out = ROOT / "figures" / "demo_fit_nn.png"
        fig.savefig(out, dpi=130)
        print(f"saved {out}")
