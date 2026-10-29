"""Step 4 demo: R_vel vs dt on the exact Gaussian flow (log-log, Euler and trapezoid), the perturbed
path's closed-form limit, and why SDE sample paths must never go into R_vel.

Run from the project root:  .venv/bin/python experiments/validation/component_checks/residual.py
"""

import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
from wgf.potentials import gaussian_flow_path, grad_quadratic  # noqa: E402
from wgf.residual import residual  # noqa: E402
from wgf.sim import simulate  # noqa: E402

torch.set_default_dtype(torch.float64)
BETA, M0, S0, T, N = 0.5, 2.0, 0.5, 2.0, 2000
gen = torch.Generator().manual_seed(0)
x0 = M0 + S0 * torch.randn(N, 1, generator=gen)
w = torch.full((N,), 1 / N)
Ks = [11, 21, 41, 81, 161]

rows = {"euler": [], "trapezoid": [], "perturbed": [], "sde": []}
for K in Ks:
    t = torch.linspace(0, T, K)
    X, g = gaussian_flow_path(x0, t, BETA, M0, S0)
    for scheme in ("euler", "trapezoid"):
        rows[scheme].append(residual(X, t, w, g=g, scheme=scheme).item())
    Xp, gp = gaussian_flow_path(x0, t, BETA, M0, S0, rate=0.5)
    rows["perturbed"].append(residual(Xp, t, w, g=gp).item())
    # SDE sample paths on the same grid, with the exact density force: R ~ 2 beta D (K - 1)
    Xs = torch.stack([X_ for _, X_ in simulate(x0, t.tolist(), 1e-3, BETA, grad_quadratic, gen=gen)])
    _, g_exact = gaussian_flow_path(x0, t, BETA, M0, S0)
    m = (M0 * torch.exp(-t))[:, None, None]
    s2 = (BETA + (S0**2 - BETA) * torch.exp(-2 * t))[:, None, None]
    rows["sde"].append(residual(Xs, t, w, g=Xs - BETA * (Xs - m) / s2).item())

limit = 0.25 * M0**2 * (1 - math.exp(-T))
dts = [T / (K - 1) for K in Ks]
print(f"{'K':>5} {'dt':>8} {'euler':>10} {'trapezoid':>10} {'perturbed':>10} {'SDE paths':>10} {'2bD(K-1)':>9}")
for i, K in enumerate(Ks):
    print(f"{K:5d} {dts[i]:8.4f} {rows['euler'][i]:10.2e} {rows['trapezoid'][i]:10.2e} "
          f"{rows['perturbed'][i]:10.4f} {rows['sde'][i]:10.1f} {2 * BETA * (K - 1):9.0f}")
print(f"perturbed limit (1-r)^2 m0^2 (1-e^(-2rT))/(2r) = {limit:.4f}")

fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
ax[0].loglog(dts, rows["euler"], "o-", label="Euler (slope 2)")
ax[0].loglog(dts, rows["trapezoid"], "s-", label="trapezoid (slope 4)")
ref = lambda p, y0: [y0 * (d / dts[0]) ** p for d in dts]
ax[0].loglog(dts, ref(2, rows["euler"][0]), "k:", lw=0.8)
ax[0].loglog(dts, ref(4, rows["trapezoid"][0]), "k:", lw=0.8)
ax[0].set(xlabel="Δt", ylabel="$R_{vel}$", title="Exact Gaussian flow: $R_{vel}$ → 0"); ax[0].legend()
ax[1].plot(Ks, rows["sde"], "o-", label="SDE sample paths")
ax[1].plot(Ks, [2 * BETA * (K - 1) for K in Ks], "k--", label="2βD(K−1)")
ax[1].plot(Ks, rows["perturbed"], "s-", label=f"perturbed path (→ {limit:.4f})")
ax[1].set(xlabel="K", ylabel="$R_{vel}$", title="Never feed SDE paths into $R_{vel}$", yscale="log"); ax[1].legend()
fig.tight_layout()
out = HERE / "figures" / "residual.png"
fig.savefig(out, dpi=130)
print(f"saved {out}")
