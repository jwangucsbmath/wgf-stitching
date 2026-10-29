"""[DATA] stage 1 demo: the forward simulator against exact answers and on the paper's systems.

Left: Gaussian (OU) flow, V = x^2/2, beta = 0.5, start N(4, 0.3^2). Histograms of the
cloud at several times on top of the exact density N(m(t), s^2(t)) with
m(t) = m0 e^{-t}, s^2(t) = beta + (s0^2 - beta) e^{-2t}. If these line up, the simulator is done.
Middle: 1-D double-well V = 1/2(|x| - 7)^2, beta = 1, 20 snapshots on [0, 4].
Right: 2-D wavy valley (paper Fig. 1 / App. E), beta = 0.00625, snapshots {0, 5, 10, 20, 30},
init N([-3, 1], 0.1^2). Noise uses sigma = sqrt(2 beta) (not the repo's sqrt(2) c_H).

Run from the project root:  .venv/bin/python experiments/validation/component_checks/forward.py
"""

import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from wgf import potentials as P  # noqa: E402
from wgf.sim import simulate  # noqa: E402

torch.set_default_dtype(torch.float64)
g = torch.Generator().manual_seed(0)
out_dir = Path(__file__).resolve().parent / "figures"
out_dir.mkdir(exist_ok=True)
cmap = plt.get_cmap("viridis")

# Gaussian flow with an exact solution
beta_ou, m0, s0 = 0.5, 4.0, 0.3
t_ou = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0]
X0 = m0 + s0 * torch.randn(20_000, 1, generator=g)
traj_ou = torch.stack([X for _, X in simulate(X0, t_ou, 0.01, beta_ou, P.grad_quadratic, gen=g)])

# double-well
t_dw = torch.linspace(0, 4, 20).tolist()
X0 = torch.randn(1000, 1, generator=g)
traj_dw = torch.stack([X for _, X in simulate(X0, t_dw, 0.01, 1.0, P.grad_double_well_1d, gen=g)])

# wavy valley
t_wv = [0.0, 5.0, 10.0, 20.0, 30.0]
X0 = torch.tensor([-3.0, 1.0]) + 0.1 * torch.randn(200, 2, generator=g)
traj_wv = torch.stack([X for _, X in simulate(X0, t_wv, 0.01, 0.00625, P.grad_wavy_valley, gen=g)])

fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(18, 4.5))

xs = torch.linspace(-3, 6, 400)
print("Gaussian flow: sample vs exact moments")
for k, t in enumerate(t_ou):
    c = cmap(k / (len(t_ou) - 1))
    m_t = m0 * math.exp(-t)
    v_t = beta_ou + (s0**2 - beta_ou) * math.exp(-2 * t)
    ax0.hist(traj_ou[k, :, 0].numpy(), bins=80, range=(-3, 6), density=True, histtype="step", color=c)
    ax0.plot(xs, torch.exp(-((xs - m_t) ** 2) / (2 * v_t)) / math.sqrt(2 * math.pi * v_t), "--", color=c,
             lw=1.2, label=f"t = {t:g}")
    X = traj_ou[k, :, 0]
    print(f"  t={t:4g}  mean {X.mean().item():7.4f} (exact {m_t:7.4f})   var {X.var().item():7.4f} (exact {v_t:7.4f})")
ax0.set(title=f"Gaussian flow, β = {beta_ou}: histograms vs exact density (dashed)", xlabel="x", ylabel="density")
ax0.legend(fontsize=8)

for k, t in enumerate(t_dw):
    ax1.hist(traj_dw[k, :, 0].numpy(), bins=60, range=(-12, 12), density=True,
             histtype="step", color=cmap(k / (len(t_dw) - 1)))
ax1.set(title="1-D double-well, β = 1 (colour = time 0→4)", xlabel="x", ylabel="density")

xs = torch.linspace(-4, 12, 200)
ys = torch.linspace(-2.5, 2.5, 120)
XX, YY = torch.meshgrid(xs, ys, indexing="xy")
VV = P.wavy_valley(torch.stack([XX, YY], -1))
ax2.contour(XX.numpy(), YY.numpy(), VV.numpy(), levels=25, colors="0.8", linewidths=0.6)
for k, t in enumerate(t_wv):
    pts = traj_wv[k].numpy()
    ax2.scatter(pts[:, 0], pts[:, 1], s=4, color=cmap(k / (len(t_wv) - 1)), label=f"t = {t:g}")
for i in range(0, 200, 20):  # a few paired particles, joined by straight chords between snapshots
    ax2.plot(traj_wv[:, i, 0].numpy(), traj_wv[:, i, 1].numpy(), color="0.4", lw=0.5)
ax2.set(title="Wavy valley, β = 0.00625", xlabel="x₁", ylabel="x₂")
ax2.legend(markerscale=3, fontsize=8)

fig.tight_layout()
path = out_dir / "forward.png"
fig.savefig(path, dpi=130)
print(f"saved {path}")
print("double-well: fraction at x > 0 at t=4:", (traj_dw[-1] > 0).double().mean().item())
print("wavy valley: mean position per snapshot:")
for t, X in zip(t_wv, traj_wv):
    print(f"  t={t:4g}  mean={X.mean(0).numpy().round(3)}  std={X.std(0).numpy().round(3)}")
