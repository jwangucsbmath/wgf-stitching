"""Steps 5-6 — loss and optimiser.

Loss (Eq. 6, Eqs. 27-28):
    L = sum_s NLL_s + lambda R_vel,   NLL_s = -(1/M_s) sum_{i in s} log rho_{t_s}(y_i)
per snapshot then summed, as the paper's sum_t; pooled=True instead takes one mean over all cells
(each snapshot weighted by its cell count, as the reference code). R_vel is Eq. 13 (wgf.residual).

Optimiser: Adam, lr with cosine decay to 0.05 lr, full batch, one step per epoch:
    m <- b1 m + (1-b1) grad,  v <- b2 v + (1-b2) grad^2,  theta <- theta - lr_n mhat/(sqrt(vhat)+eps)
    lr_n = lr_min + (lr - lr_min)(1 + cos(pi n / epochs))/2
"""

import time

import torch

from .kde import log_density
from .residual import residual


def nll(model, data, pooled=False):
    """data: dict {t: (M_t, D)}. Per-snapshot mean NLL summed over snapshots (or one pooled mean)."""
    terms = [-log_density(Y, model.centres_at(t), model.h, model.w) for t, Y in data.items()]
    if pooled:
        return torch.cat(terms).mean()
    return sum(term.mean() for term in terms)


def loss(model, data, lam=1.0, scheme="trapezoid", pooled=False):
    """Returns (L, NLL, R_vel)."""
    data_term = nll(model, data, pooled)
    R = residual(model.X, model.t_grid, model.w, g=model.force(), scheme=scheme)
    return data_term + lam * R, data_term, R


def fit(model, data, epochs=2000, lr=5e-3, lam=1.0, scheme="trapezoid", pooled=False,
        log_every=100, log=print, track=None):
    """Train in place. track: dict name -> callable(model) -> float, logged every epoch.

    Returns a history dict with one value per epoch of: loss, nll, R and lamR (= lam * R) at the
    parameters before that epoch's step; lr used by that step; h (mean over dimensions), c_H and the
    tracked quantities after it. Plus the run time in seconds.
    """
    track = dict(track or {})
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=0.05 * lr)
    hist = {k: [] for k in ["loss", "nll", "R", "lamR", "lr", "h", "c_H", *track]}
    t0 = time.time()
    for ep in range(epochs):
        opt.zero_grad()
        L, d, R = loss(model, data, lam, scheme, pooled)
        L.backward()
        lr_now = opt.param_groups[0]["lr"]
        opt.step()
        sched.step()
        values = [L.item(), d.item(), R.item(), lam * R.item(), lr_now, model.h.mean().item(), model.c_H.item()]
        values += [fn(model) for fn in track.values()]
        for k, v in zip(hist, values):
            hist[k].append(v)
        if log is not None and log_every and (ep % log_every == 0 or ep == epochs - 1):
            extra = "  ".join(f"{k} {hist[k][-1]:.4f}" for k in track)
            log(f"epoch {ep:5d}  NLL {d.item():8.4f}  R {R.item():9.5f}  h {hist['h'][-1]:.4f}  "
                f"c_H {hist['c_H'][-1]:.4f}  {extra}")
    hist["seconds"] = time.time() - t0
    return hist
