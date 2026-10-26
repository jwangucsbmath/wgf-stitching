"""Step 2 — the model curve: a Gaussian KDE over particles, its score and its velocity field.

    rho(y) = sum_k w_k phi_h(y - x_k),   phi_h(z) = prod_d (2 pi h_d^2)^{-1/2} exp(-z_d^2 / (2 h_d^2))   (Eq. 8)

Shapes: centres (..., N, D), query points (M, D), bandwidth h (D,), weights w (N,) with sum 1.
Nothing is learned here.
"""

import math

import torch


def _log_kernel_matrix(y, centres, h, w):
    # log(w_k phi_h(y_m - x_k)) for all m, k: (..., M, N)
    z = (y[..., :, None, :] - centres[..., None, :, :]) / h
    norm = torch.log(h).sum() + 0.5 * h.shape[-1] * math.log(2 * math.pi)
    return torch.log(w) - 0.5 * (z**2).sum(-1) - norm


def log_density(y, centres, h, w):
    """log rho(y) = logsumexp_k [log w_k + log phi_h(y - x_k)]   (Eq. 8). Returns (M,)."""
    return torch.logsumexp(_log_kernel_matrix(y, centres, h, w), dim=-1)


def score(y, centres, h, w):
    """grad log rho(y) = sum_k r_k(y) (x_k - y) / h^2,  r_k(y) = w_k phi_h(y - x_k) / rho(y)   (Eq. 36).

    r_k is a softmax over k; division by h^2 is elementwise. Returns (M, D).
    """
    r = torch.softmax(_log_kernel_matrix(y, centres, h, w), dim=-1)  # (M, N)
    return (r @ centres - y) / h**2


def score_at_centres(centres, h, w, loo=False):
    """Score at every centre x_j, for one slice (N, D) or many slices (K, N, D) at once.

    loo=False: the exact score of rho (Eq. 36), the term k = j included.
    loo=True:  leave-one-out (Eq. 37), k = j removed from numerator and denominator:
               sum_{k != j} w_k phi_h(x_j - x_k)(x_k - x_j) / h^2  /  sum_{l != j} w_l phi_h(x_j - x_l)
    """
    N = centres.shape[-2]
    logits = _log_kernel_matrix(centres, centres, h, w)  # (..., N_j, N_k)
    if loo:
        if N < 2:
            raise ValueError("the leave-one-out score needs at least two particles")
        eye = torch.eye(N, dtype=torch.bool, device=centres.device)
        logits = logits.masked_fill(eye, float("-inf"))
    r = torch.softmax(logits, dim=-1)
    return (r @ centres - centres) / h**2


def nw_velocity(y, centres, vel, h, w):
    """Nadaraya–Watson velocity v(y) = sum_k r_k(y) xdot_k   (Eq. 9 / 35).

    With this v the KDE with moving centres satisfies the continuity equation exactly (Claim 1).
    vel: (N, D) particle velocities. Returns (M, D).
    """
    r = torch.softmax(_log_kernel_matrix(y, centres, h, w), dim=-1)
    return r @ vel


def silverman(x):
    """Silverman's rule per dimension: h_d = sigma_d (4 / ((D + 2) n))^{1/(D+4)}. x: (n, D) -> (D,)."""
    n, D = x.shape
    return x.std(0) * (4.0 / ((D + 2) * n)) ** (1.0 / (D + 4))


def sample(centres, h, w, M, gen=None):
    """M draws from rho: particle k with probability w_k, then x_k + h * eps.

    Particles are chosen by systematic resampling (u_m = (m + U) / M), so each particle's share
    of the draws is within 1/M of w_k; i.i.d. choices would add sampling noise to metrics.
    """
    like = dict(dtype=centres.dtype, device=centres.device)
    u = (torch.arange(M, **like) + torch.rand(1, generator=gen, **like)) / M
    k = torch.searchsorted(torch.cumsum(w, 0).to(**like), u).clamp(max=len(w) - 1)
    return centres[k] + h * torch.randn(M, centres.shape[-1], generator=gen, **like)
