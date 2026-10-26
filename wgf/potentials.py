"""Test potentials V(x) and interaction kernels W(r), with closed-form gradients.

Each energy is a function (..., D) -> (...); each grad_* returns (..., D).
`autograd_grad` builds the gradient of any energy (e.g. an MLP) by autodiff,
and is used in the tests to check the closed forms.
"""

import math
from typing import Callable

import torch

# ── confining potentials V ────────────────────────────────────────────────────


def quadratic(x, a=1.0):
    """V = a/2 |x|^2 (Ornstein–Uhlenbeck when combined with noise)."""
    return 0.5 * a * (x**2).sum(-1)


def grad_quadratic(x, a=1.0):
    return a * x


def double_well_1d(x, c=7.0):
    """V = 1/2 (|x| - c)^2, the repo's 1-D double-well. Kink at x = 0."""
    return 0.5 * (x.abs().sum(-1) - c) ** 2


def grad_double_well_1d(x, c=7.0):
    # sign(0) = 0 picks the symmetric subgradient at the kink
    return x - c * torch.sign(x)


def wavy_valley(x, a=0.6, b=0.3):
    """V = a (x2 - sin(pi x1 / 2))^2 - b x1 (paper sec. 5.1 and App. E: K = 0.6, tau = 0.3).

    Our repo notes (00_reference_repo_structure.md) list b = 0.5; we follow the paper.
    """
    x1, x2 = x[..., 0], x[..., 1]
    return a * (x2 - torch.sin(0.5 * math.pi * x1)) ** 2 - b * x1


def grad_wavy_valley(x, a=0.6, b=0.3):
    x1, x2 = x[..., 0], x[..., 1]
    r = x2 - torch.sin(0.5 * math.pi * x1)
    d1 = -2 * a * r * 0.5 * math.pi * torch.cos(0.5 * math.pi * x1) - b
    d2 = 2 * a * r
    return torch.stack([d1, d2], dim=-1)


# ── interaction kernels W (even, smooth, so grad W(0) = 0) ─────────────────────


def quadratic_interaction(r, a=1.0):
    """W(r) = a/2 |r|^2: attraction toward the mean, sum_j grad W(x_i - x_j)/N = a (x_i - mean)."""
    return 0.5 * a * (r**2).sum(-1)


def grad_quadratic_interaction(r, a=1.0):
    return a * r


def gaussian_interaction(r, a=1.0, s=1.0):
    """W(r) = -a exp(-|r|^2 / (2 s^2)): short-range attraction (a > 0) or repulsion (a < 0)."""
    return -a * torch.exp(-(r**2).sum(-1) / (2 * s**2))


def grad_gaussian_interaction(r, a=1.0, s=1.0):
    return (a / s**2) * torch.exp(-(r**2).sum(-1, keepdim=True) / (2 * s**2)) * r


# ── autodiff gradient of any energy ────────────────────────────────────────────


def autograd_grad(energy: Callable[[torch.Tensor], torch.Tensor], create_graph=False):
    """Return x -> grad energy(x), for energies that act row-wise on (..., D).

    Set create_graph=True when the gradient sits inside a loss that is itself
    differentiated (e.g. training V_theta through the residual).
    """

    def grad_fn(x):
        track = torch.is_grad_enabled() and x.requires_grad
        with torch.enable_grad():
            x_ = x if track else x.detach().requires_grad_(True)
            (g,) = torch.autograd.grad(energy(x_).sum(), x_, create_graph=create_graph)
        return g

    return grad_fn


# ── exact Gaussian (OU) flow, used to test the residual (plan step 4) ───────────────


def gaussian_flow_path(x0, t_grid, beta, m0, s0, rate=1.0):
    """Particle paths of the exact Gaussian flow of V = x^2/2 with temperature beta, and the force.

        x_k(t) = m(t) + s(t)/s0 (x_k(0) - m0),   m(t) = m0 e^{-rate t},   s(t)^2 = beta + (s0^2 - beta) e^{-2t}
        g(x, t) = x - beta (x - m(t)) / s(t)^2   (grad V + beta grad log rho_t, rho_t = N(m(t), s(t)^2))

    With rate = 1 the paths satisfy xdot = -g exactly. Any other rate gives the "perturbed" path of
    the plan, whose own-density force leaves v + g = (1 - rate) m(t) for every particle.
    x0: (N, D) samples of N(m0, s0^2); t_grid: (K,). Returns X (K, N, D) and g (K, N, D).
    """
    t = t_grid[:, None, None]
    m = m0 * torch.exp(-rate * t)
    s2 = beta + (s0**2 - beta) * torch.exp(-2 * t)
    X = m + torch.sqrt(s2) / s0 * (x0[None] - m0)
    g = X - beta * (X - m) / s2
    return X, g
