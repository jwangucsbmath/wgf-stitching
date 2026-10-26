"""Steps 5-6 — the stitching model: learnable particle paths X plus an energy.

    rho_t = KDE of the centres X[j] (Eq. 8),   F = c_H int rho log rho + c_V int V drho + (c_W/2) iint W drho drho

| parameter | shape | trained? |
|---|---|---|
| X        | (K, N, D) particle positions on t_grid              | yes |
| V        | any energy module (..., D) -> (...): QuadraticV (step 5) or an MLP (step 6) | yes |
| h        | (D,) bandwidth, softplus, init given (Silverman)    | yes |
| c_H      | entropy weight, softplus                            | optional (`learn_c_H`) |
| c_V      | fixed 1: the potential carries the scale            | no |
| W, c_W   | optional interaction (step 7)                       | yes if given |
| w, t_grid| uniform 1/N, fixed grid                             | no |
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .energy import force, grad_x


def inv_softplus(y):
    y = torch.as_tensor(y, dtype=torch.get_default_dtype())
    return y + torch.log(-torch.expm1(-y))  # stable log(e^y - 1)


class QuadraticV(nn.Module):
    """V_a(x) = a |x|^2 / 2 with one unconstrained scalar a, init 0 (V = 0 at start, like a zero-init net)."""

    def __init__(self, a0=0.0):
        super().__init__()
        self.a = nn.Parameter(torch.tensor(float(a0)))

    def forward(self, x):
        return 0.5 * self.a * (x**2).sum(-1)


class FixedV(nn.Module):
    """A known potential with no parameters (reference model with V fixed to the truth V*)."""

    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x)


class Stitching(nn.Module):
    def __init__(self, t_grid, X_init, h0, V, c_H=0.5, learn_c_H=False, c_V=1.0, W=None, c_W=1.0, loo=False):
        super().__init__()
        K, N, D = X_init.shape
        if len(t_grid) != K:
            raise ValueError("t_grid and X_init disagree on K")
        self.register_buffer("t_grid", torch.as_tensor(t_grid, dtype=X_init.dtype))
        self.register_buffer("w", torch.full((N,), 1.0 / N, dtype=X_init.dtype))
        self.X = nn.Parameter(X_init.clone())
        self.raw_h = nn.Parameter(inv_softplus(torch.as_tensor(h0, dtype=X_init.dtype)))
        if learn_c_H:
            self.raw_c_H = nn.Parameter(inv_softplus(c_H))
        else:
            self.register_buffer("raw_c_H", inv_softplus(c_H))
        self.c_V = c_V
        self.V = V
        self.W = W
        self.raw_c_W = nn.Parameter(inv_softplus(c_W)) if W is not None else None
        self.loo = loo

    h = property(lambda self: F.softplus(self.raw_h))
    c_H = property(lambda self: F.softplus(self.raw_c_H))
    c_W = property(lambda self: F.softplus(self.raw_c_W) if self.W is not None else 0.0)

    def grad_V(self, x):
        return grad_x(self.V, x)

    def force(self, X=None):
        """g at every particle of every slice (K, N, D), Eq. 7."""
        X = self.X if X is None else X
        return force(X, self.h, self.w, self.grad_V, self.c_H, self.c_V, self.c_W, self.W, loo=self.loo)

    def centres_at(self, s):
        """Centres at time s (N, D): the grid node if s is one, else linear interpolation."""
        t = self.t_grid
        tol = 1e-9 * float(t[-1] - t[0])
        if not float(t[0]) - tol <= float(s) <= float(t[-1]) + tol:
            raise ValueError(f"time {s} is outside the model grid [{float(t[0])}, {float(t[-1])}]")
        j = int(torch.clamp(torch.searchsorted(t, torch.tensor(float(s), dtype=t.dtype), right=True) - 1, 0, len(t) - 2))
        a = (float(s) - float(t[j])) / float(t[j + 1] - t[j])
        if abs(a) < 1e-9:
            return self.X[j]
        if abs(a - 1) < 1e-9:
            return self.X[j + 1]
        return (1 - a) * self.X[j] + a * self.X[j + 1]

    @torch.no_grad()
    def moments_at(self, s):
        """Mean and per-dimension variance of the model KDE at time s: particle variance + h^2."""
        c = self.centres_at(s)
        mean = self.w @ c
        var = self.w @ (c - mean) ** 2 + self.h**2
        return mean, var
