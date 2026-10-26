"""Step 1 — forward simulation of the Wasserstein gradient-flow SDE (interacting particles).

Index convention: superscript n = time step, subscripts i, j = particles.

    X^{n+1}_i = X^n_i - dt * [ grad V(X^n_i) + sum_j w_j grad W(X^n_i - X^n_j) ]
                + sqrt(2 * beta * dt) * eps^n_i

with uniform weights w_j = 1/N by default. This is the particle discretisation of

    d rho/dt = div( rho * grad( V + W * rho ) ) + beta * Laplace(rho),

i.e. the Wasserstein gradient flow of
F(rho) = int V drho + 1/2 iint W(x - y) drho(x) drho(y) + beta * int rho log rho,
for even W. The noise uses sigma = sqrt(2 beta) (see 00_reference_repo_structure.md, sec. 9.3).

Every step is fully vectorised: all forces are computed from the cloud at time n
before any particle moves, so the result never depends on particle order.

[DATA] stage 1: known formulas in, snapshots out, nothing trained.
grad V and grad W are plain callables, so stage 3 (prediction) reuses the same
stepper with a learned grad V_theta. Steps run under torch.no_grad(); a learned
gradient must enable grad itself (see wgf.potentials.autograd_grad, wgf.nn.grad_x).
"""

import math
from typing import Callable, List, Optional, Tuple, Union

import torch

GradFn = Callable[[torch.Tensor], torch.Tensor]


def drift(X: torch.Tensor, grad_V: Optional[GradFn] = None, grad_W: Optional[GradFn] = None,
          w: Optional[torch.Tensor] = None) -> torch.Tensor:
    """grad V(x_i) + sum_j w_j grad W(x_i - x_j) at every particle (the SDE drift is minus this).

    X: (N, D) cloud at one time. grad_V: (N, D) -> (N, D). grad_W: (..., D) -> (..., D),
    applied to the differences x_i - x_j; the j = i term is kept and vanishes for an even,
    smooth W. w: (N,) weights summing to 1, uniform 1/N if None. O(N^2 D) when grad_W is given.
    """
    g = torch.zeros_like(X) if grad_V is None else grad_V(X)
    if grad_W is not None:
        diff = X[:, None, :] - X[None, :, :]  # (N, N, D): x_i - x_j
        gW = grad_W(diff)
        g = g + (gW.mean(dim=1) if w is None else torch.einsum("ijd,j->id", gW, w))
    return g


def _noise(X, dt, beta, gen):
    if beta == 0:
        return torch.zeros_like(X)
    eps = torch.randn(X.shape, generator=gen, dtype=X.dtype, device=X.device)
    return math.sqrt(2.0 * beta * dt) * eps


@torch.no_grad()
def em_step(X, dt, beta, grad_V, grad_W=None, w=None, gen=None):
    """One explicit Euler–Maruyama step (paper Eq. 15 with sigma = sqrt(2 beta)):

        X^{n+1}_i = X^n_i - dt [grad V(X^n_i) + sum_j w_j grad W(X^n_i - X^n_j)] + sqrt(2 beta dt) eps^n_i

    All forces come from the cloud at time n before any particle moves. X: (N, D).
    """
    return X - dt * drift(X, grad_V, grad_W, w) + _noise(X, dt, beta, gen)


@torch.no_grad()
def heun_step(X, dt, beta, grad_V, grad_W=None, w=None, gen=None):
    """One stochastic Heun step for additive noise: predict with Euler–Maruyama, evaluate the
    force on the whole predicted cloud, then step from X^n with the average force and the same
    noise. Second order in dt when beta = 0.
    """
    kick = _noise(X, dt, beta, gen)
    g0 = drift(X, grad_V, grad_W, w)
    g1 = drift(X - dt * g0 + kick, grad_V, grad_W, w)
    return X - 0.5 * dt * (g0 + g1) + kick


_SCHEMES = {"euler": em_step, "heun": heun_step}


@torch.no_grad()
def _run(X, t_a, t_b, dt, beta, grad_V, grad_W, w, gen, step):
    # ceil(gap / dt) equal sub-steps, so t_b is hit exactly and no step exceeds dt;
    # the 1e-9 slack stops float noise (5 / 0.01 = 500.0000001) adding a step
    n_sub = max(1, math.ceil((t_b - t_a) / dt - 1e-9))
    h = (t_b - t_a) / n_sub
    for _ in range(n_sub):
        X = step(X, h, beta, grad_V, grad_W, w, gen)
    return X


@torch.no_grad()
def simulate(
    X0: Union[torch.Tensor, Callable[[Optional[torch.Generator]], torch.Tensor]],
    t_obs,
    dt: float,
    beta: float,
    grad_V: Optional[GradFn],
    grad_W: Optional[GradFn] = None,
    w: Optional[torch.Tensor] = None,
    paired: bool = True,
    gen: Optional[torch.Generator] = None,
    scheme: str = "euler",
) -> List[Tuple[float, torch.Tensor]]:
    """Snapshots of the SDE at the times t_obs, as a list of (t, X_t) with X_t of shape (N, D).

    X0: the cloud at t_obs[0], or a sampler gen -> (N, D) tensor.
    paired=True: one run; row i is the same particle in every snapshot.
    paired=False: a fresh run from t_obs[0] for every snapshot, so rows of different snapshots
    do not correspond. With a sampler each run draws a fresh X0; with a fixed X0 tensor each
    run uses fresh noise and its rows are randomly permuted.
    scheme: "euler" (Euler–Maruyama) or "heun".
    """
    step = _SCHEMES[scheme]
    t_obs = [float(t) for t in t_obs]
    if any(b <= a for a, b in zip(t_obs, t_obs[1:])):
        raise ValueError("t_obs must be strictly increasing")
    draw = X0 if callable(X0) else (lambda g: X0)

    if paired:
        X = draw(gen)
        out = [(t_obs[0], X)]
        for t_a, t_b in zip(t_obs, t_obs[1:]):
            X = _run(X, t_a, t_b, dt, beta, grad_V, grad_W, w, gen, step)
            out.append((t_b, X))
        return out

    out = []
    for t in t_obs:
        X = draw(gen)
        if not callable(X0):
            X = X[torch.randperm(len(X), generator=gen, device=X.device)]
        if t > t_obs[0]:
            X = _run(X, t_obs[0], t, dt, beta, grad_V, grad_W, w, gen, step)
        out.append((t, X))
    return out
