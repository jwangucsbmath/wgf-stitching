"""Step 4 — the residual on exact and perturbed Gaussian flows (numbers from the plan)."""

import math

import pytest
import torch

from wgf.energy import force
from wgf.potentials import gaussian_flow_path, grad_quadratic
from wgf.residual import residual

torch.set_default_dtype(torch.float64)

BETA, M0, S0, T, N = 0.5, 2.0, 0.5, 2.0, 2000


def x0(seed=0):
    return M0 + S0 * torch.randn(N, 1, generator=torch.Generator().manual_seed(seed))


def R_exact(K, scheme, rate=1.0, seed=0):
    t = torch.linspace(0, T, K)
    X, g = gaussian_flow_path(x0(seed), t, BETA, M0, S0, rate)
    return residual(X, t, torch.full((N,), 1 / N), g=g, scheme=scheme)


def test_exact_path_converges_at_the_right_order():
    """Euler: halving dt divides R by ~4 (6.1e-3 -> 1.5e-3); trapezoid by ~16 (4.1e-6 -> 2.6e-7)."""
    e21, e41 = R_exact(21, "euler"), R_exact(41, "euler")
    t21, t41 = R_exact(21, "trapezoid"), R_exact(41, "trapezoid")
    assert e21.item() == pytest.approx(6.1e-3, rel=0.1) and e41.item() == pytest.approx(1.5e-3, rel=0.1)
    assert t21.item() == pytest.approx(4.1e-6, rel=0.15) and t41.item() == pytest.approx(2.6e-7, rel=0.15)
    assert 3.5 < (e21 / e41).item() < 4.5
    assert 14 < (t21 / t41).item() < 18


def test_perturbed_path_closed_form():
    """Mean decaying at rate r = 0.5: R -> (1 - r)^2 m0^2 (1 - e^{-2rT}) / (2r) = 0.8647."""
    r = 0.5
    limit = (1 - r) ** 2 * M0**2 * (1 - math.exp(-2 * r * T)) / (2 * r)
    assert limit == pytest.approx(0.8647, abs=1e-4)
    assert R_exact(41, "trapezoid", rate=r).item() == pytest.approx(0.8648, abs=2e-4)
    assert R_exact(161, "trapezoid", rate=r).item() == pytest.approx(limit, abs=1e-4)


def test_force_fn_interface_matches_precomputed_g():
    t = torch.linspace(0, T, 11)
    X, g = gaussian_flow_path(x0(), t, BETA, M0, S0)
    w = torch.full((N,), 1 / N)
    via_fn = residual(X, t, w, force_fn=lambda Xj, j: g[j])
    assert torch.allclose(via_fn, residual(X, t, w, g=g))
    with pytest.raises(ValueError):
        residual(X, t, w, g=g, scheme="midpoint")


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_kde_force_separates_exact_from_perturbed(seed):
    """c_H = beta, h = 0.3: exact path ~0.04-0.05, perturbed ~0.83-0.94; exact < perturbed / 10."""
    t = torch.linspace(0, T, 41)
    h, w = torch.tensor([0.3]), torch.full((N,), 1 / N)
    out = {}
    for rate in (1.0, 0.5):
        X, _ = gaussian_flow_path(x0(seed), t, BETA, M0, S0, rate)
        g = force(X, h, w, grad_quadratic, c_H=BETA)
        out[rate] = residual(X, t, w, g=g).item()
    assert 0.02 < out[1.0] < 0.08 and 0.7 < out[0.5] < 1.0
    assert out[1.0] < out[0.5] / 10


def test_energy_dissipation_identity():
    """EDE: F(T) - F(0) = -int_0^T ||g||^2_rho dt = -2.0116, with
    F(t) = (m^2 + s^2)/2 - (beta/2) log(2 pi e s^2) and ||g||^2_rho = m^2 + (s - beta/s)^2."""
    m = lambda t: M0 * math.exp(-t)
    s2 = lambda t: BETA + (S0**2 - BETA) * math.exp(-2 * t)
    F = lambda t: 0.5 * (m(t) ** 2 + s2(t)) - 0.5 * BETA * math.log(2 * math.pi * math.e * s2(t))
    tt = torch.linspace(0, T, 20001, dtype=torch.float64)
    dissipation = torch.tensor([m(t) ** 2 + (math.sqrt(s2(t)) - BETA / math.sqrt(s2(t))) ** 2 for t in tt.tolist()])
    integral = torch.trapezoid(dissipation, tt).item()
    assert F(T) - F(0) == pytest.approx(-integral, abs=1e-6)
    assert F(T) - F(0) == pytest.approx(-2.0116, abs=1e-4)
    # the same integral from the particles of the exact path
    t = torch.linspace(0, T, 401)
    X, g = gaussian_flow_path(x0(), t, BETA, M0, S0)
    mc = torch.trapezoid((g**2).sum(-1).mean(-1), t).item()
    assert mc == pytest.approx(integral, rel=0.03)
