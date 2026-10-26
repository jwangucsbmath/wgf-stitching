"""Checks for the forward step, against closed-form flows where possible.

Reference flow: V = aV/2 |x|^2, W = aW/2 |r|^2. The interaction force is
aW (x_i - mean), so the mean obeys dm/dt = -aV m and deviations from the mean
decay at rate aV + aW. With beta = 0 every particle has the exact path
    x_i(t) = m0 e^{-aV t} + (x_i(0) - m0) e^{-(aV + aW) t},
and with noise the per-coordinate variance is
    v(t) = v_inf + (v0 - v_inf) e^{-2 (aV + aW) t},   v_inf = beta / (aV + aW).
"""

import math

import pytest
import torch

from wgf import potentials as P
from wgf.sim import drift, em_step, heun_step, simulate

torch.set_default_dtype(torch.float64)

AV, AW = 1.0, 0.5


def gV(x):
    return P.grad_quadratic(x, AV)


def gW(r):
    return P.grad_quadratic_interaction(r, AW)


def exact_deterministic(X0, t):
    m0 = X0.mean(0, keepdim=True)
    return m0 * math.exp(-AV * t) + (X0 - m0) * math.exp(-(AV + AW) * t)


def sim_stack(*args, **kwargs):
    """simulate() as one (len(t_obs), N, D) tensor."""
    return torch.stack([X for _, X in simulate(*args, **kwargs)])


def gen(seed=0):
    return torch.Generator().manual_seed(seed)


# ── the step itself ─────────────────────────────────────────────────────────────


def test_vectorised_equals_frozen_loop():
    """Vectorised drift = a loop that reads every neighbour at time n."""
    X = torch.randn(30, 2, generator=gen())
    gW_ = lambda r: P.grad_gaussian_interaction(r, a=1.0, s=0.7)
    frozen = X.clone()
    loop = torch.stack(
        [
            gV(frozen[i]) + torch.stack([gW_(frozen[i] - frozen[j]) for j in range(len(X))]).mean(0)
            for i in range(len(X))
        ]
    )
    assert torch.allclose(drift(X, gV, gW_), loop, atol=1e-12)


def test_in_place_loop_is_order_dependent():
    """Updating particles in place mixes times n and n+1; the vectorised step does not."""
    X = torch.randn(10, 2, generator=gen())
    dt = 0.1
    X_inplace = X.clone()
    for i in range(len(X)):
        g_i = gV(X_inplace[i]) + gW(X_inplace[i] - X_inplace).mean(0)
        X_inplace[i] = X_inplace[i] - dt * g_i
    X_vec = em_step(X, dt, 0.0, gV, gW)
    assert not torch.allclose(X_inplace, X_vec)


def test_permutation_equivariance():
    X = torch.randn(40, 3, generator=gen())
    perm = torch.randperm(40, generator=gen(1))
    gW_ = lambda r: P.grad_gaussian_interaction(r, a=-0.5, s=1.0)
    for step in (em_step, heun_step):
        a = step(X, 0.05, 0.0, gV, gW_)[perm]
        b = step(X[perm], 0.05, 0.0, gV, gW_)
        assert torch.allclose(a, b, atol=1e-12)


def test_self_term_vanishes():
    r = torch.zeros(5, 2)
    assert torch.all(P.grad_gaussian_interaction(r) == 0)
    assert torch.all(P.grad_quadratic_interaction(r) == 0)


def test_uniform_weights_match_mean():
    X = torch.randn(25, 2, generator=gen())
    w = torch.full((25,), 1 / 25)
    assert torch.allclose(drift(X, gV, gW, w), drift(X, gV, gW), atol=1e-12)


def test_seeded_noise_reproducible():
    X = torch.randn(50, 2, generator=gen())
    a = em_step(X, 0.01, 0.3, gV, gW, gen=gen(7))
    b = em_step(X, 0.01, 0.3, gV, gW, gen=gen(7))
    assert torch.equal(a, b)


# ── closed-form gradients vs autodiff ──────────────────────────────────────────


@pytest.mark.parametrize(
    "energy, grad",
    [
        (P.quadratic, P.grad_quadratic),
        (P.wavy_valley, P.grad_wavy_valley),
        (P.gaussian_interaction, P.grad_gaussian_interaction),
        (P.quadratic_interaction, P.grad_quadratic_interaction),
    ],
)
def test_closed_form_grads(energy, grad):
    x = torch.randn(100, 2, generator=gen()) * 2
    assert torch.allclose(grad(x), P.autograd_grad(energy)(x), atol=1e-10)


def test_double_well_grad_away_from_kink():
    x = torch.linspace(-10, 10, 200).reshape(-1, 1)
    x = x[x.abs().squeeze() > 1e-6]
    assert torch.allclose(P.grad_double_well_1d(x), P.autograd_grad(P.double_well_1d)(x), atol=1e-10)


# ── convergence against the exact flow ─────────────────────────────────────────


def test_deterministic_convergence_orders():
    """beta = 0: Euler error ~ dt, Heun error ~ dt^2."""
    X0 = torch.randn(50, 2, generator=gen()) + 1.0
    T = 1.0
    exact = exact_deterministic(X0, T)
    for scheme, order in (("euler", 1), ("heun", 2)):
        errs = []
        for dt in (0.02, 0.01, 0.005):
            XT = sim_stack(X0, [0.0, T], dt, 0.0, gV, gW, scheme=scheme)[-1]
            errs.append((XT - exact).abs().max().item())
        rates = [math.log2(errs[k] / errs[k + 1]) for k in range(2)]
        assert all(abs(r - order) < 0.1 for r in rates), (scheme, errs, rates)


@pytest.mark.parametrize("scheme", ["euler", "heun"])
def test_stochastic_moments(scheme):
    """beta > 0: mean and variance follow the closed-form OU-with-interaction flow."""
    beta, N, D, T = 0.4, 4000, 2, 1.0
    m0, v0 = 2.0, 0.25
    X0 = m0 + v0**0.5 * torch.randn(N, D, generator=gen())
    XT = sim_stack(X0, [0.0, T], 0.01, beta, gV, gW, scheme=scheme, gen=gen(1))[-1]

    v_inf = beta / (AV + AW)
    v_exact = v_inf + (v0 - v_inf) * math.exp(-2 * (AV + AW) * T)
    m_exact = m0 * math.exp(-AV * T)
    # 8000 pooled coordinates: SE(mean) ~ 0.006, relative SE(var) ~ 0.016
    assert abs(XT.mean().item() - m_exact) < 0.03
    assert abs(XT.var().item() / v_exact - 1) < 0.06


def test_simulate_hits_observation_times():
    X0 = torch.randn(20, 2, generator=gen())
    t_obs = [0.0, 0.3, 1.0, 1.005]  # last gap is shorter than dt
    traj = sim_stack(X0, t_obs, 0.01, 0.0, gV, gW, scheme="heun")
    assert traj.shape == (4, 20, 2)
    assert torch.equal(traj[0], X0)
    # Heun error here is ~1e-5; landing one sub-step off would cost ~1e-2
    for k, t in enumerate(t_obs):
        assert torch.allclose(traj[k], exact_deterministic(X0, t), atol=1e-4)


def test_simulate_rejects_unsorted_times():
    with pytest.raises(ValueError):
        simulate(torch.zeros(3, 1), [0.0, 1.0, 1.0], 0.1, 0.0, gV)


# ── exact-answer checks for stage 1 ───────────────────────────────────────────────


def test_noise_scale():
    """V = W = 0: pure Brownian motion, per-coordinate variance 2 beta t.

    beta = 0.3 is where sigma = sqrt(2 beta) and the reference code's sqrt(2) c_H disagree
    (0.6 vs 0.18 at t = 1). grad_W is None, not a zero callable: a zero W would still
    build the (N, N, D) difference tensor, 6 GB at N = 20000.
    """
    beta, dt, n_steps, N, D = 0.3, 1e-3, 1000, 20000, 2
    gen_ = gen(0)
    X = torch.zeros(N, D)
    zero = lambda x: torch.zeros_like(x)
    for _ in range(n_steps):
        X = em_step(X, dt, beta, grad_V=zero, gen=gen_)
    t = n_steps * dt
    # sample variance is accurate to ~1% at N = 20000
    assert torch.allclose(X.var(dim=0), torch.full((D,), 2 * beta * t), rtol=0.05)


def test_gaussian_flow_and_weak_order():
    """V = x^2/2, W = 0, Gaussian start: m(t) = m0 e^{-t}, s^2(t) = beta + (s0^2 - beta) e^{-2t}.

    Euler–Maruyama has weak order 1, so halving dt should roughly halve the bias of
    the mean and the variance. Starting from a point (s0 = 0) at m0 = 5 makes the bias
    (~0.1 in the mean, ~7% in the variance at dt = 0.1) far larger than the Monte Carlo
    error at N = 400k.
    """
    beta, m0, T, N = 0.5, 5.0, 1.0, 400_000
    m_ex = m0 * math.exp(-T)
    v_ex = beta + (0.0 - beta) * math.exp(-2 * T)
    bias = {}
    for k, dt in enumerate((0.1, 0.05)):
        XT = sim_stack(torch.full((N, 1), m0), [0.0, T], dt, beta, P.grad_quadratic, gen=gen(k))[-1]
        bias[dt] = (XT.mean().item() - m_ex, XT.var().item() - v_ex)
    for i in range(2):  # mean, variance
        ratio = bias[0.1][i] / bias[0.05][i]
        assert 1.6 < ratio < 2.5, (i, bias)

    # and at a small step the moments are right within Monte Carlo error
    s0 = 0.5
    X0 = m0 + s0 * torch.randn(50_000, 1, generator=gen(3))
    XT = sim_stack(X0, [0.0, T], 0.002, beta, P.grad_quadratic, gen=gen(4))[-1]
    v_ex = beta + (s0**2 - beta) * math.exp(-2 * T)
    assert abs(XT.mean().item() - m_ex) < 0.02  # SE ~ 0.003
    assert abs(XT.var().item() / v_ex - 1) < 0.04  # SE ~ 0.6 %, EM bias ~ 0.3 %


def test_double_well_equilibrium():
    """V = (|x| - 7)^2 / 2, beta = 1: equilibrium is two unit-variance bumps at +-7.

    Test |x| (mean ~ 7, variance ~ 1), not the split between wells: the barrier
    V(0) = 24.5 means crossings take ~e^{24.5} time units, so the split is frozen in
    by the initial condition.
    """
    X0 = torch.randn(20_000, 1, generator=gen(0))
    XT = sim_stack(X0, [0.0, 10.0], 0.01, 1.0, P.grad_double_well_1d, gen=gen(1))[-1]
    a = XT.abs()
    assert abs(a.mean().item() - 7.0) < 0.05
    assert abs(a.var().item() - 1.0) < 0.05


@pytest.mark.parametrize("scheme", ["euler", "heun"])
def test_interaction_conserves_centre_of_mass(scheme):
    """V = 0, beta = 0, even W: pair forces cancel (grad W is odd), the mean is fixed.

    This also checks that the self-term j = i adds nothing.
    """
    X0 = torch.randn(300, 2, generator=gen(0)) ** 3 + torch.tensor([2.0, -1.0])  # skewed cloud
    for gW_ in (lambda r: P.grad_gaussian_interaction(r, a=1.0, s=0.8),
                lambda r: P.grad_gaussian_interaction(r, a=-2.0, s=0.5),
                gW):
        traj = sim_stack(X0, [0.0, 0.5, 1.0], 0.01, 0.0, None, grad_W=gW_, scheme=scheme)
        drift_of_mean = (traj.mean(1) - X0.mean(0)).abs().max().item()
        assert drift_of_mean < 1e-12
        assert (traj[-1] - X0).abs().max() > 1e-2  # the particles did move


def test_deterministic_runs_identical():
    X0 = torch.randn(100, 2, generator=gen(0))
    gW_ = lambda r: P.grad_gaussian_interaction(r, a=0.5, s=1.0)
    a = sim_stack(X0, [0.0, 0.5, 1.0], 0.01, 0.0, gV, gW_)
    b = sim_stack(X0, [0.0, 0.5, 1.0], 0.01, 0.0, gV, gW_)
    assert torch.equal(a, b)


@pytest.mark.parametrize("paired", [True, False])
def test_seeded_snapshots_reproducible(paired):
    sample = lambda g: torch.randn(200, 2, generator=g)
    a = sim_stack(sample, [0.0, 0.5, 1.0], 0.01, 0.3, gV, paired=paired, gen=gen(5))
    b = sim_stack(sample, [0.0, 0.5, 1.0], 0.01, 0.3, gV, paired=paired, gen=gen(5))
    assert torch.equal(a, b)


def test_paired_vs_unpaired():
    """Paired keeps particle identity across snapshots; unpaired rows do not correspond."""
    N = 5000
    sample = lambda g: torch.randn(N, 1, generator=g)
    t_obs = [0.0, 0.1]
    corr = lambda S: torch.corrcoef(torch.stack([S[0, :, 0], S[1, :, 0]]))[0, 1].item()

    paired = sim_stack(sample, t_obs, 0.01, 0.5, gV, paired=True, gen=gen(0))
    unpaired = sim_stack(sample, t_obs, 0.01, 0.5, gV, paired=False, gen=gen(1))
    assert corr(paired) > 0.9
    assert abs(corr(unpaired)) < 0.05
    # same marginals either way
    assert abs(paired[1].var().item() - unpaired[1].var().item()) < 0.06


def test_stepper_accepts_learned_gradient():
    """Stage 3 reuse: an autograd gradient works inside the no_grad stepper."""
    X0 = torch.randn(50, 2, generator=gen(0))
    a = sim_stack(X0, [0.0, 1.0], 0.01, 0.0, P.autograd_grad(P.quadratic))
    b = sim_stack(X0, [0.0, 1.0], 0.01, 0.0, P.grad_quadratic)
    assert torch.allclose(a, b, atol=1e-12)
    assert not a.requires_grad


def test_snapshot_list_shapes_and_identity_with_fixed_x0():
    """simulate returns [(t, X_t)]; unpaired runs from a fixed X0 do not keep row identity."""
    X0 = torch.randn(3000, 2, generator=gen(0))
    t_obs = [0.0, 0.05, 0.1]
    for paired in (True, False):
        snaps = simulate(X0, t_obs, 0.01, 0.2, gV, paired=paired, gen=gen(1))
        assert [t for t, _ in snaps] == t_obs
        assert all(X.shape == (3000, 2) for _, X in snaps)
    paired = simulate(X0, t_obs, 0.01, 0.2, gV, paired=True, gen=gen(1))
    unpaired = simulate(X0, t_obs, 0.01, 0.2, gV, paired=False, gen=gen(1))
    assert torch.equal(paired[0][1], X0)
    row_dist = lambda S: (S[1][1] - S[2][1]).norm(dim=1).mean().item()
    assert row_dist(paired) < 0.2 < row_dist(unpaired)
