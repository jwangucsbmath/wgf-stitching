"""Held-out protocol: time/trajectory split, regimes, OT-interpolation baseline, fixed-V* model."""

import pytest
import torch

from wgf import potentials as P
from wgf.baselines import ot_interpolate_1d, ot_interpolate_snapshots
from wgf.datasets import SPLITS, double_well_regimes, split_snapshots
from wgf.energy import grad_x
from wgf.model import FixedV

torch.set_default_dtype(torch.float64)


def gen(seed=0):
    return torch.Generator().manual_seed(seed)


def toy_snaps(M=40, K=20):
    """Paired snapshots whose values encode (time, trajectory): x = 1000 t_index + cell id."""
    return {float(k): (1000 * k + torch.arange(M, dtype=torch.float64))[:, None] for k in range(K)}


@pytest.mark.parametrize("split", list(SPLITS))
def test_split_times_and_trajectories_disjoint(split):
    train, test = split_snapshots(toy_snaps(), gen(), split)
    idx = SPLITS[split]
    assert sorted(train) == [float(i) for i in idx]
    assert set(train).isdisjoint(test) and len(train) + len(test) == 20
    assert 0.0 in train and 19.0 in train  # endpoints train: test times are interpolation only
    ids_tr = {int(v) % 1000 for Y in train.values() for v in Y[:, 0]}
    ids_te = {int(v) % 1000 for Y in test.values() for v in Y[:, 0]}
    assert ids_tr.isdisjoint(ids_te) and len(ids_tr) == len(ids_te) == 20
    # the same trajectories at every train time
    assert all({int(v) % 1000 for v in Y[:, 0]} == ids_tr for Y in train.values())


def test_interleaved_is_ten_ten_and_rejects_open_ends():
    assert len(SPLITS["interleaved"]) == 10
    with pytest.raises(ValueError):
        split_snapshots(toy_snaps(), gen(), [0, 2, 4])  # last snapshot not in train


def test_regimes_share_one_test_set():
    regimes, test = double_well_regimes(gen(), M=200)
    assert len(test) == 10 and all(len(Y) == 100 for Y in test.values())
    assert sorted(regimes["dense"]) == sorted(regimes["trickle"])
    assert len(regimes["terminal"]) == 2
    assert all(len(Y) == 10 for Y in regimes["trickle"].values())
    t0 = min(regimes["dense"])
    assert torch.equal(regimes["terminal"][t0], regimes["dense"][t0])


def test_ot_interpolation_gaussians():
    """Interpolating N(0,1) and N(4, 2^2) at a = 1/4 gives N(1, 1.25^2) in quantiles."""
    u = (torch.arange(4000) + 0.5) / 4000
    q = torch.distributions.Normal(0.0, 1.0).icdf(u)[:, None]
    X = ot_interpolate_1d(q, 4 + 2 * q, 0.25, 4000)
    # empirical quantiles interpolate between order statistics, so compare up to that grid error
    assert X.mean().item() == pytest.approx(1.0, abs=1e-6)
    assert X.std().item() == pytest.approx(1.25 * q.std().item(), rel=5e-3)
    assert torch.equal(X[:, 0].argsort(), torch.arange(4000))  # monotone (comonotone coupling)


def test_ot_interpolation_snapshots_hits_nodes_and_brackets():
    tr = {0.0: torch.zeros(5, 1), 2.0: 2 * torch.ones(7, 1)}
    assert torch.allclose(ot_interpolate_snapshots(tr, 0.5, 10), torch.full((10, 1), 0.5))
    assert torch.allclose(ot_interpolate_snapshots(tr, 2.0, 3), torch.full((3, 1), 2.0))
    with pytest.raises(ValueError):
        ot_interpolate_snapshots(tr, 3.0, 3)


def test_fixed_v_has_no_parameters_and_true_gradient():
    V = FixedV(P.double_well_1d)
    assert len(list(V.parameters())) == 0
    x = torch.tensor([[-8.0], [-1.0], [0.5], [7.0], [9.0]])
    assert torch.allclose(grad_x(V, x), P.grad_double_well_1d(x))
