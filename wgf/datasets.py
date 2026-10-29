"""Snapshot datasets of the plan, built with the step-1 simulator (no learning)."""

import torch

from . import potentials as P
from .sim import simulate


def quadratic_snapshots(gen, M=1000, beta=0.5, m0=2.0, s0=0.5, times=(0.0, 0.5, 1.0, 1.5, 2.0), dt=1e-3):
    """Plan step 5 data: V = x^2/2, beta = 0.5, X0 ~ N(2, 0.5^2), unpaired snapshots of M cells.

    Returns a dict {t: (M, 1)}. Exact moments: m(t) = m0 e^{-t}, s(t)^2 = beta + (s0^2 - beta) e^{-2t}.
    """
    sampler = lambda g: m0 + s0 * torch.randn(M, 1, generator=g)
    return dict(simulate(sampler, list(times), dt, beta, P.grad_quadratic, paired=False, gen=gen))


def double_well_snapshots(gen, M=1000, beta=1.0, n_snap=20, T=4.0, dt=1e-2):
    """Plan step 6b data: V = (|x| - 7)^2 / 2, beta = 1, X0 ~ N(0, 1), paired run of M cells,
    n_snap snapshots on [0, T]. Returns a dict {t: (M, 1)}; the regime (dense / terminal / trickle)
    is chosen by the caller."""
    times = torch.linspace(0, T, n_snap).tolist()
    return dict(simulate(lambda g: torch.randn(M, 1, generator=g), times, dt, beta,
                         P.grad_double_well_1d, gen=gen))


# ── held-out protocol (shared by every regime, so it cannot drift between steps) ────────────────

SPLITS = {
    # 20 snapshots, indices 0..19; both endpoints always train (the model only interpolates)
    "interleaved": [0, 2, 4, 6, 8, 11, 13, 15, 17, 19],  # 10 train / 10 test
    "midpoint": list(range(0, 20, 2)) + [19],  # 11 / 9: every test time midway between two train times
    "coarse": [0, 4, 8, 12, 16, 19],  # long gaps: where physics should beat interpolation
}


def split_snapshots(snaps, gen, split="interleaved", train_frac=0.5):
    """Split paired snapshots {t: (M, D)} by time AND by trajectory.

    Times: `split` names a preset in SPLITS or gives the train indices; the rest are test times.
    Cells: one random permutation of the M trajectories; the first train_frac go to train, the rest
    to test, at every time. Test cells therefore never share a trajectory (and its random well
    assignment) with any training cell, as with real data where each cell is measured once.
    Returns (train {t: cells}, test {t: cells}).
    """
    times = sorted(snaps)
    idx = SPLITS[split] if isinstance(split, str) else list(split)
    if idx[0] != 0 or idx[-1] != len(times) - 1:
        raise ValueError("the first and last snapshot must be training times")
    M = len(snaps[times[0]])
    perm = torch.randperm(M, generator=gen)
    tr, te = perm[: int(train_frac * M)], perm[int(train_frac * M):]
    train = {times[i]: snaps[times[i]][tr] for i in idx}
    test = {t: snaps[t][te] for i, t in enumerate(times) if i not in idx}
    return train, test


def double_well_regimes(gen, split="interleaved", M=2000, trickle_cells=10, **sim):
    """The three step-6b regimes on one held-out split. All are scored on the same test times/cells.
      dense    - every training time, all training cells
      terminal - first and last training time only
      trickle  - every training time, `trickle_cells` cells (a fresh subset per time)
    sim: beta, n_snap, T, dt for double_well_snapshots. Returns ({regime: train dict}, test dict)."""
    train, test = split_snapshots(double_well_snapshots(gen, M=M, **sim), gen, split)
    times = sorted(train)
    n = len(train[times[0]])
    regimes = {
        "dense": train,
        "terminal": {times[0]: train[times[0]], times[-1]: train[times[-1]]},
        "trickle": {t: Y[torch.randperm(n, generator=gen)[:trickle_cells]] for t, Y in train.items()},
    }
    return regimes, test
