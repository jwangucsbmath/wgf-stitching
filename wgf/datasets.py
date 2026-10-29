"""Snapshot datasets of the plan, built with the step-1 simulator (no learning)."""

import math

import torch

from . import potentials as P
from .sim import em_step, simulate


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


# ── saved double-well dataset (wgf.datastore): one simulation, every split drawn from it ────────

SPLIT_NOTES = {
    "interleaved": "10 train / 10 test times, alternating (test times 9 and 10 are adjacent)",
    "midpoint": "11 train / 9 test times; every test time lies midway between two training times",
    "coarse": "6 train / 14 test times; long gaps between training times",
}


def double_well_dataset(seed=0, M=2000, n_snap=20, T=4.0, dt=1e-2, beta=1.0, c=7.0, train_frac=0.5,
                        trickle_cells=10, n_paths=12, splits=("interleaved", "midpoint", "coarse")):
    """Simulate the 1-D double-well once and draw every split. Returns (samples, settings, splits)
    in the wgf.datastore format.

    One torch.Generator(seed) drives, in this order: X0 ~ N(0, 1); the Euler–Maruyama run (paired,
    ceil(gap / dt) equal sub-steps between observation times, as wgf.sim.simulate); one permutation of
    the M trajectories (first train_frac -> train, rest -> test, shared by all splits); then, restarted
    from that same state for each split, the trickle subsets. This is the stream of
    double_well_regimes(Generator(seed), split), so the arrays are the ones the earlier results used.
    Trajectories 0..n_paths-1 are also kept at every integration step (figures only).
    """
    gen = torch.Generator().manual_seed(seed)
    times = torch.linspace(0, T, n_snap, dtype=torch.float64)
    grad_V = lambda x: P.grad_double_well_1d(x, c)
    X = torch.randn(M, 1, generator=gen, dtype=torch.float64)
    snaps, fine_t, fine_X, steps = [X], [float(times[0])], [X[:n_paths]], []
    t_list = times.tolist()
    for t_a, t_b in zip(t_list, t_list[1:]):
        n_sub = max(1, math.ceil((t_b - t_a) / dt - 1e-9))
        h = (t_b - t_a) / n_sub
        steps.append(h)
        for i in range(n_sub):
            X = em_step(X, h, beta, grad_V, gen=gen)
            fine_t.append(t_b if i == n_sub - 1 else t_a + (i + 1) * h)
            fine_X.append(X[:n_paths])
        snaps.append(X)

    perm = torch.randperm(M, generator=gen)
    n_tr = int(train_frac * M)
    tr, te = perm[:n_tr], perm[n_tr:]
    state = gen.get_state()
    out = {}
    for name in splits:
        idx = SPLITS[name]
        gen.set_state(state)
        trickle = [tr[torch.randperm(n_tr, generator=gen)[:trickle_cells]].tolist() for _ in idx]
        out[name] = {
            "name": name, "description": SPLIT_NOTES.get(name, ""),
            "train_time_idx": list(idx), "val_time_idx": [],
            "test_time_idx": [j for j in range(n_snap) if j not in idx],
            "train_traj_ids": tr.tolist(), "val_traj_ids": [], "test_traj_ids": te.tolist(),
            "validation": "none: nothing is tuned or selected on held-out data (no early stopping)",
            "regimes": {
                "dense": {"time_idx": list(idx), "traj_ids": tr.tolist()},
                "terminal": {"time_idx": [idx[0], idx[-1]], "traj_ids": tr.tolist()},
                "trickle": {"time_idx": list(idx), "traj_ids_per_time": trickle},
            },
        }

    samples = {"times": times, "traj_ids": torch.arange(M), "X": torch.stack(snaps),
               "paths": {"traj_ids": torch.arange(n_paths), "times": torch.tensor(fine_t, dtype=torch.float64),
                         "X": torch.stack(fine_X)}}
    settings = {
        "system": "double_well_1d", "dimension": 1, "paired": True,
        "potential": {"name": "wgf.potentials.double_well_1d", "formula": "V(x) = (|x| - c)^2 / 2", "c": c,
                      "gradient": "x - c sign(x) (sign(0) = 0)"},
        "initial_distribution": "N(0, 1)",
        "diffusion": {"beta": beta, "sigma": math.sqrt(2 * beta),
                      "sde": "dX = -grad V(X) dt + sqrt(2 beta) dB"},
        "integration": {"method": "Euler-Maruyama (wgf.sim.em_step)", "dt_max": dt,
                        "rule": "ceil(gap / dt_max) equal sub-steps between consecutive observation times",
                        "step_sizes": sorted(set(round(h, 15) for h in steps))},
        "n_trajectories": M,
        "observation_times": {"n": n_snap, "T": T, "rule": "linspace(0, T, n)"},
        "train_frac": train_frac, "trickle_cells": trickle_cells,
        "fine_paths": f"trajectories 0..{n_paths - 1} at every integration step (samples.pt['paths'])",
    }
    seeds = {"data_generation": seed,
             "splits": "no separate seed: the trajectory permutation and trickle subsets continue the "
                       "data-generation generator after the simulation (as double_well_regimes)"}
    return samples, {"settings": settings, "seeds": seeds}, out
