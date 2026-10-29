"""Saved datasets (wgf/datastore.py, experiments/validation/double_well/data): the committed dw-001
is the data of the earlier results, splits keep held-out cells out of training and initialisation,
and new datasets never overwrite old ones."""

import json
import sys
from pathlib import Path

import pytest
import torch

from wgf import datastore
from wgf.datasets import SPLITS, double_well_dataset, double_well_regimes

torch.set_default_dtype(torch.float64)
DW = Path(__file__).resolve().parents[1] / "experiments" / "validation" / "double_well"
sys.path.insert(0, str(DW))
from run import REGIMES, train_model  # noqa: E402


@pytest.fixture(scope="module")
def dw001():
    return datastore.load(DW / "data", "dw-001")


def test_dw001_is_the_data_of_the_earlier_results(dw001):
    """Bit-for-bit what double_well_regimes(Generator(0), split) gave the 2026-10-26 runs."""
    assert dw001.meta["seeds"]["data_generation"] == 0 and tuple(dw001.X.shape) == (20, 2000, 1)
    for name in SPLITS:
        regimes, test = double_well_regimes(torch.Generator().manual_seed(0), split=name)
        for r, tr in regimes.items():
            new = dw001.train(name, r)
            assert list(new) == list(tr) and all(torch.equal(new[t], tr[t]) for t in tr), (name, r)
        held = dw001.held_out(name, "test")
        assert list(held) == list(test) and all(torch.equal(held[t], test[t]) for t in test)


def test_fine_paths_agree_with_the_snapshots(dw001):
    p = dw001.paths
    for j, t in enumerate(dw001.times.tolist()):
        k = int(torch.argmin((p["times"] - t).abs()))
        assert abs(float(p["times"][k]) - t) < 1e-12
        assert torch.equal(p["X"][k], dw001.cells(j, p["traj_ids"].tolist()))


@pytest.mark.parametrize("name", list(SPLITS))
def test_splits_are_disjoint_and_regimes_use_training_cells_only(dw001, name):
    s = dw001.split(name)
    tr, te = set(s["train_traj_ids"]), set(s["test_traj_ids"])
    assert not tr & te and not tr & set(s["val_traj_ids"]) and len(tr | te) == 2000
    assert not set(s["train_time_idx"]) & set(s["test_time_idx"])
    assert sorted(s["train_time_idx"] + s["test_time_idx"] + s["val_time_idx"]) == list(range(20))
    for r in REGIMES:
        idx, ids = dw001.regime_cells(name, r)
        assert set(idx) <= set(s["train_time_idx"]) and all(set(i) <= tr for i in ids)


def test_training_never_reads_held_out_cells(dw001):
    """Replace every sample outside (training times x training trajectories) by NaN: the regimes'
    training data and the fitted models (paths, bandwidth, network) come out identical."""
    name = "interleaved"
    s = dw001.split(name)
    keep = torch.zeros(dw001.X.shape[:2], dtype=torch.bool)
    keep[torch.tensor(s["train_time_idx"])[:, None], torch.tensor(s["train_traj_ids"])[None, :]] = True
    poisoned = datastore.Dataset(**{**dw001.__dict__, "X": dw001.X.masked_fill(~keep[..., None], float("nan"))})
    assert torch.isnan(poisoned.X).sum() == (~keep).sum()
    cfg = {"model": {"N": 10, "K": 6, "score": "full", "V_theta": {"d_in": 1, "hidden": [8, 8]},
                     "c_H": {"init": 1.0, "learned": True}},
           "train": {"epochs": 3, "lr": 5e-3, "lam": 1.0, "residual_scheme": "trapezoid", "nll": {"pooled": False}},
           "seeds": {"init": 0}, "data": {"diffusion": {"beta": 1.0}}}
    for r in REGIMES:
        clean, dirty = dw001.train(name, r), poisoned.train(name, r)
        assert all(torch.equal(clean[t], dirty[t]) for t in clean)
        a, _ = train_model(clean, cfg)
        b, _ = train_model(dirty, cfg)
        for (k, x), (_, y) in zip(a.state_dict().items(), b.state_dict().items()):
            assert torch.equal(x, y), (r, k)


def test_create_never_overwrites_and_load_checks_hashes(tmp_path):
    samples, meta, splits = double_well_dataset(seed=3, M=40, n_paths=2)
    a = datastore.create(tmp_path, "dw", {"description": "a", **meta}, samples, splits)
    b = datastore.create(tmp_path, "dw", {"description": "b", **meta}, samples, splits)
    assert (a.id, b.id) == ("dw-001", "dw-002") and a.meta["description"] == "a"
    assert [i for i, _ in datastore.list_datasets(tmp_path)] == ["dw-001", "dw-002"]
    with pytest.raises(datastore.DatasetNotFound, match="dw-003"):
        datastore.load(tmp_path, "dw-003")
    with pytest.raises(datastore.DatasetNotFound, match="no split"):
        a.split("nope")
    sp = json.loads((b.path / "splits.json").read_text())
    sp["coarse"]["test_traj_ids"] = sp["coarse"]["test_traj_ids"][:-1]
    (b.path / "splits.json").write_text(json.dumps(sp))
    with pytest.raises(datastore.DatasetCorrupt, match="splits.json"):
        datastore.load(tmp_path, "dw-002")
    x = torch.load(a.path / "samples.pt", weights_only=True)
    x["X"][0, 0, 0] += 1
    torch.save(x, a.path / "samples.pt")
    with pytest.raises(datastore.DatasetCorrupt, match="samples.pt"):
        datastore.load(tmp_path, "dw-001")
