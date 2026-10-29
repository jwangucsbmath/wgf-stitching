"""[DATA] Saved simulated datasets with IDs: simulate once, then always load.

    <root>/<id>/
        dataset.json   id, description, simulation settings, seeds, how the splits were drawn,
                       sha256 of samples.pt and of splits.json, provenance
        samples.pt     times (T,), traj_ids (M,), X (T, M, D) with X[j, i] = trajectory traj_ids[i] at
                       times[j]; paths: a few trajectories at every integration step (figures only)
        splits.json    {name: split}: time indices and trajectory IDs for train / val / test, and for
                       each observation regime the trajectory IDs observed at each training time

`create` always makes a new ID (<prefix>-001, -002, ...) in a new folder and never writes into an
existing one. `load` checks both hashes. Nothing here simulates; the caller passes the arrays in.
Trajectory-ID lists are ordered: rows are taken in that order, which fixes e.g. which OT pairs the
path initialisation picks, so the order is part of the split.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import torch

from .runs import MissingRunFile, RunMismatch, _now, data_hash, provenance, write_json

# subclasses of the run errors, so the experiment command lines print them as one-line errors


class DatasetNotFound(MissingRunFile):
    """No dataset with that ID (or no split / regime with that name)."""


class DatasetCorrupt(RunMismatch):
    """A saved file no longer matches the hash recorded when the dataset was created."""


def _splits_hash(splits):
    return data_hash(json.dumps(splits, sort_keys=True))


@dataclass
class Dataset:
    id: str
    path: Path
    meta: dict
    times: torch.Tensor  # (T,)
    traj_ids: torch.Tensor  # (M,)
    X: torch.Tensor  # (T, M, D)
    paths: dict
    splits: dict

    def split(self, name):
        if name not in self.splits:
            raise DatasetNotFound(f"dataset {self.id} has no split {name!r}; it has {sorted(self.splits)}")
        return self.splits[name]

    def split_id(self, name):
        self.split(name)
        return f"{self.id}:{name}"

    def cells(self, j, ids):
        """Samples of the trajectories `ids` (in that order) at time index j: (len(ids), D)."""
        row = {int(t): i for i, t in enumerate(self.traj_ids.tolist())}
        return self.X[j, torch.tensor([row[i] for i in ids], dtype=torch.long)]

    def _by_time(self, time_idx, ids_per_time):
        return {float(self.times[j]): self.cells(j, ids) for j, ids in zip(time_idx, ids_per_time)}

    def regime_cells(self, split, regime):
        """(time indices, trajectory IDs per time) that a training regime observes."""
        s = self.split(split)
        if regime not in s["regimes"]:
            raise DatasetNotFound(f"split {self.id}:{split} has no regime {regime!r}; it has {sorted(s['regimes'])}")
        r = s["regimes"][regime]
        per_time = r.get("traj_ids_per_time") or [r["traj_ids"]] * len(r["time_idx"])
        return r["time_idx"], per_time

    def train(self, split, regime):
        """{t: cells} of one training regime: training times and training trajectories only."""
        return self._by_time(*self.regime_cells(split, regime))

    def held_out(self, split, part="test"):
        """{t: cells} of the validation or test part of a split."""
        s = self.split(split)
        idx, ids = s[f"{part}_time_idx"], s[f"{part}_traj_ids"]
        return self._by_time(idx, [ids] * len(idx))


def list_datasets(root):
    """[(id, dataset.json contents)] of every dataset under root, sorted by ID."""
    out = []
    for p in sorted(Path(root).glob("*/dataset.json")):
        out.append((p.parent.name, json.loads(p.read_text())))
    return out


def load(root, dataset_id):
    path = Path(root) / dataset_id
    if not (path / "dataset.json").exists():
        have = [i for i, _ in list_datasets(root)]
        raise DatasetNotFound(f"no dataset {dataset_id!r} under {root} (have: {have or 'none'}); "
                              f"datasets are only made by an explicit `generate`")
    meta = json.loads((path / "dataset.json").read_text())
    samples = torch.load(path / "samples.pt", weights_only=True)
    splits = json.loads((path / "splits.json").read_text())
    if data_hash(samples) != meta["samples_sha256"]:
        raise DatasetCorrupt(f"{path / 'samples.pt'} does not match samples_sha256 in dataset.json")
    if _splits_hash(splits) != meta["splits_sha256"]:
        raise DatasetCorrupt(f"{path / 'splits.json'} does not match splits_sha256 in dataset.json")
    return Dataset(dataset_id, path, meta, samples["times"], samples["traj_ids"], samples["X"],
                   samples.get("paths", {}), splits)


def create(root, prefix, meta, samples, splits):
    """Save a new dataset under the next free ID <prefix>-NNN and return it loaded back."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    taken = [int(p.name.rsplit("-", 1)[1]) for p in root.glob(f"{prefix}-*") if p.name.rsplit("-", 1)[1].isdigit()]
    n = max(taken, default=0) + 1
    while True:
        dataset_id = f"{prefix}-{n:03d}"
        try:
            (root / dataset_id).mkdir()
            break
        except FileExistsError:
            n += 1
    path = root / dataset_id
    torch.save(samples, path / "samples.pt")
    (path / "splits.json").write_text(json.dumps(splits, indent=1) + "\n")
    write_json(path / "dataset.json", {"id": dataset_id, "created": _now(), **meta,
                                       "samples_sha256": data_hash(samples), "splits_sha256": _splits_hash(splits),
                                       "provenance": provenance()})
    return load(root, dataset_id)
