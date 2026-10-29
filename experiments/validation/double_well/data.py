"""Double-well validation: the saved simulated datasets. Simulate once, then load by ID.

  .venv/bin/python experiments/validation/double_well/data.py list
  .venv/bin/python experiments/validation/double_well/data.py show [dw-001]
  .venv/bin/python experiments/validation/double_well/data.py generate --seed 1 [--M 2000] [--n-snap 20] [--T 4] [--dt 0.01] [--beta 1]
  .venv/bin/python experiments/validation/double_well/data.py verify [dw-001]

`generate` simulates a new dataset and always writes a new ID (dw-002, dw-003, ...); it never touches an
existing dataset. `verify` re-simulates a saved dataset from its recorded seed and settings and compares,
writing nothing (run.py eval --verify-data does the same). run.py and the notebook only load, dw-001 by
default. dw-001 is seed 0 with the defaults below: bit-for-bit the data behind the results recorded on
2026-10-26.
"""

import argparse
import json
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
from wgf import datastore  # noqa: E402
from wgf.datasets import SPLITS, double_well_dataset  # noqa: E402

torch.set_default_dtype(torch.float64)
DATA_ROOT = HERE / "data"
DEFAULT_DATASET = "dw-001"
PREFIX = "dw"


def load(dataset_id=DEFAULT_DATASET):
    """The saved dataset; never simulates. Errors if the ID does not exist."""
    return datastore.load(DATA_ROOT, dataset_id)


def verify_dataset(ds):
    """Re-simulate ds from its recorded seed and settings and compare with the saved arrays and splits.
    Writes nothing. Returns {"regenerated": True, "match": bool, "differences": [...]}."""
    s = ds.meta["settings"]
    samples, _, splits = double_well_dataset(
        ds.meta["seeds"]["data_generation"], s["n_trajectories"], s["observation_times"]["n"],
        s["observation_times"]["T"], s["integration"]["dt_max"], s["diffusion"]["beta"], s["potential"]["c"],
        s["train_frac"], s["trickle_cells"], len(ds.paths["traj_ids"]), tuple(ds.splits))
    saved = {"times": ds.times, "traj_ids": ds.traj_ids, "X": ds.X}
    diffs = []
    for k, v in saved.items():
        new = samples[k]
        if new.shape != v.shape:
            diffs.append(f"{k}: shape {tuple(new.shape)} regenerated vs {tuple(v.shape)} saved")
        elif not torch.equal(new, v):
            diffs.append(f"{k}: values differ (max |difference| {(new.double() - v.double()).abs().max().item():.3g})")
    for k in ("times", "X"):
        new, old = samples["paths"][k], ds.paths[k]
        if new.shape != old.shape or not torch.equal(new, old):
            diffs.append(f"paths/{k}: the fine-resolution trajectories differ")
    for name, sp in ds.splits.items():
        changed = [k for k in sp if splits.get(name, {}).get(k) != sp[k]]
        if changed:
            diffs.append(f"split {name}: {', '.join(changed)} differ")
    return {"regenerated": True, "match": not diffs, "differences": diffs,
            "seed": ds.meta["seeds"]["data_generation"], "samples_sha256": ds.meta["samples_sha256"]}


def describe(ds):
    m = ds.meta
    lines = [f"# dataset {ds.id}  ({ds.path})", f"description = {m.get('description', '')}",
             f"created = {m['created']}   commit = {m['provenance']['commit']}",
             f"samples: X {tuple(ds.X.shape)} (time, trajectory, dim), {len(ds.times)} times "
             f"{[round(t, 3) for t in ds.times.tolist()]}",
             f"fine paths: {tuple(ds.paths['X'].shape)} for trajectory IDs {ds.paths['traj_ids'].tolist()}",
             "", "## settings"]
    lines += [f"{k} = {json.dumps(v)}" for k, v in m["settings"].items()]
    lines += ["", "## seeds"] + [f"{k} = {v}" for k, v in m["seeds"].items()]
    lines += ["", "## splits (time indices into the times above; trajectory counts)"]
    for name, s in ds.splits.items():
        lines.append(f"{ds.split_id(name):18} train t {s['train_time_idx']}  test t {s['test_time_idx']}  "
                     f"val t {s['val_time_idx']}  | trajectories train {len(s['train_traj_ids'])}, "
                     f"val {len(s['val_traj_ids'])}, test {len(s['test_traj_ids'])}")
        for r in s["regimes"]:
            idx, ids = ds.regime_cells(name, r)
            lines.append(f"{'':18}   regime {r:8}: {len(idx)} times x {sorted(set(map(len, ids)))} cells")
    lines += ["", f"samples_sha256 = {m['samples_sha256']}", f"splits_sha256 = {m['splits_sha256']}"]
    return "\n".join(lines)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="command", required=True)
    sub.add_parser("list", help="every saved dataset")
    sh = sub.add_parser("show", help="settings, seeds, splits and hashes of one dataset")
    sh.add_argument("id", nargs="?", default=DEFAULT_DATASET)
    vf = sub.add_parser("verify", help="re-simulate a dataset from its seed and compare (writes nothing)")
    vf.add_argument("id", nargs="?", default=DEFAULT_DATASET)
    g = sub.add_parser("generate", help="simulate a NEW dataset under a new ID (never overwrites)")
    g.add_argument("--seed", type=int, required=True, help="data-generation seed")
    g.add_argument("--M", type=int, default=2000, help="trajectories")
    g.add_argument("--n-snap", type=int, default=20, help="observation times on [0, T]")
    g.add_argument("--T", type=float, default=4.0)
    g.add_argument("--dt", type=float, default=1e-2, help="largest Euler-Maruyama step")
    g.add_argument("--beta", type=float, default=1.0, help="diffusion coefficient (sigma = sqrt(2 beta))")
    g.add_argument("--description", default="")
    a = ap.parse_args()
    try:
        if a.command == "list":
            for i, m in datastore.list_datasets(DATA_ROOT):
                s = m["settings"]
                print(f"{i}  seed {m['seeds']['data_generation']}  M {s['n_trajectories']}  "
                      f"{s['observation_times']['n']} times on [0, {s['observation_times']['T']}]  "
                      f"beta {s['diffusion']['beta']}  splits {list(json.loads((DATA_ROOT / i / 'splits.json').read_text()))}"
                      f"  {m.get('description', '')}")
        elif a.command == "show":
            print(describe(load(a.id)))
        elif a.command == "verify":
            rep = verify_dataset(load(a.id))
            if not rep["match"]:
                sys.exit(f"error: regenerating {a.id} from seed {rep['seed']} does not reproduce the saved data:\n  "
                         + "\n  ".join(rep["differences"]))
            print(f"{a.id}: regenerated from seed {rep['seed']} and identical to the saved samples, paths and splits")
        else:
            if a.n_snap != 20:
                sys.exit("error: the split presets (wgf.datasets.SPLITS) index 20 observation times; keep --n-snap 20")
            samples, meta, splits = double_well_dataset(a.seed, a.M, a.n_snap, a.T, a.dt, a.beta, splits=tuple(SPLITS))
            ds = datastore.create(DATA_ROOT, PREFIX, {"description": a.description, **meta}, samples, splits)
            print(f"created {ds.id} at {ds.path}\n")
            print(describe(ds))
    except (datastore.DatasetNotFound, datastore.DatasetCorrupt) as e:
        sys.exit(f"error: {e}")
