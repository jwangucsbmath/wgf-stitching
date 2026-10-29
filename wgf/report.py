"""[EVAL] Shared helpers for the `plot` commands: they read saved files only (wgf/runs.py layout).
Call matplotlib.use("Agg") in the script before importing this module when no display is wanted."""

import matplotlib.pyplot as plt

from .runs import Run


def mean(xs):
    return sum(xs) / len(xs)


def md_table(header, rows):
    """A Markdown table; floats are printed with 3 decimals."""
    cell = lambda v: f"{v:.3f}" if isinstance(v, float) else str(v)
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    return lines + ["| " + " | ".join(cell(v) for v in r) + " |" for r in rows]


def method_notes(methods):
    """One bullet per method: label, kind (stitching KDE / SDE rollout / OT baseline), dynamics, description."""
    dyn = lambda d: "no dynamics" if d == "none" else f"{d} dynamics"
    return [f"- **{v['label']}** ({v['kind']}; {dyn(v['dynamics'])}): {v['description']}" for v in methods.values()]


def write_tables(ev, lines):
    text = "\n".join(lines) + "\n"
    print(text)
    (ev.path / "tables.md").write_text(text)


def history_figure(ev, tags):
    """Loss, NLL, lambda R, c_H and h per epoch for every model in `tags`, from the run's history/."""
    run = Run.open(ev.path.parents[1], finished=False)
    hists = {tag: run.load_history(tag) for tag in tags}
    keys = [("loss", "loss"), ("nll", "NLL"), ("lamR", "$\\lambda R$"), ("c_H", "$c_H$"), ("h", "$h$")]
    fig, ax = plt.subplots(1, len(keys), figsize=(3.4 * len(keys), 3.2))
    for a, (k, label) in zip(ax, keys):
        for tag, h in hists.items():
            a.plot(h[k], lw=1, label=tag)
        a.set(title=label, xlabel="epoch")
        if k == "lamR":
            a.set_yscale("log")
    ax[0].legend(fontsize=7)
    fig.tight_layout()
    out = ev.path / "training_history.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"saved {out}")
