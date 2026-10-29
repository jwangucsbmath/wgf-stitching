# Paper reproduction (arXiv:2607.04738)

This folder is for the experiments that appear in the paper. **None has been reproduced yet.** Everything in the repository so far is either a component check or an additional validation on our own simulated data (see [`../validation/`](../validation)). The shared implementation those experiments will use is in [`wgf/`](../../wgf).

| paper section | experiment | data | status |
|---|---|---|---|
| §5.1, Fig. 1 | wavy valley vs. JKOnet* | simulated, 2-D, 40 training cells, irregular times | not started (simulator and potential are in `wgf/`) |
| §5.2, Tables 1/4 | 15 synthetic potentials, paired and unpaired | simulated, 2-D | not started |
| §5.3, Fig. 3, Tables 2/5 | embryoid-body scRNA-seq, static and time-conditioned V, gappy variant | third-party data | not started |
| §5.4, Fig. 4, Table 3 | CIS: interacting particles (V and W) | simulated McKean–Vlasov | not started (needs step 7, W_θ) |
| §5.5, Fig. 5 | chiral (non-gradient) flow | simulated | not started |

Several of the paper's own experiments also use simulated data, so the split between this folder and `../validation/` is by *purpose*, not by data source: an experiment goes here only when it reproduces something the paper reports. Hyperparameters per paper experiment are in `00_reference_repo_structure.md` §8.
