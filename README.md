# Lunar Overconfidence

DDPG and TD3 on continuous LunarLander, studying how layer normalization changes measured critic bias and how bias relates to policy performance.

The notebook compares no normalization, critic-only LayerNorm, and actor plus critic LayerNorm across five training seeds: 30 runs of 300,000 transitions each. Evaluation uses 100 held-out episodes per checkpoint. Bias uses completed Monte Carlo trajectories; reward includes all episodes. Bias-performance associations do not establish causation.

## Start here

Run `uv sync`, open `experiments.ipynb`, and select the project `.venv` Python kernel. The notebook contains the experiment name and settings, followed by separate cells for creation, training, evaluation, analysis, and figures. Implementation stays in `lunar_overconfidence/`.

New experiments go to `outputs/YYYYMMDD-HHMMSS-name/`, with `experiment.json` and algorithm, normalization, and seed subdirectories. Tables are saved in `tables/<comparison>/` and PNG/PDF figures in `<comparison>/figures/` for `critic-vs-none`, `both-vs-none`, and `both-vs-critic`.

## Code

- `ddpg.py`, `td3.py`: explicit, separate training loops and updates.
- `networks.py`, `config.py`, `checkpoints.py`: models, settings, and snapshots.
- `evaluation.py`: matched Monte Carlo returns and checkpoint evaluation.
- `analysis.py`: bias/performance summaries and saved-table loading.
- `experiments.py`: named experiments and notebook stage functions.
- `plotting/`: main figures, diagnostics, and shared style.

The main figures show reward and Q1 bias through training, bias versus performance, and normalization differences paired by training seed. PNG and PDF versions are saved. Diagnostics show TD3's critics and Monte Carlo completion rates.

The notebook includes the measurement assumptions and optional TensorBoard/video cells. Checkpoints support evaluation, not training resumption.

## Tests

```sh
uv run python -m unittest discover -s tests -v
```

Tests cover training, checkpoint compatibility, Monte Carlo alignment, analysis, figures, and notebook stages. The suite launches short training runs in temporary directories.
