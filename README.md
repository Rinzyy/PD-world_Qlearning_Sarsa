# PD-World Tabular TD System

Python implementation of tabular Q-learning and SARSA for the 5x5 PD-World assignment.

## Install

```bash
python3 -m pip install -r requirements.txt
```

## Run all required experiments

```bash
python3 -m pdworld.cli run-all --output /Users/tuyrindy/Desktop/AIClass/artifacts --seeds 7 19
```

## Run one experiment

```bash
python3 -m pdworld.cli run-exp --exp 2 --seed 7 --output /Users/tuyrindy/Desktop/AIClass/artifacts
```

## Regenerate attractive paths (Experiment 2)

```bash
python3 -m pdworld.cli attractive-paths --exp2-dir /Users/tuyrindy/Desktop/AIClass/artifacts/exp2/seed_7
```

## Run tests

```bash
pytest -q
```

## Artifacts

For each run:

- `timeseries.csv` (step, reward, cumulative bank account)
- `episode_lengths.csv` (operators to terminal per completed episode)
- `q_<snapshot>.csv` and `q_<snapshot>.png`
- `metadata.json`
- `cumulative_reward.png`, `episode_lengths.png`
- Experiment 2 also writes `attractive_paths_<snapshot>_x{0|1}.png`
