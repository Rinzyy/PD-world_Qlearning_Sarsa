# PD-World Reinforcement Learning

A batch experiment runner for the PD-World reinforcement learning assignment, using Q-Learning and SARSA.

## 🚀 Setup

Install dependencies:
```bash
python3 -m pip install -r requirements.txt
```

## 🏃 Running Experiments

Experiments generate results in the `artifacts/` directory. For this submission, please use seeds **7** and **19**.

**Run all experiments for seeds 7 and 19:**
```bash
python3 -m pdworld.adapters.batch.cli run-all --seeds 7 19
```

**Run a single experiment (e.g., Exp 1 with seed 7):**
```bash
python3 -m pdworld.adapters.batch.cli run-exp --exp 1 --seed 7
```

**Options for `run-exp`:**
- `--exp {1,2,3}`: Experiment ID
- `--seed`: RNG seed (e.g., 7 or 19)
- `--output`: Output directory (default: `artifacts`)
