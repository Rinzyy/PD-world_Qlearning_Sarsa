# PD-World Reinforcement Learning

A batch experiment runner for the PD-World reinforcement learning assignment. This software simulates two learning algorithms (Q-Learning and SARSA) and three distinct action selection policies to see how agents navigate and learn the underlying world environment.

## 🚀 Getting Started

Ensure you have your dependencies installed:

```bash
python3 -m pip install -r requirements.txt
```

### Running Experiments

You can execute experiments headlessly using the batch CLI:

**Run a single experiment:**
```bash
python3 -m pdworld.adapters.batch.cli run-exp --exp 1 --seed 7
```
Options:
- `--exp {1,2,3}`: Experiment ID
- `--seed`: Random number generator seed (use 7 or 19 for this analysis)
- `--output`: Artifacts output root directory (default: `artifacts`)

**Run all experiments:**
```bash
python3 -m pdworld.adapters.batch.cli run-all --seeds 7 19
```
Options:
- `--seeds`: List of RNG seeds (use 7 and 19 for this analysis)
- `--output`: Artifacts output root directory (default: `artifacts`)

All experiment results (such as JSON metadata, run summaries, and CSV tables) are written to the configured output directory.

## 🧠 Learning Algorithms Explained

Reinforcement Learning agents learn to make decisions by navigating the grid, taking actions, and updating a "Q-Table" (a lookup table keeping track of the predicted reward or value for a state/action combination). This simulation offers two learning updates:

### 1. Q-Learning
Q-Learning is an **off-policy** algorithm. When updating the Q-value for the current state and action, it looks ahead at the *next state* and optimistically assumes it will take the *best possible action* (the one with the maximum Q-value), regardless of the current policy.

### 2. SARSA (State-Action-Reward-State-Action)
SARSA is an **on-policy** algorithm. When updating the Q-value for the current state and action, it looks at the *next state* and specifically considers the *next action* that the current policy actually chooses to take. It learns based on what the agent will really do, factoring in exploration behavior.

## 🕹️ Action Policies Explained

The "Policy" determines how the agent decides what to do at every step. It's a fundamental tradeoff between *Exploiting* (doing what you know works) and *Exploring* (trying new things to see if they're better).

### 1. PRANDOM (Random Policy)
The agent completely ignores the Q-Table. At every step, it chooses a random valid action.
- **Use Case:** High exploration, builds the Q-table evenly at the beginning, but never utilizes its knowledge to maximize reward.

### 2. PGREEDY (Greedy Policy)
The agent acts completely deterministically (mostly). It looks at the Q-table for its current state and *always* picks the action with the highest expected value. If there's a tie, it chooses randomly among the tied actions.
- **Use Case:** High exploitation, but it can easily get stuck in a "local maximum" if it hasn't explored the grid enough.

### 3. PEXPLOIT (Exploitation Policy)
This is an $\epsilon$-greedy approach. 
- **80% of the time:** It acts greedily (highest Q-value).
- **20% of the time:** It acts randomly.
- **Use Case:** The best of both worlds. It heavily leverages its knowledge while still occasionally deviating to explore the map and potentially discover better paths.
