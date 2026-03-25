# Reinforcement Learning — Text Flappy Bird

Comparing tabular Monte Carlo and Semi-Gradient SARSA(λ) on Text Flappy Bird to study how state space size, generalisation, and hyperparameter interactions drive algorithm selection in practice.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![gymnasium](https://img.shields.io/badge/gymnasium-0.29.0-orange.svg)](https://gymnasium.farama.org/)

---

## Key Results

**Greedy Evaluation — 200 Episodes (ε = 0):**

| Agent | Config | Mean Score | Max Score | Std |
|-------|--------|-----------|-----------|-----|
| MC | γ=0.99 | 8.9 | 13 | 3.6 |
| SARSA(λ) | α=0.1, λ=0.9 | 185.0 | 500 | 136.5 |
| SARSA(λ) best | α=0.01, λ=0.9 | 473.3 | 500 | 84.2 |

**Key finding:** A critical α–λ interaction governs SARSA(λ) performance — eligibility traces amplify weight updates, requiring a conservative step size (α=0.01) to remain stable with λ=0.9. Reducing α from 0.1 to 0.01 improves mean greedy score from 185 to 473.

**Generalisation (trained on gap=4, evaluated without retraining):**

| Config | MC mean | SARSA(λ) mean |
|--------|---------|---------------|
| Train (gap=4) | 9.0 | 478.9 |
| A: gap=6 (easier) | 8.8 | 500.0 |
| B: gap=2 (harder) | 8.6 | 22.6 |
| C: h=20, w=25 (larger grid) | 4.0 | 228.1 |

---

## Methodology

```
TextFlappyBird-v0  (dx, dy observation, 308 states)
        |
        ├── Agent 1: GLIE Every-Visit MC
        │     ├── Q-table: defaultdict (dx,dy) → [Q_fall, Q_flap]
        │     ├── ε = 1/k (GLIE decay)
        │     └── Update: incremental mean return per episode
        │
        └── Agent 2: Semi-Gradient SARSA(λ)
              ├── Tile coding: 8 tilings × 8×8 tiles → 512-dim features
              ├── ε = 0.1 (fixed), weight clipping at ±1e6
              └── Update: TD error + eligibility trace per step
                         w ← w + α·δ·z
```

**Parameter sweeps:** λ ∈ {0, 0.5, 0.9, 0.99}, α ∈ {0.01, 0.05, 0.1, 0.3, 0.5}, γ ∈ {0.5, 0.8, 0.9, 0.95, 0.99}

---

## Project Structure

```
├── rl_text_flappy_bird.ipynb     # Main notebook (all sections)
├── results/
│   ├── figures/                  # All plots (learning curves, Q-value heatmaps, sweeps)
│   └── metrics/
│       └── sweep_results.pkl     # Serialised parameter sweep results
├── report/
│   └── rl_report.pdf             # 3-page Springer LNCS report
├── requirements.txt
└── README.md
```

---

## Quick Start

**Prerequisites:** Python 3.10+

<details>
<summary><b>Setup — Click to expand</b></summary>

### Step 1: Clone and create environment
```bash
git clone https://github.com/wengchienwei/rl-text-flappy-bird.git
cd rl-text-flappy-bird
python -m venv venv
source venv/bin/activate  # Mac/Linux
# OR: venv\Scripts\activate  (Windows)
```

### Step 2: Install dependencies
```bash
pip install -r requirements.txt
pip install git+https://gitlab-research.centralesupelec.fr/stergios.christodoulidis/text-flappy-bird-gym.git
```

### Step 3: Run the notebook
```bash
jupyter lab
```
Open `rl_text_flappy_bird.ipynb` and run all cells top to bottom.

**Note:** Full rerun (all training + sweeps) takes approximately 20–30 minutes. Sweep results are cached in `results/metrics/sweep_results.pkl` — load them with the recovery cell in Section 4 to skip retraining.

</details>

<details>
<summary><b>Troubleshooting — Click to expand</b></summary>

**Issue:** `ModuleNotFoundError: No module named 'gym'`
- **Solution:** The package uses `gymnasium`. Ensure `import gymnasium as gym` in all cells.

**Issue:** Evaluation hangs indefinitely
- **Solution:** `max_steps=500` is set in `evaluate()`. If missing, add it to prevent infinite episodes when the greedy policy becomes near-perfect.

**Issue:** `RuntimeWarning: overflow encountered in matmul`
- **Solution:** Weight clipping `np.clip(self.w, -1e6, 1e6)` is applied after each update in `SarsaLambdaAgent`. This warning appears only at large α values (≥0.3) and is handled gracefully.

</details>

---

## Notebook Structure

| Section | Content |
|---------|---------|
| 0 | Environment setup and observation analysis |
| 1 | State representation — Q-table and tile coding |
| 2 | GLIE Every-Visit MC agent |
| 3 | Semi-Gradient SARSA(λ) agent |
| 4 | Head-to-head comparison and parameter sweeps |
| 5 | Generalisation test across environment configurations |
| 6 | screen-v0 analysis and original Flappy Bird compatibility |

---

## Tech Stack

**Core:** Python 3.10, gymnasium 0.29.0, numpy, matplotlib  
**Environment:** [Text Flappy Bird Gym](https://gitlab-research.centralesupelec.fr/stergios.christodoulidis/text-flappy-bird-gym)

---

## Documentation

Full results, analysis, and discussion available in [`report/rl_report.pdf`](report/rl_report.pdf).

---

## Author

**Chien-Wei WENG**  
MSc Data Sciences and Business Analytics  
CentraleSupélec × ESSEC Business School

[GitHub](https://github.com/wengchienwei) | [LinkedIn](https://www.linkedin.com/in/chien-wei-weng-74a6881b8/)

---

**Academic Project | Reinforcement Learning (2026)**
