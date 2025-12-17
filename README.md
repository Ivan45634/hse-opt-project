# Optimization Project

This repository contains the implementation of various optimization algorithms and their experimental validation on different tasks.

## Project Structure

- `src/`: Core library containing optimizer implementations (e.g., Muon, TAIA).
- `experiments/`: Jupyter notebooks with experimental setups and analysis.
  - `experiment_matrix_quadratic.ipynb`: Experiments on ill-conditioned quadratic problems.
  - `experiment_matrix_completion_movielens100k.ipynb`: Matrix completion on MovieLens 100K dataset.
  - `mushroom_invariance_test.ipynb`: Invariance testing on the Mushroom dataset.
- `Report.pdf`: Detailed project report.

## Installation

```bash
pip install -r requirements.txt
```

## Description

The project focuses on comparing and analyzing state-of-the-art optimizers. Key components include:

- **Muon**: Momentum Orthogonalized by Newton-Schulz optimization.
- **Experiments**: Verification of scale invariance, convergence analysis on quadratic problems, and performance benchmarking on matrix completion tasks.

Please refer to `Report.pdf` for a detailed analysis of the results and methodology.
