# RL4RLA (ICML 2026)

Official code for **RL4RLA: Teaching ML to Discover Randomized Linear Algebra Algorithms through Curriculum Design and Graph-based Search**.

**Paper:** [ICML 2026](https://arxiv.org/abs/2605.18004)

**Project Page:** https://www.jinglongxiong.xyz/RL4RLA_page/

This repo runs batch experiments for Monte Carlo Graph Search (MCGS) and MCTS over symbolic linear-algebra primitive programs (configs under `hp/`).

## Requirements

Python 3.9 (tested).

## Setup

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

Run experiments using YAML configuration files in the `hp/` directory:

```bash
# Run all experiments in a directory
python run_batch.py hp/mcgs_ucd/

# Run specific config files
python run_batch.py hp/mcts/leverage_score_subsampling/0.yml

# Run with parallelism (4 experiments at once)
python run_batch.py hp/mcgs_ucd/ --parallel 4

# Repeat each experiment 5 times with different seeds
python run_batch.py hp/mcgs_ucd/ --repeat 5

# Retry failed experiments up to 3 times
python run_batch.py hp/mcgs_ucd/ --retry 3

# Dry run (see what would be executed)
python run_batch.py hp/mcgs_ucd/ --dry-run
```

## Configuration

Experiment configurations are YAML files in `hp/` organized by:
- `hp/mcgs_ucd/` - MCGS with UCD policy
- `hp/mcgs_uct/` - MCGS with UCT policy  
- `hp/mcts/` - MCTS experiments

Results are logged to `logs/` directory (auto-created based on config path).
