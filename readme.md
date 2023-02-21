# Bayesian Meta-Learning
This repository has the code for the Master's thesis "Differentiable Annealed Importance Sampling for Neural Processes" by Lucas Florin. 
It also contains code created by Kolja Bauer and Le Yang for the Praktikum Project "Evaluation of the log marginal predictive likelihood for Bayesian learning"

## Getting Started
Clone this repository and run
```
pip install . 
```
from the source directory.

Create and activate the Conda environment: 
```
conda env create --file environment.yml
conda activate alr
```

Get and install the following repositories: 
- [`cw2`](https://github.com/ALRhub/cw2) (Until pull request is accepted use the branch `dev-lucas`)
- [`sweepwork`](https://github.com/ALRhub/sweepwork)
- [`metalearning_benchmarks`](https://github.com/michaelvolpp/metalearning_benchmarks)
- [`metalearning_eval_util`](https://github.com/michaelvolpp/metalearning_eval_util) (request access)
- [`neural_process`](https://github.com/michaelvolpp/neural_process) (branch `dais`) or [this repo](https://github.com/Lucas-Florin/dais_np)



## Run locally
```
python scripts/baseline.py scripts/quadratic_mc.yaml -o -e quadratic_mc
```
Each YAML file represents a different experiment. 

## Run on bwUniCuster
These are the steps to replicate the experiments in Lucas Florin's thesis. 
Each YAML file represents a different experiment. Here we use the experiment training on the Quadratic1D benchmark with likleihood weighting loss. 

1. Run HPO sweep with 240 HP combinations: 
```
python scripts/sweep.py scripts/quadratic_mc.yaml -s -o -e quadratic_mc_sweep
```

2. Repeat the 5 best HP combinations with 5 seeds each: 
```
python scripts/baseline.py scripts/quadratic_mc.yaml -s -o -e quadratic_mc_repeat
```

3. Repeat the best HP combination with a new test set: 
```
python scripts/baseline.py scripts/quadratic_mc.yaml -s -o -e quadratic_mc_repeat_test
```

4. Train the final model: 
```
python scripts/baseline.py scripts/quadratic_mc.yaml -s -o -e quadratic_mc
```

5. Run evaluations with more compute:
```
python scripts/baseline.py scripts/quadratic_mc.yaml -s -o -e quadratic_mc_long
```
Here, you have to uncomment and change the lines relevant to the desired evaluation method. 

## Generate Plots
Use the functions in `scripts/thesis_plots.py` to generate the plots shown in the thesis. 
Use the functions in `scripts/wandb_utils.py` to get the relevant runs from WandB. 