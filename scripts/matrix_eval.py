import argparse
import numpy as np

from matplotlib import pyplot as plt
from scipy.special import logsumexp

parser = argparse.ArgumentParser(description='Evaluate metrics from given data matrix')
parser.add_argument("matrix_path")
args = parser.parse_args()

def load_matrix(path):
    return np.load(path)

def eval_matrix(matrix):
    matrix = matrix[:, :10000, :]
    assert matrix.ndim == 3 # (n_context_sizes, n_samples, n_tasks)
    n_samples = matrix.shape[1]

    slicing_indices = [0, 1, 2, 4] # corresponding to context set sizes [0, 1, 2, 5]

    eval_matrix = np.take(matrix, slicing_indices, axis=0)
    likelihoods_estimates = logsumexp(eval_matrix, axis=1) - np.log(n_samples)
    median_likelihoods = np.median(likelihoods_estimates, axis=1)
    objective = np.mean(median_likelihoods)

    print("objective function: " + str(objective))

def estimates_over_time(matrix):
    assert matrix.ndim == 3 # (n_context_sizes, n_samples, n_tasks)
    n_context_sizes = matrix.shape[0]
    n_samples = matrix.shape[1]
    n_tasks = matrix.shape[2]

    print(n_samples)

    slicing_indices = [0, 1, 2, 3, 6, 8] # corresponding to context set sizes [0, 1, 2, 4, 8, 16]

    log_likelihoods = [[] for x in range(n_context_sizes)]

    start_idx = 4

    for j in range(start_idx, int(np.log(n_samples)) + 1):
        # logsumexp over the samples
        log_likelihoods_per_task = logsumexp(matrix[:, :int(np.exp(j)), :], axis=1) - j
        assert log_likelihoods_per_task.shape == (n_context_sizes, n_tasks)

        for k, l in enumerate(slicing_indices):
            # take median across tasks
            median_log_likelihoods = np.median(log_likelihoods_per_task, axis=1)
            assert median_log_likelihoods.shape == (n_context_sizes,)
            log_likelihoods[k].append(median_log_likelihoods[l].item())

    x = np.arange(start_idx, int(np.log(n_samples)) + 1)
    line_symbols = ['o-', 'o--', 'o-.', 'o:', 'v-', '+-']

    context_sizes = [0, 1, 2, 4, 8, 16]

    for k in range(0, len(slicing_indices)):
        plt.plot(x, log_likelihoods[k], f'r{line_symbols[k]}', label=f'MC, {context_sizes[k]} context points')
    
    plt.xlabel("Log num samples")
    plt.ylabel("log predictive likelihood estimate")
    plt.legend()
    plt.title('Estimates over time')
    plt.show()
    plt.savefig("plot.png")


if __name__ == "__main__":
    matrix = load_matrix(args.matrix_path)
    eval_matrix(matrix)
    estimates_over_time(matrix)