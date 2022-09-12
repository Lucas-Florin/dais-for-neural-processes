import numpy as np
import pandas as pd
import seaborn as sns
from scipy.special import logsumexp
from matplotlib import pyplot as plt
context_sizes = [0, 1, 2, 4, 5, 6, 8, 12, 16]

#seed_list = ["12", "123", "1234", "12345", "123456"]
seed_list = ["823"]

def plot_medians(data, estimators):
    assert data[0].ndim == 4 # (n_seeds, n_context, n_samples, n_task)
    n_estimators = len(estimators)

    median_log_likelihoods_list = []

    for estimator in range(n_estimators):
        median_log_likelihoods_list.append(average_over_seeds(data[estimator]))

    x = list(range(len(context_sizes)))
    plt.xticks(x, context_sizes)
    line_symbols = ['ro-', 'go-', 'bo-', 'yo-', 'mo-', 'co-']

    context_sizes = [0, 1, 2, 4, 8, 16]

    for k in range(n_estimators):
        plt.plot(x, median_log_likelihoods_list[k], f'{line_symbols[k]}', label=f'{estimators[k]}')
    
    plt.xlabel("context set size")
    plt.ylabel("log predictive likelihood")
    plt.legend()
    plt.title('Median log predictive likelihood estimate over tasks, averaged over 3 seeds')
    #plt.show()
    name = ""
    for estimator in estimators:
        name += estimator

    plt.savefig(f"median_estimate_{name}_5_seeds.png")

def average_over_seeds(data):
    n_seed = data.shape[0]
    n_context = data.shape[1]
    n_samples = data.shape[2]
    n_task = data.shape[3]

    log_likelihoods_per_task = logsumexp(data, axis=2) - np.log(n_samples)
    assert log_likelihoods_per_task.shape == (n_seed, n_context, n_task)
    mean_over_seed = logsumexp(log_likelihoods_per_task, axis=0) - np.log(n_seed)
    assert mean_over_seed.shape == (n_context, n_task)
    median_log_likelihoods_mc = np.median(mean_over_seed, axis=1)
    assert median_log_likelihoods_mc.shape == (n_context,)

    return median_log_likelihoods_mc

def average_over_samples_and_tasks(data):
    n_seed = data.shape[0]
    n_context = data.shape[1]
    n_samples = data.shape[2]
    n_task = data.shape[3]

    log_likelihoods_per_task = logsumexp(data, axis=2) - np.log(n_samples)
    assert log_likelihoods_per_task.shape == (n_seed, n_context, n_task)
    median_log_likelihoods_mc = np.median(log_likelihoods_per_task, axis=2)
    assert median_log_likelihoods_mc.shape == (n_seed, n_context)
    return median_log_likelihoods_mc

def plot_estimate_over_context_median(data):
    median_log_likelihoods_mc = average_over_seeds(data)
    
    xi = list(range(len(context_sizes)))
    plt.xticks(xi, context_sizes)
    plt.plot(median_log_likelihoods_mc, "o-")
    plt.xlabel("context set size")
    plt.ylabel("log marginal likelihood")
    plt.legend()
    plt.show()
def plot_estimate_over_context_box(data, estimator):
    n_seed = data.shape[0]
    n_context = data.shape[1]
    n_samples = data.shape[2]
    n_task = data.shape[3]
    log_likelihoods_per_task = logsumexp(data, axis=2) - np.log(n_samples)
    assert log_likelihoods_per_task.shape == (n_seed, n_context, n_task)
    mean_over_seed = logsumexp(log_likelihoods_per_task, axis=0) - np.log(n_seed)
    assert mean_over_seed.shape == (n_context, n_task)
    log_likelihoods = []
    for i in range(n_context):
        log_likelihoods.append(mean_over_seed[i])
    xi = list(range(len(context_sizes)))
    fig, ax = plt.subplots()
    ax.set_xticklabels(context_sizes)
    ax.set_xlabel("context set size")
    ax.set_ylabel("log marginal likelihood")
    ax.set_title(estimator)
    ax.boxplot(log_likelihoods, showfliers=False)
    plt.savefig(f"{estimator}_boxplot.png")

def plot_prediction_over_runs(data, estimators):
    n_estimators = len(estimators)
    assert data[0].ndim == 4 # (n_seeds, n_context, n_samples, n_task)
    n_seeds = data[0].shape[0]

    array = {"context set size": [], "log predictive likelihood": [], "estimator": []}
    df_lineplot = pd.DataFrame(array)

    for estimator in range(n_estimators):
        median_log_likelihoods_mc = average_over_samples_and_tasks(data[estimator])
        assert median_log_likelihoods_mc.ndim == 2 # (n_seeds, n_context)

        xData = np.tile(np.array(context_sizes), reps=n_seeds)
        yData = median_log_likelihoods_mc.flatten()

        array = {"context set size": xData, "log predictive likelihood": yData, "estimator": estimators[estimator]}
        df_tmp = pd.DataFrame(array)
        df_lineplot = df_lineplot.append(df_tmp, ignore_index = True)


    sns.lineplot(x="context set size", y="log predictive likelihood", data= df_lineplot, hue="estimator", marker="o")
    plt.show()

def main():
    #estimators = ["mc", "iwmc", "ais_small", "elbo"]
    estimators = ["mc", "iwmc"]

    data = [[] for x in range(len(estimators))]

    print(data)

    for i, estimator in enumerate(estimators):
        for seed in seed_list:
            data[i].append(np.load(f"log_likelihood_probs_VI_256_{estimator}_seed_{seed}.npy"))
        data[i] = np.array(data[i])

    print(len(data))
    print(data[0].shape)


    #plot_prediction_over_runs([data[0], data[0][:, :, :2000, :], data[0][:, :, :1000, :], data[2]], ["mc_10k", "mc_2k", "mc_1k", "ais_small"])
    plot_prediction_over_runs([data[0], data[1]], ["mc", "iwmc"])

    #plot_medians(data, estimators)
    #plot_medians([data[0], data[2]], ["mc", "ais_small"])
    
    #plot_medians([data[1], data[3]], ["iwmc", "elbo"])


    #for i in range(len(estimators)):
    #    plot_estimate_over_context_box(data[i], estimators[i])

if __name__ == "__main__":
    main()