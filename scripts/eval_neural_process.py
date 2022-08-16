from ast import Pass
from mimetypes import init
import os
import torch
torch.manual_seed(0)
import math
import numpy as np
import sys
from matplotlib import pyplot as plt
from metalearning_benchmarks import MetaLearningBenchmark
from metalearning_benchmarks import benchmark_dict as BM_DICT
from neural_process.neural_process import NeuralProcess
from pprint import pprint
from os import path

import logging
import wandb
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.normal import Normal

from bayesian_meta_learning.ais import ais_trajectory
from bayesian_meta_learning import utils

from eval_util.util import log_likelihood_mc_per_datapoint
from scipy.special import logsumexp

logger = logging.getLogger(__name__)

def lmlhd_mc(np_model: NeuralProcess, task, n_samples = 10):
    task_x, task_y = task # (n_tsk, n_tst, d_x), (n_tsk, n_tst, d_y)

    mu_z, var_z = np_model.aggregator.last_agg_state
    assert mu_z.ndim == 2 # (n_tsk, d_z)
    assert var_z.ndim == 2 # (n_tsk, d_z)

    n_tsk = mu_z.shape[0]
    d_z = mu_z.shape[1]
    n_points = task_x.shape[1]
    d_x = task_x.shape[2]
    d_y = task_y.shape[2]

    #TODO: Create function "sample num_samples from Normal(mu, var), mu and var must be torch tensors, num_samples must be Integer, we return torch tensor"
    latent_distribution =  Normal(mu_z, torch.sqrt(var_z))

    current_sample = latent_distribution.sample((n_samples,)) # (n_marg, n_tsk, d_z)
    logger.warning("shape of current_sample: " + str(current_sample.shape))

    current_sample = torch.unsqueeze(torch.transpose(current_sample, dim0=0, dim1=1), dim=1)
    assert current_sample.shape == (n_tsk, 1, n_samples, d_z)

    logger.warning("shape of current_sample: " + str(current_sample.shape))

    test_set_x = torch.from_numpy(task_x).float()

    assert test_set_x.ndim == 3  # (n_tsk, n_tst, d_x)
    assert current_sample.ndim == 4  # (n_tsk, n_ls, n_marg, d_z)


    mu_y, std_y = np_decode(np_model, test_set_x, current_sample) # shape will be (n_tsk, n_ls, n_marg, n_tst, d_y)

    #TODO: Create function that gives you lmlhd and lmlhd_per_sample for decoder output mu_y and std_y
    # (mu_y: torch.tensor, std_y: torch.tensor, y_true: np.ndarray) -> (lmlhd: np.ndarray, lmlhd_samples: np.ndarray)

    mu_y = torch.transpose(torch.squeeze(mu_y, dim=1), dim0=0, dim1=1).detach().numpy()
    std_y = torch.transpose(torch.squeeze(std_y, dim=1), dim0=0, dim1=1).detach().numpy()
    logger.warning("mu_y shape: " + str(mu_y.shape))
    #mu, var = np_model.predict(x=task.x) # predict returns variance!!

    assert mu_y.shape == (n_samples, n_tsk, n_points, d_y) # (n_samples, n_tasks, n_points, d_y)
    assert std_y.shape == (n_samples, n_tsk, n_points, d_y)# (n_samples, n_tasks, n_points, d_y)

    #lmlhd = log_marginal_likelihood_mc(mu_y, std_y, task_y)

    lmlhd = log_likelihood_mc_per_datapoint(y_pred=mu_y, sigma_pred=std_y, y_true=task_y)
    assert lmlhd.shape == (n_samples, n_tsk, n_points)
    lmlhd_samples = np.sum(lmlhd, axis=-1)  # points per task
    assert lmlhd_samples.shape == (n_samples, n_tsk)
    lmlhd = logsumexp(lmlhd_samples, axis=0) - math.log(n_samples)  # samples
    ## check output
    assert lmlhd.shape == (n_tsk,)

    return lmlhd, lmlhd_samples


def sample_normal(mu: torch.tensor, var: torch.tensor, num_samples: int) -> torch.tensor:
    assert mu.shape == var.shape
    #samples = ...
    #assert samples.shape == (num_samples, mu.shape)
    pass 

def get_dataset_likelihood(mu_y: torch.tensor, std_y: torch.tensor, y_true: np.ndarray):
    pass

def lmlhd_ais(np_model: NeuralProcess, task, n_samples = 10, chain_length=500):
    task_x, task_y = task # (n_tsk, n_tst, d_x), (n_tsk, n_tst, d_y)
    assert task_x.ndim == 3
    assert task_y.ndim == 3
    assert isinstance(task_x, np.ndarray)
    assert isinstance(task_y, np.ndarray)

    task_x_torch = torch.from_numpy(task_x).float()
    task_y_torch = torch.from_numpy(task_y).float()

    log_prior = construct_log_prior(np_model, n_samples)
    log_posterior = construct_log_posterior(np_model, log_prior, task_x_torch, task_y_torch)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    forward_schedule = torch.linspace(0, 1, chain_length, device=device)

    # initial state should have shape n_samples x n_task x d_z, last_agg_state has dimension n_task x d_z
    mu_z, var_z = np_model.aggregator.last_agg_state
    mu_z = mu_z.repeat(n_samples, 1)
    var_z = var_z.repeat(n_samples, 1)

    n_tsk = task_x.shape[0]
    d_z = mu_z.shape[-1]

    initial_state = torch.normal(mu_z, torch.sqrt(var_z))
    assert initial_state.shape == (n_samples * n_tsk, d_z)

    return ais_trajectory(log_prior, log_posterior, initial_state, n_samples=n_samples, forward=True, schedule = forward_schedule, initial_step_size = 0.01, device = device)

# TODO: make method accept proposal and target distribution (method should not change model state)
def lmlhd_iwmc(np_model: NeuralProcess, task, n_samples = 10):
    task_x, task_y = task # (n_tsk, n_tst, d_x), (n_tsk, n_tst, d_y)
    mu_z_context, var_z_context = np_model.aggregator.last_agg_state

    n_tsk = mu_z_context.shape[0]
    d_z = mu_z_context.shape[1]
    n_points = task_x.shape[1]
    d_x = task_x.shape[2]
    d_y = task_y.shape[2]

    # mu_z and var_z have shape n_task x d_z. We need mean to have shape n_samples x n_tsk x d_z

    mu_z_context = mu_z_context.repeat(n_samples, 1, 1)
    std_z_context = torch.sqrt(var_z_context.repeat(n_samples, 1, 1))
    assert mu_z_context.shape == (n_samples, n_tsk, d_z)
    assert std_z_context.shape == (n_samples, n_tsk, d_z)

    # Generate samples from prior adapted on larger context set (Largest context set that the model has seen during meta-training)
    _, max_context_size = np_model.settings["n_context_meta"]
    np_model.adapt(x = task_x[:, :max_context_size, :], y = task_y[:, :max_context_size, :])
    
    mu_z_target, var_z_target = np_model.aggregator.last_agg_state

    # mu_z and var_z have shape n_task x d_z. We need mean to have shape n_samples x n_task x d_z
    # TODO: Use new function to sample from normal
    mu_z_target = mu_z_target.repeat(n_samples, 1, 1)
    std_z_target = torch.sqrt(var_z_target.repeat(n_samples, 1, 1))
    assert mu_z_target.shape == (n_samples, n_tsk, d_z)
    assert std_z_target.shape == (n_samples, n_tsk, d_z)

    #logger.warning("mu_z context: " + str(mu_z_context[0, :, :]) + ", std_z_context: " + str(std_z_context[0, :, :]))
    #logger.warning("mu_z target: " + str(mu_z_target[0, :, :]) + ", std_z_context: " + str(std_z_target[0, :, :]))
    
    target_z_samples = Normal(mu_z_target, std_z_target).sample() 
    assert target_z_samples.shape == (n_samples, n_tsk, d_z)
    
    # Evaluate probabilities of z samples in both distributions
    log_context_probs = torch.sum(Normal(mu_z_context, std_z_context).log_prob(target_z_samples), dim=-1)
    log_target_probs = torch.sum(Normal(mu_z_target, std_z_target).log_prob(target_z_samples), dim=-1)
    assert log_context_probs.shape == (n_samples, n_tsk)
    assert log_target_probs.shape == (n_samples, n_tsk)

    # Compute likelihood on target set for every latent sample
    target_z_samples = torch.unsqueeze(torch.transpose(target_z_samples, dim0=0, dim1=1), dim=1)
    test_set_x = torch.from_numpy(task_x).float()
    test_set_y = torch.from_numpy(task_y).float()

    assert target_z_samples.shape == (n_tsk, 1, n_samples, d_z)
    assert test_set_x.shape == (n_tsk, n_points, d_x)
    assert test_set_y.shape == (n_tsk, n_points, d_y)

    mu_y, std_y = np_decode(np_model, test_set_x, target_z_samples)

    assert mu_y.shape == (n_tsk, 1, n_samples, n_points, d_y)
    assert std_y.shape == (n_tsk, 1, n_samples, n_points, d_y)

    mu_y = torch.transpose(torch.squeeze(mu_y, dim=1), dim0=0, dim1=1).detach().numpy()
    std_y = torch.transpose(torch.squeeze(std_y, dim=1), dim0=0, dim1=1).detach().numpy()
    assert mu_y.shape == (n_samples, n_tsk, n_points, d_y)
    assert std_y.shape == (n_samples, n_tsk, n_points, d_y)

    log_lhds_per_datapoint = log_likelihood_mc_per_datapoint(y_pred=mu_y, sigma_pred=std_y, y_true=task_y)
    assert log_lhds_per_datapoint.shape == (n_samples, n_tsk, n_points)
    log_lhds_samples = np.sum(log_lhds_per_datapoint, axis=-1)  # points per task
    assert log_lhds_samples.shape == (n_samples, n_tsk)

    log_importance_weights = (log_context_probs - log_target_probs).detach().numpy()
    assert log_importance_weights.shape == (n_samples, n_tsk)

    log_lhd = logsumexp(np.add(log_lhds_samples, log_importance_weights), axis=0) - np.log(n_samples)
    assert log_lhd.shape == (n_tsk,)
    return log_lhd, np.add(log_lhds_samples, log_importance_weights), log_lhds_samples, log_importance_weights

def np_decode(np_model, x, z):
    assert x.ndim == 3  # (n_tsk, n_tst, d_x)
    assert z.ndim == 4  # (n_tsk, n_ls, n_marg, d_z)

    x = np_model._normalize_x(x)
    mu_y, std_y = np_model.decoder.decode(x, z) # decoder returns std!!
    mu_y = np_model._denormalize_mu_y(mu_y)

    return mu_y, std_y

def estimates_over_time(np_model, task, max_samples):
    _, log_likelihood_probs_mc = lmlhd_mc(np_model, task, n_samples = max_samples)

    mu_z_context, var_z_context = np_model.aggregator.last_agg_state

    # In the future, this method will not change the model state
    _, log_likelihood_probs_iwmc, log_lhds_samples, log_importance_weights = lmlhd_iwmc(np_model, task, n_samples = max_samples)

    task_x, task_y = task # (n_tsk, n_tst, d_x), (n_tsk, n_tst, d_y)
    n_tsk = task_x.shape[0]
    n_points = task_x.shape[1]
    d_y = task_y.shape[-1]
    d_z = mu_z_context.shape[-1]
    
    assert log_likelihood_probs_mc.shape == (max_samples, n_tsk)
    assert log_likelihood_probs_iwmc.shape == (max_samples, n_tsk)
    
    # TODO: Create function plot_likelihoods_box: (np_model, task, distribution1, distribution2, num_samples) that generates a boxplot of likelihood distributions
    mu_z_context = mu_z_context.repeat(max_samples, 1, 1)
    std_z_context = torch.sqrt(var_z_context.repeat(max_samples, 1, 1))
    assert mu_z_context.shape == (max_samples, n_tsk, d_z)
    assert std_z_context.shape == (max_samples, n_tsk, d_z)
    
    context_z_samples = Normal(mu_z_context, std_z_context).sample() 
    assert context_z_samples.shape == (max_samples, n_tsk, d_z)

    context_z_samples = torch.unsqueeze(torch.transpose(context_z_samples, dim0=0, dim1=1), dim=1)

    test_set_x = torch.from_numpy(task_x).float()

    mu_y, std_y = np_decode(np_model, test_set_x, context_z_samples)

    assert mu_y.shape == (n_tsk, 1, max_samples, n_points, d_y)
    assert std_y.shape == (n_tsk, 1, max_samples, n_points, d_y)

    mu_y = torch.transpose(torch.squeeze(mu_y, dim=1), dim0=0, dim1=1).detach().numpy()
    std_y = torch.transpose(torch.squeeze(std_y, dim=1), dim0=0, dim1=1).detach().numpy()
    assert mu_y.shape == (max_samples, n_tsk, n_points, d_y)
    assert std_y.shape == (max_samples, n_tsk, n_points, d_y)

    log_lhds_per_datapoint = log_likelihood_mc_per_datapoint(y_pred=mu_y, sigma_pred=std_y, y_true=task_y)
    assert log_lhds_per_datapoint.shape == (max_samples, n_tsk, n_points)
    log_lhds_samples_context = np.sum(log_lhds_per_datapoint, axis=-1)  # points per task

    plt.boxplot([log_lhds_samples_context[:, 0], log_lhds_samples[:, 0]])
    plt.title("Likelihoods of latent samples from different distributions")
    plt.xticks([1, 2], ["context (0 datapoints)", "target (32 datapoints)"])
    plt.show()

    # TODO: Create function plot_weights_likelihoods (np_model, task, distribution1, distribution2, num_samples) that plots importance weights and likelihoods per latent sample
    '''
    x = np.arange(0, max_samples)
    plt.plot(x, log_lhds_samples[:, 0], 'ro-', label='Log Likelihoods')
    plt.plot(x, log_importance_weights[:, 0], 'go-', label='Log Importance Weights')
    plt.xlabel("Samples")
    plt.legend()
    plt.show()
    '''

    log_likelihoods_mc = []
    log_likelihoods_iwmc = []
    for i in range(3, int(np.log(max_samples)) + 1):
        log_likelihoods_mc_per_task = logsumexp(log_likelihood_probs_mc[:int(np.exp(i)), :], axis=0) - i
        assert log_likelihoods_mc_per_task.shape == (n_tsk,)
        log_likelihoods_mc.append((logsumexp((log_likelihoods_mc_per_task), axis=0) - np.log(n_tsk)).item())

        log_likelihoods_iwmc_per_task = logsumexp(log_likelihood_probs_iwmc[:int(np.exp(i)), :], axis=0) - i
        assert log_likelihoods_iwmc_per_task.shape == (n_tsk,)
        log_likelihoods_iwmc.append((logsumexp(log_likelihoods_iwmc_per_task, axis=0) - np.log(n_tsk)).item())

    logger.warning("log_likelihoods_mc: " + str(log_likelihoods_mc))
    logger.warning("log_likelihoods_iwmc: " + str(log_likelihoods_iwmc))
    x = np.arange(3, int(np.log(max_samples)) + 1)
    plt.plot(x, log_likelihoods_mc, 'ro-', label='MC')
    plt.plot(x, log_likelihoods_iwmc, 'go-', label='IWMC')
    plt.xlabel("Log num samples")
    plt.ylabel("log predictive likelihood estimate")
    plt.legend()
    plt.title('Estimates over time, n_context_points = 4')
    plt.show()

def collate_benchmark(benchmark: MetaLearningBenchmark):
    # collate test data
    x = np.zeros((benchmark.n_task, benchmark.n_datapoints_per_task, benchmark.d_x))
    y = np.zeros((benchmark.n_task, benchmark.n_datapoints_per_task, benchmark.d_y))
    for l, task in enumerate(benchmark):
        x[l] = task.x
        y[l] = task.y

    return x, y

def plot(
    np_model: NeuralProcess,
    benchmark: MetaLearningBenchmark,
    n_task_max: int,
    n_context_points: int,
    fig,
    axes,
):
    # determine n_task
    n_task_plot = min(n_task_max, benchmark.n_task)

    # evaluate predictions
    n_samples = 500
    x_min = benchmark.x_bounds[0, 0]
    x_max = benchmark.x_bounds[0, 1]
    x_plt_min = x_min - 0.1 * (x_max - x_min)
    x_plt_max = x_max + 0.1 * (x_max - x_min)
    x_plt = np.linspace(x_plt_min, x_plt_max, 128)
    x_plt = np.reshape(x_plt, (-1, 1))

    x = np.zeros((n_task_plot, benchmark.n_datapoints_per_task, benchmark.d_x))
    y = np.zeros((n_task_plot, benchmark.n_datapoints_per_task, benchmark.d_y))
    for k in range(0, n_task_plot):
        task = benchmark.get_task_by_index(k)
        x[k] = task.x
        y[k] = task.y

    # plot predictions
    for i in range(0, n_context_points + 1):
        np_model.adapt(x=x[:, :i, :], y=y[:, :i, :]) # adapt model on context set of size i

        mu, _ = np_model.predict(x=x_plt, n_samples=n_samples)
        assert mu.shape == (n_task_plot, n_samples, x_plt.shape[0], benchmark.d_y)

        lmlhd_ais_estimates = lmlhd_ais(np_model, (x, y), n_samples = 10, chain_length=10000)
        lmlhd_mc_estimates, _ = lmlhd_mc(np_model, (x, y), n_samples = 10000)
        lmlhd_iwmc_estimates, _, _, _ = lmlhd_iwmc(np_model, (x, y), n_samples = 10000)
        assert lmlhd_ais_estimates.shape == (n_task_plot,)
        assert lmlhd_mc_estimates.shape == (n_task_plot,)
        assert lmlhd_iwmc_estimates.shape == (n_task_plot,)

        for l in range(n_task_plot):
            ax = axes[i, l]
            ax.clear()
            ax.scatter(x[l, i:, :], y[l, i:, :], s=15, color="g", alpha=1.0, zorder=3)
            ax.scatter(x[l, :i, :], y[l, :i, :], marker="x", s=30, color="r", alpha=1.0, zorder=3)
            
            for s in range(n_samples):
                ax.plot(x_plt, mu[l, s, :, :], color="b", alpha=0.3, label="posterior", zorder=2)
            
            print("number of context points: " + str(i) + ", task: " + str(l))
            print("MC estimate: " + str(round(lmlhd_mc_estimates[l].item(), 3)))
            print("AIS estimate: " + str(round(lmlhd_ais_estimates[l].item(), 3)))
            print("IWMC estimate: " + str(round(lmlhd_iwmc_estimates[l].item(), 3)))

            ax.grid(zorder=1)
            if(i == 0):
                ax.set_title(f"Predictions (Task {l:d})")
            ax.text(0, 0, "MC: " + str(round(lmlhd_mc_estimates[l].item(), 3)) + ", AIS: " + str(round(lmlhd_ais_estimates[l].item(), 3)) + ", IWMC: " + str(round(lmlhd_iwmc_estimates[l].item(), 3)), horizontalalignment='left', verticalalignment='bottom', transform=ax.transAxes)

    fig.tight_layout()
    plt.show(block=False)
    plt.pause(0.001)

def construct_log_prior(model, n_samples):
    mu_z, var_z = model.aggregator.last_agg_state
    assert mu_z.ndim == 2 # (n_tsk, d_z)
    assert var_z.ndim == 2 # (n_tsk, d_z)
    # mu_z and var_z have shape n_task x d_z. We need mean and var to have shape (n_samples * n_task) x d_z
    mu_z = mu_z.repeat(n_samples, 1)
    std_z = torch.sqrt(var_z.repeat(n_samples, 1))
    return lambda z : torch.sum(Normal(mu_z, std_z).log_prob(z), dim=-1)

def construct_log_posterior(model, log_prior, test_set_x, test_set_y):
    return lambda z: log_prior(z) + log_likelihood_fn(model, test_set_x, test_set_y, z)

def log_likelihood_fn(model, test_set_x, test_set_y, z):
    assert test_set_x.ndim == 3  # (n_tsk, n_tst, d_x)
    assert test_set_y.ndim == 3  # (n_tsk, n_tst, d_y)
    assert z.ndim == 2  # (n_samples * n_tsk, d_z)
    n_tsk = test_set_x.shape[0]
    n_samples = z.shape[0] // n_tsk
    d_z = z.shape[1]

    z_decoder = torch.reshape(z, (n_samples, n_tsk, d_z))
    z_decoder = torch.unsqueeze(z_decoder.transpose(dim0=0, dim1=1), dim=1) # (n_tsk, n_ls, n_samples, d_z)
    assert z_decoder.shape == (n_tsk, 1, n_samples, d_z)
    mu_y, std_y = np_decode(model, test_set_x, z_decoder) # shape will be (n_tsk, n_ls, n_samples, n_tst, d_y)
    mu_y = torch.squeeze(mu_y, dim=1) # shape will be (n_tsk, n_samples, n_tst, d_y)
    std_y = torch.squeeze(std_y, dim=1) # shape will be (n_tsk, n_samples, n_tst, d_y)
    assert mu_y.ndim == 4
    assert std_y.ndim == 4
    test_set_y = torch.unsqueeze(test_set_y, dim=1).repeat(1, n_samples, 1, 1) # shape will be (n_tsk, n_samples, n_tst, d_y)
    result = torch.sum(Normal(mu_y, std_y).log_prob(test_set_y), dim=[2,3]) # sum over d_y and over data points per task
    result = result.transpose(dim0=0, dim1=1)
    result = torch.reshape(result, (-1,))
    assert result.shape == (n_samples * n_tsk,)
    return result

def train(model, benchmark_meta, benchmark_val, benchmark_test, config):
    # Log in to your W&B account
    wandb.login()
    wandb.init(
      # Set the project where this run will be logged
      project="Eval Neural Process", 
      # We pass a run name (otherwise it’ll be randomly assigned, like sunshine-lollypop-10)
      name=f"experiment_3", 
      config={"n_tasks_train": config["n_tasks_train"], "n_hidden_units": config["n_hidden_units"]}
    )

    log_loss = lambda n_meta_tasks_seen, np_model, metrics: wandb.log(metrics) if metrics is not None else None

    # callback switched to None for much faster meta-training during debugging
    model.meta_train(
        benchmark_meta=benchmark_meta,
        benchmark_val=benchmark_val,
        n_tasks_train=config["n_tasks_train"],
        validation_interval=config["validation_interval"],
        callback=log_loss,
    )

    # Mark the run as finished
    wandb.finish()
    model.save_model()

def main():
    # logpath
    logpath = os.path.dirname(os.path.abspath(__file__))
    logpath = os.path.join(logpath, os.path.join("..", "log"))
    os.makedirs(logpath, exist_ok=True)

    ## config
    config = dict()
    # model and benchmark
    config["model"] = "StandardNP"
    config["benchmark"] = "Quadratic1D"
    # logging
    config["logpath"] = logpath
    # seed
    config["seed"] = 1234
    # meta data
    config["data_noise_std"] = 0.1
    config["n_task_meta"] = 256
    config["n_datapoints_per_task_meta"] = 64
    config["seed_task_meta"] = 1234
    config["seed_x_meta"] = 2234
    config["seed_noise_meta"] = 3234
    # validation data
    config["n_task_val"] = 16 
    config["n_datapoints_per_task_val"] = 64
    config["seed_task_val"] = 1236
    config["seed_x_val"] = 2236
    config["seed_noise_val"] = 3236
    # test data
    config["n_task_test"] = 256 
    config["n_datapoints_per_task_test"] = 64
    config["seed_task_test"] = 1235
    config["seed_x_test"] = 2235
    config["seed_noise_test"] = 3235

    # generate benchmarks
    benchmark_meta = BM_DICT[config["benchmark"]](
        n_task=config["n_task_meta"],
        n_datapoints_per_task=config["n_datapoints_per_task_meta"],
        output_noise=config["data_noise_std"],
        seed_task=config["seed_task_meta"],
        seed_x=config["seed_x_meta"],
        seed_noise=config["seed_noise_meta"],
    )
    benchmark_val = BM_DICT[config["benchmark"]](
        n_task=config["n_task_val"],
        n_datapoints_per_task=config["n_datapoints_per_task_val"],
        output_noise=config["data_noise_std"],
        seed_task=config["seed_task_val"],
        seed_x=config["seed_x_val"],
        seed_noise=config["seed_noise_val"],
    )
    benchmark_test = BM_DICT[config["benchmark"]](
        n_task=config["n_task_test"],
        n_datapoints_per_task=config["n_datapoints_per_task_test"],
        output_noise=config["data_noise_std"],
        seed_task=config["seed_task_test"],
        seed_x=config["seed_x_test"],
        seed_noise=config["seed_noise_test"],
    )

    # architecture
    config["d_x"] = benchmark_meta.d_x
    config["d_y"] = benchmark_meta.d_y
    config["d_z"] = 16
    config["aggregator_type"] = "BA"
    config["loss_type"] = "MC"
    config["input_mlp_std_y"] = ""
    config["self_attention_type"] = None
    config["f_act"] = "relu"
    config["n_hidden_layers"] = 2
    config["n_hidden_units"] = 64
    config["latent_prior_scale"] = 1.0
    config["decoder_output_scale"] = config["data_noise_std"]

    # training
    config["n_tasks_train"] = int(2**19)
    config["validation_interval"] = config["n_tasks_train"] // 4
    config["device"] = "cuda"
    config["adam_lr"] = 1e-4
    config["batch_size"] = config["n_task_meta"]
    config["n_samples"] = 16
    config["n_context"] = [1, config["n_datapoints_per_task_meta"] // 2,]

    # generate NP model
    model = NeuralProcess(
        logpath=config["logpath"],
        seed=config["seed"],
        d_x=config["d_x"],
        d_y=config["d_y"],
        d_z=config["d_z"],
        n_context=config["n_context"],
        aggregator_type=config["aggregator_type"],
        loss_type=config["loss_type"],
        input_mlp_std_y=config["input_mlp_std_y"],
        self_attention_type=config["self_attention_type"],
        latent_prior_scale=config["latent_prior_scale"],
        f_act=config["f_act"],
        n_hidden_layers=config["n_hidden_layers"],
        n_hidden_units=config["n_hidden_units"],
        decoder_output_scale=config["decoder_output_scale"],
        device=config["device"],
        adam_lr=config["adam_lr"],
        batch_size=config["batch_size"],
        n_samples=config["n_samples"],
    )
    # If there is no trained model to load, the model can be trained with the following line
    #train(model, benchmark_meta, benchmark_val, benchmark_test, config)
    
    model.load_model(config["n_tasks_train"])
    
    n_task_plot = 4
    n_context_points = 4

    fig, axes = plt.subplots(
        nrows=n_context_points + 1,
        ncols=n_task_plot,
        sharex=True,
        sharey=True,
        squeeze=False,
        figsize=(5 * n_task_plot, 11),
    )

    fig.subplots_adjust(wspace=1, hspace=-100)

    plot(
        np_model=model,
        n_task_max=n_task_plot,
        benchmark=benchmark_test,
        n_context_points = n_context_points,
        fig=fig,
        axes=axes,
    )
    fig.savefig('temp.png', dpi=fig.dpi)
    fig.savefig('temp.pdf')
    plt.show()

if __name__ == "__main__":
    main()
