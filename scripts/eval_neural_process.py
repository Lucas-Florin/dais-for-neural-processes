from ast import Pass
import os
import torch
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

from metalearning_eval_util.util import log_marginal_likelihood_mc

logger = logging.getLogger(__name__)

def lmlhd_mc(np_model: NeuralProcess, task, n_samples = 10):
    y_true = np.expand_dims(task.y, axis=0) # shape needs to be n_tasks x n_points x d_y

    mu_z, var_z = np_model.aggregator.last_agg_state
    assert mu_z.ndim == 2 # (n_tsk, d_z)
    assert var_z.ndim == 2 # (n_tsk, d_z)
    mu_z = torch.squeeze(mu_z)
    var_z = torch.squeeze(var_z)
    latent_distribution =  Normal(mu_z, torch.sqrt(var_z))

    current_sample = latent_distribution.sample((n_samples,))
    logger.warning("shape of current_sample: " + str(current_sample.shape))
    current_sample = torch.unsqueeze(torch.unsqueeze(current_sample, dim=0), dim=0)
    test_set_x = torch.unsqueeze(torch.from_numpy(task.x).float(), dim=0)
    assert test_set_x.ndim == 3  # (n_tsk, n_tst, d_x)
    assert current_sample.ndim == 4  # (n_tsk, n_ls, n_marg, d_z)

    mu_y, std_y = np_decode(np_model, test_set_x, current_sample) # shape will be (n_tsk, n_ls, n_marg, n_tst, d_y)

    mu_y = torch.transpose(torch.squeeze(mu_y, dim=1), dim0=0, dim1=1).detach().numpy()
    std_y = torch.transpose(torch.squeeze(std_y, dim=1), dim0=0, dim1=1).detach().numpy()
    logger.warning("mu_y shape: " + str(mu_y.shape))
    #mu, var = np_model.predict(x=task.x) # predict returns variance!!

    assert mu_y.shape == (n_samples, 1, task.x.shape[0], task.y.shape[1]) # (n_samples, n_tasks, n_points, d_y)
    assert std_y.shape == (n_samples, 1, task.x.shape[0], task.y.shape[1])# (n_samples, n_tasks, n_points, d_y)

    #lmlhd = log_marginal_likelihood_mc(mu_y, std_y, y_true)

    y_true_torch = torch.from_numpy(y_true)
    mu_y_torch = torch.from_numpy(mu_y)
    std_y_torch = torch.from_numpy(std_y)

    log_likelihood_probs = torch.sum(utils.log_normal(y_true_torch, mu_y_torch, torch.log(torch.pow(std_y_torch, 2))), dim=[1,2]) # sum over d_y and then sum over data points in data set
    log_likelihood = torch.logsumexp(log_likelihood_probs, dim=0) - np.log(n_samples) # Average over samples

    return log_likelihood, log_likelihood_probs


def lmlhd_ais(np_model: NeuralProcess, task, n_samples = 10, chain_length=500):

    log_prior = construct_log_prior(np_model, n_samples)

    task_x_torch = torch.from_numpy(task.x).float()
    task_y_torch = torch.from_numpy(task.y).float()

    #reshaping of task.x to (n_tsk, n_tst, d_x)

    task_x_torch = torch.unsqueeze(task_x_torch, dim=0)
    task_y_torch = torch.unsqueeze(task_y_torch, dim=0)

    log_posterior = construct_log_posterior(np_model, log_prior, task_x_torch, task_y_torch)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    forward_schedule = torch.linspace(0, 1, chain_length, device=device)

    # initial state should have shape n_samples x d_z. Also for multiple tasks?
    # last_agg_state has dimension n_task x d_z
    mu_z, var_z = np_model.aggregator.last_agg_state

    # This can only handle a single task
    mu_z = torch.squeeze(mu_z, dim=0)
    mu_z = mu_z.repeat(n_samples, 1)
    var_z = torch.squeeze(var_z, dim=0)
    var_z = var_z.repeat(n_samples, 1)

    initial_state = torch.normal(mu_z, torch.sqrt(var_z))

    logger.warning("shape of initial state: " + str(initial_state.size()))

    return ais_trajectory(log_prior, log_posterior, initial_state, forward=True, schedule = forward_schedule, initial_step_size = 0.01, device = device)

def lmlhd_iwmc(np_model: NeuralProcess, task, n_samples = 10):
    mu_z_context, var_z_context = np_model.aggregator.last_agg_state

    # mu_z and var_z have shape n_task x d_z. We need mean to have shape n_samples x d_z
    #logger.warning("shape of mu_y: " + str(mu_z_context.size()) + ", shape of var_z_context: " + str(var_z_context.size()))

    mu_z_context = torch.squeeze(mu_z_context, dim=0)
    var_z_context = torch.squeeze(var_z_context, dim=0)

    mu_z_context = mu_z_context.repeat(n_samples, 1)
    std_z_context = torch.sqrt(var_z_context.repeat(n_samples, 1))
    #logger.warning("shape of mu_y: " + str(mu_z_context.size()) + ", shape of var_y: " + str(std_z_context.size()))

    # Generate samples from prior adapted on whole target set

    np_model.adapt(x = task.x, y = task.y)
    
    mu_z_target, var_z_target = np_model.aggregator.last_agg_state

    # mu_z and var_z have shape n_task x d_z. We need mean to have shape n_samples x d_z

    mu_z_target = torch.squeeze(mu_z_target, dim=0)
    var_z_target = torch.squeeze(var_z_target, dim=0)

    mu_z_target = mu_z_target.repeat(n_samples, 1)
    std_z_target = torch.sqrt(var_z_target.repeat(n_samples, 1))
    
    target_z_samples = Normal(mu_z_target, std_z_target).sample() 
    
    # Evaluate probabilities of z samples in both distributions
    log_context_probs = torch.sum(Normal(mu_z_context, std_z_context).log_prob(target_z_samples), dim=-1)
    log_target_probs = torch.sum(Normal(mu_z_target, std_z_target).log_prob(target_z_samples), dim=-1)

    # Compute likelihood on target set for every latent sample
    target_z_samples = torch.unsqueeze(torch.unsqueeze(target_z_samples, dim=0), dim=0)
    test_set_x = torch.unsqueeze(torch.from_numpy(task.x).float(), dim=0)
    test_set_y = torch.unsqueeze(torch.from_numpy(task.y).float(), dim=0)

    assert test_set_x.ndim == 3  # (n_tsk, n_tst, d_x)
    assert target_z_samples.ndim == 4  # (n_tsk, n_ls, n_marg, d_z)

    mu_y, std_y = np_decode(np_model, test_set_x, target_z_samples)

    assert mu_y.ndim == 5  # (n_tsk, 1, n_samples, n_tst, d_y)
    assert std_y.ndim == 5  # (n_tsk, 1, n_samples, n_tst, d_y)

    #logger.warning("shapes after decoder: mu_y: " +str(mu_y.shape) + ", std_y: " + str(std_y.shape))

    mu_y = torch.squeeze(torch.squeeze(mu_y, dim=0), dim=0)
    std_y = torch.squeeze(torch.squeeze(std_y, dim=0), dim=0)
    
    # TODO: Refactor so that utils.log_normal is not used (directly use Normal dist)
    log_likelihood_probs = torch.sum(utils.log_normal(test_set_y, mu_y, torch.log(torch.pow(std_y, 2))), dim=-1) # sum over d_y and then sum over data points in data set
    assert log_likelihood_probs.shape == (n_samples,)

    log_importance_weights = log_context_probs - log_target_probs

    #logger.warning(log_importance_weights)

    log_lhd = torch.logsumexp(log_likelihood_probs + log_importance_weights, dim=0) - np.log(n_samples)
    return log_lhd.detach(), (log_likelihood_probs + log_importance_weights).detach()

def np_decode(np_model, x, z):
    assert x.ndim == 3  # (n_tsk, n_tst, d_x)
    assert z.ndim == 4  # (n_tsk, n_ls, n_marg, d_z)

    x = np_model._normalize_x(x)
    mu_y, std_y = np_model.decoder.decode(x, z) # decoder returns std!!
    mu_y = np_model._denormalize_mu_y(mu_y)

    return mu_y, std_y

def estimates_over_time(np_model, task, max_samples):
    _, log_likelihood_probs_mc = lmlhd_mc(np_model, task, n_samples = max_samples)
    _, log_likelihood_probs_iwmc = lmlhd_iwmc(np_model, task, n_samples = max_samples)
    log_likelihoods_mc = []
    log_likelihoods_iwmc = []
    for i in range(1, int(np.log(max_samples))):
        log_likelihoods_mc.append((torch.logsumexp(log_likelihood_probs_mc[:int(np.exp(i))], dim=0) - i).item())
        log_likelihoods_iwmc.append((torch.logsumexp(log_likelihood_probs_iwmc[:int(np.exp(i))], dim=0) - i).item())

    #logger.warning("log likelihoods: " + str(log_likelihoods))
    x = np.arange(1, int(np.log(max_samples)))
    l1 = plt.plot(x, log_likelihoods_mc, 'ro-', x, log_likelihoods_iwmc, 'go-')
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

    # plot predictions
    for i in range(0, n_context_points + 1):
        for l in range(n_task_plot):
            task = benchmark.get_task_by_index(l)

            np_model.adapt(x=task.x[:i], y=task.y[:i])
            ax = axes[i, l]
            ax.clear()
            
            ax.scatter(task.x[i:], task.y[i:], s=15, color="g", alpha=1.0, zorder=3)
            ax.scatter(task.x[:i], task.y[:i], marker="x", s=30, color="r", alpha=1.0, zorder=3)
            
            for s in range(n_samples):
                mu, _ = np_model.predict(x=x_plt)
                ax.plot(x_plt, mu, color="b", alpha=0.3, label="posterior", zorder=2)

            lmlhd_mc_estimate, _ = lmlhd_mc(np_model, task, n_samples = 100)
            lmlhd_ais_estimate = lmlhd_ais(np_model, task, n_samples = 10, chain_length=20)

            lmlhd_iwae_estimate, _ = lmlhd_iwmc(np_model, task, n_samples = 100) # This method adapts on entire target set!
            
            print("number of context points: " + str(i) + ", task: " + str(l))
            print("MC estimate: " + str(round(lmlhd_mc_estimate.item(), 3)))
            print("AIS estimate: " + str(round(lmlhd_ais_estimate.item(), 3)))
            print("IWMC estimate: " + str(round(lmlhd_iwae_estimate.item(), 3)))

            ax.grid(zorder=1)

            if(i == 0):
                ax.set_title(f"Predictions (Task {l:d})")

            ax.text(0, 0, "MC: " + str(round(lmlhd_mc_estimate.item(), 3)) + ", AIS: " + str(round(lmlhd_ais_estimate.item(), 3)) + ", IWMC: " + str(round(lmlhd_iwae_estimate.item(), 3)), horizontalalignment='left', verticalalignment='bottom', transform=ax.transAxes)

    fig.tight_layout()
    plt.show(block=False)
    plt.pause(0.001)

def construct_log_prior(model, n_samples):
    mu_z, var_z = model.aggregator.last_agg_state

    # mu_z and var_z have shape n_task x d_z. We need mean to have shape n_samples x d_z
    #logger.warning("shape of mu_y: " + str(mu_z.size()) + ", shape of var_y: " + str(var_z.size()))

    mu_z = torch.squeeze(mu_z, dim=0)
    var_z = torch.squeeze(var_z, dim=0)
    mu_z = mu_z.repeat(n_samples, 1)
    std_z = torch.sqrt(var_z.repeat(n_samples, 1))
    #logger.warning("shape of mu_y: " + str(mu_z.size()) + ", shape of var_y: " + str(std_z.size()))

    return lambda z : torch.sum(Normal(mu_z, std_z).log_prob(z), dim=-1)

def construct_log_posterior(model, log_prior, test_set_x, test_set_y):
    return lambda z: log_prior(z) + log_likelihood_fn(model, test_set_x, test_set_y, torch.unsqueeze(torch.unsqueeze(z, dim=0), dim=0))

def log_likelihood_fn(model, test_set_x, test_set_y, z):
    assert test_set_x.ndim == 3  # (n_tsk, n_tst, d_x)
    assert z.ndim == 4  # (n_tsk, n_ls, n_marg, d_z)
    mu_y, std_y = np_decode(model, test_set_x, z)
    mu_y = torch.squeeze(torch.squeeze(mu_y, dim=0), dim=0)
    std_y = torch.squeeze(torch.squeeze(std_y, dim=0), dim=0)
    result = torch.sum(utils.log_normal(test_set_y, mu_y, torch.log(torch.pow(std_y, 2))), dim=1) # sum over d_y and then sum over data points in data set
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
    config["n_datapoints_per_task_val"] = 32
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
    config["n_context"] = [
        4,
        4,
    ]

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

    task = benchmark_test.get_task_by_index(0)
    model.adapt(x=task.x[:1], y=task.y[:1])

    estimates_over_time(model, task, 100000)
    '''
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
    plt.show()
    '''


if __name__ == "__main__":
    main()
