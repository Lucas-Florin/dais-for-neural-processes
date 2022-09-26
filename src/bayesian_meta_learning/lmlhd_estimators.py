import math
import numpy as np
import torch

from bayesian_meta_learning.ais import ais_trajectory
from eval_util.util import log_likelihood_mc_per_datapoint
from scipy.special import logsumexp
from torch.distributions.normal import Normal


def lmlhd_mc(decode, distribution, task, n_samples = 100, batch_size = None): # -> tuple[np.ndarray, np.ndarray]
    task_x, task_y = task # (n_tsk, n_tst, d_x), (n_tsk, n_tst, d_y)
    mu_z, var_z = distribution
    assert mu_z.ndim == 2 # (n_tsk, d_z)
    assert var_z.ndim == 2 # (n_tsk, d_z)

    n_tsk = mu_z.shape[0]
    d_z = mu_z.shape[1]
    n_points = task_x.shape[1]
    d_x = task_x.shape[2]
    d_y = task_y.shape[2]
    test_set_x = torch.from_numpy(task_x).float()

    if batch_size is None:
        batch_size = n_tsk

    lmlhd = np.zeros((n_tsk))
    lmlhd_samples = np.zeros((n_samples, n_tsk))

    for i in range(n_tsk // batch_size):
        current_sample = sample_normal(mu_z[(i*batch_size):((i+1)*batch_size), :], var_z[(i*batch_size):((i+1)*batch_size), :], n_samples)
        current_sample = torch.unsqueeze(torch.transpose(current_sample, dim0=0, dim1=1), dim=1)

        assert test_set_x.ndim == 3  # (n_tsk, n_tst, d_x)
        assert current_sample.ndim == 4  # (batch_size, n_ls, n_marg, d_z)
        assert current_sample.shape == (batch_size, 1, n_samples, d_z)

        mu_y, std_y = decode(test_set_x[(i*batch_size):((i+1)*batch_size), :, :], current_sample) # shape will be (batch_size, n_ls, n_marg, n_tst, d_y)
        mu_y = torch.transpose(torch.squeeze(mu_y, dim=1), dim0=0, dim1=1).detach().numpy()
        std_y = torch.transpose(torch.squeeze(std_y, dim=1), dim0=0, dim1=1).detach().numpy()

        assert mu_y.shape == (n_samples, batch_size, n_points, d_y) # (n_samples, n_tasks, n_points, d_y)
        assert std_y.shape == (n_samples, batch_size, n_points, d_y)# (n_samples, n_tasks, n_points, d_y)

        lmlhd_batch, lmlhd_samples_batch = get_dataset_likelihood(mu_y, std_y, task_y[(i*batch_size):((i+1)*batch_size), :, :])
        assert lmlhd_batch.shape == (batch_size,)
        assert lmlhd_samples_batch.shape == (n_samples, batch_size)

        lmlhd[(i*batch_size):((i+1)*batch_size)] = lmlhd_batch
        lmlhd_samples[:, (i*batch_size):((i+1)*batch_size)] = lmlhd_samples_batch

    return lmlhd, lmlhd_samples

def sample_normal(mu: torch.tensor, var: torch.tensor, num_samples: int): # -> tuple[torch.tensor, torch.tensor]
    """
    a sample function which takes the mu and variance of normal distributions and 
    samples from these distributions

    @param mu: a torch tensor contains the mu of normal distributions
    @param var: a torch tensor contains the variance of the normal distributions
    @return: a torch tensor, which contain the samples from these distributons
            it should have the shape(num_samples, mu.shape)
    """
    assert mu.shape == var.shape
    n_tsk = mu.shape[0]
    d_z = mu.shape[1]
    latent_distribution =  Normal(mu, torch.sqrt(var))
    samples = latent_distribution.sample((num_samples,)) # (number_samples, mu.shape)
    assert samples.shape == (num_samples, n_tsk, d_z)

    return samples

def get_dataset_likelihood(mu_y: torch.tensor, std_y: torch.tensor, y_true: np.ndarray):
    assert mu_y.shape == std_y.shape
    n_samples = mu_y.shape[0]
    n_tsk = mu_y.shape[1]
    n_points= mu_y.shape[2]

    lmlhd = log_likelihood_mc_per_datapoint(y_pred=mu_y, sigma_pred=std_y, y_true=y_true)
    assert lmlhd.shape == (n_samples, n_tsk, n_points)
    lmlhd_samples = np.sum(lmlhd, axis=-1)  # points per task
    assert lmlhd_samples.shape == (n_samples, n_tsk)
    lmlhd = logsumexp(lmlhd_samples, axis=0) - math.log(n_samples)  # samples

    return lmlhd, lmlhd_samples

def lmlhd_ais(decode, context_distribution, task, n_samples = 10, chain_length=500, 
    n_leapfrog_steps=10, initial_stepsize=0.01):

    task_x, task_y = task # (n_tsk, n_tst, d_x), (n_tsk, n_tst, d_y)
    assert task_x.ndim == 3
    assert task_y.ndim == 3
    assert isinstance(task_x, np.ndarray)
    assert isinstance(task_y, np.ndarray)

    task_x_torch = torch.from_numpy(task_x).float()
    task_y_torch = torch.from_numpy(task_y).float()

    log_prior = construct_log_prior(context_distribution, n_samples)
    log_posterior = construct_log_posterior(decode, log_prior, task_x_torch, task_y_torch)
    
    forward_schedule = torch.linspace(0, 1, chain_length, device=device)

    # initial state should have shape n_samples x n_task x d_z, last_agg_state has dimension n_task x d_z
    mu_z, var_z = context_distribution
    mu_z = mu_z.repeat(n_samples, 1)
    var_z = var_z.repeat(n_samples, 1)

    n_tsk = task_x.shape[0]
    d_z = mu_z.shape[-1]

    initial_state = torch.normal(mu_z, torch.sqrt(var_z))
    assert initial_state.shape == (n_samples * n_tsk, d_z)

    return ais_trajectory(log_prior, log_posterior, initial_state, n_samples=n_samples, forward=True, schedule = forward_schedule, initial_step_size = initial_stepsize, num_leapfrog_steps=n_leapfrog_steps, device = device)

def construct_log_prior(context_distribution, n_samples):
    mu_z, var_z = context_distribution
    assert mu_z.ndim == 2 # (n_tsk, d_z)
    assert var_z.ndim == 2 # (n_tsk, d_z)
    # mu_z and var_z have shape n_task x d_z. We need mean and var to have shape (n_samples * n_task) x d_z
    mu_z = mu_z.repeat(n_samples, 1)
    std_z = torch.sqrt(var_z.repeat(n_samples, 1))
    return lambda z : torch.sum(Normal(mu_z, std_z).log_prob(z), dim=-1)

def construct_log_posterior(decode, log_prior, test_set_x, test_set_y):
    return lambda z: log_prior(z) + log_likelihood_fn(decode, test_set_x, test_set_y, z)

def log_likelihood_fn(decode, test_set_x, test_set_y, z):
    assert test_set_x.ndim == 3  # (n_tsk, n_tst, d_x)
    assert test_set_y.ndim == 3  # (n_tsk, n_tst, d_y)
    assert z.ndim == 2  # (n_samples * n_tsk, d_z)
    n_tsk = test_set_x.shape[0]
    n_samples = z.shape[0] // n_tsk
    d_z = z.shape[1]

    z_decoder = torch.reshape(z, (n_samples, n_tsk, d_z))
    z_decoder = torch.unsqueeze(z_decoder.transpose(dim0=0, dim1=1), dim=1) # (n_tsk, n_ls, n_samples, d_z)
    assert z_decoder.shape == (n_tsk, 1, n_samples, d_z)
    mu_y, std_y = decode(test_set_x, z_decoder) # shape will be (n_tsk, n_ls, n_samples, n_tst, d_y)
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

def lmlhd_iwmc(decode, context_distribution, target_distribution, task, n_samples = 1000, batch_size = None):
    mu_context, var_context = context_distribution
    mu_target, var_target = target_distribution

    assert mu_context.shape == var_context.shape # (n_tsk, d_z)
    assert mu_target.shape == var_target.shape # (n_tsk, d_z)
    assert mu_context.shape == mu_target.shape # (n_tsk, d_z)

    task_x, task_y = task # (n_tsk, n_tst, d_x), (n_tsk, n_tst, d_y)
    n_tsk = mu_context.shape[0]
    d_z = mu_context.shape[1]
    n_points = task_x.shape[1]
    d_x = task_x.shape[2]
    d_y = task_y.shape[2]
    test_set_x = torch.from_numpy(task_x).float()
    test_set_y = torch.from_numpy(task_y).float()

    if batch_size is None:
        batch_size = n_tsk

    lmlhd = np.zeros((n_tsk))
    lmlhd_samples = np.zeros((n_samples, n_tsk))
    log_importance_weights = np.zeros((n_samples, n_tsk))

    for i in range(n_tsk // batch_size):
        target_z_samples = sample_normal(mu_target[(i*batch_size):((i+1)*batch_size), :], var_target[(i*batch_size):((i+1)*batch_size), :], n_samples)
        assert target_z_samples.shape == (n_samples, batch_size, d_z)

        # Evaluate probabilities of z samples in both distributions
        log_context_probs = torch.sum(Normal(mu_context[(i*batch_size):((i+1)*batch_size), :].repeat(n_samples, 1, 1), torch.sqrt(var_context[(i*batch_size):((i+1)*batch_size), :].repeat(n_samples, 1, 1))).log_prob(target_z_samples), dim=-1)
        log_target_probs = torch.sum(Normal(mu_target[(i*batch_size):((i+1)*batch_size), :].repeat(n_samples, 1, 1), torch.sqrt(var_target[(i*batch_size):((i+1)*batch_size), :].repeat(n_samples, 1, 1))).log_prob(target_z_samples), dim=-1)
        assert log_context_probs.shape == (n_samples, batch_size)
        assert log_target_probs.shape == (n_samples, batch_size)

        log_importance_weights_batch = (log_context_probs - log_target_probs).detach().numpy()
        assert log_importance_weights_batch.shape == (n_samples, batch_size)
        log_importance_weights[:, (i*batch_size):((i+1)*batch_size)] = log_importance_weights_batch

        target_z_samples = torch.unsqueeze(torch.transpose(target_z_samples, dim0=0, dim1=1), dim=1)
        mu_y, std_y = decode(test_set_x[(i*batch_size):((i+1)*batch_size), :, :], target_z_samples)
        assert mu_y.shape == (batch_size, 1, n_samples, n_points, d_y)
        assert std_y.shape == (batch_size, 1, n_samples, n_points, d_y)

        mu_y = torch.transpose(torch.squeeze(mu_y, dim=1), dim0=0, dim1=1).detach().numpy()
        std_y = torch.transpose(torch.squeeze(std_y, dim=1), dim0=0, dim1=1).detach().numpy()
        assert mu_y.shape == (n_samples, batch_size, n_points, d_y)
        assert std_y.shape == (n_samples, batch_size, n_points, d_y)
        
        _, lmlhd_samples_batch = get_dataset_likelihood(mu_y, std_y, task_y[(i*batch_size):((i+1)*batch_size), :, :])
        assert lmlhd_samples_batch.shape == (n_samples, batch_size)
        lmlhd_samples[:, (i*batch_size):((i+1)*batch_size)] = lmlhd_samples_batch

        lmlhd_batch = logsumexp(np.add(lmlhd_samples_batch, log_importance_weights_batch), axis=0) - np.log(n_samples)
        assert lmlhd_batch.shape == (batch_size,)
        lmlhd[(i*batch_size):((i+1)*batch_size)] = lmlhd_batch

    return lmlhd, np.add(lmlhd_samples, log_importance_weights), lmlhd_samples, log_importance_weights


def lmlhd_elbo(decode, context_distribution, target_distribution, task, n_samples = 1000, batch_size = None):
    mu_context, var_context = context_distribution
    mu_target, var_target = target_distribution
    assert mu_context.shape == var_context.shape
    assert mu_target.shape == var_target.shape
    assert mu_context.shape == mu_target.shape
    task_x, task_y = task
    n_tsk = mu_context.shape[0]
    d_z = mu_context.shape[1]
    n_points = task_x.shape[1]
    d_x = task_x.shape[2]
    d_y = task_y.shape[2]
    test_set_x = torch.from_numpy(task_x).float()
    test_set_y = torch.from_numpy(task_y).float()

    if batch_size is None:
        batch_size = n_tsk

    lmlhd = np.zeros((n_tsk))
    lmlhd_samples = np.zeros((n_samples, n_tsk))
    log_importance_weights = np.zeros((n_samples, n_tsk))

    for i in range(n_tsk // batch_size):
        target_z_samples = sample_normal(mu_target[(i*batch_size):((i+1)*batch_size), :], var_target[(i*batch_size):((i+1)*batch_size), :], n_samples)
        assert target_z_samples.shape == (n_samples, batch_size, d_z)
        # Evaluate probabilities of z samples in both distributions
        log_context_probs = torch.sum(Normal(mu_context[(i*batch_size):((i+1)*batch_size), :].repeat(n_samples, 1, 1), torch.sqrt(var_context[(i*batch_size):((i+1)*batch_size), :].repeat(n_samples, 1, 1))).log_prob(target_z_samples), dim=-1)
        log_target_probs = torch.sum(Normal(mu_target[(i*batch_size):((i+1)*batch_size), :].repeat(n_samples, 1, 1), torch.sqrt(var_target[(i*batch_size):((i+1)*batch_size), :].repeat(n_samples, 1, 1))).log_prob(target_z_samples), dim=-1)
        assert log_context_probs.shape == (n_samples, batch_size)
        assert log_target_probs.shape == (n_samples, batch_size)
        log_importance_weights_batch = (log_context_probs - log_target_probs).detach().numpy()
        assert log_importance_weights_batch.shape == (n_samples, batch_size)
        log_importance_weights[:, (i*batch_size):((i+1)*batch_size)] = log_importance_weights_batch

        target_z_samples = torch.unsqueeze(torch.transpose(target_z_samples, dim0=0, dim1=1), dim=1)
        mu_y, std_y = decode(test_set_x[(i*batch_size):((i+1)*batch_size), :, :], target_z_samples)
        assert mu_y.shape == (batch_size, 1, n_samples, n_points, d_y)
        assert std_y.shape == (batch_size, 1, n_samples, n_points, d_y)
        mu_y = torch.transpose(torch.squeeze(mu_y, dim=1), dim0=0, dim1=1).detach().numpy()
        std_y = torch.transpose(torch.squeeze(std_y, dim=1), dim0=0, dim1=1).detach().numpy()
        assert mu_y.shape == (n_samples, batch_size, n_points, d_y)
        assert std_y.shape == (n_samples, batch_size, n_points, d_y)
        _, lmlhd_samples_batch = get_dataset_likelihood(mu_y, std_y, task_y[(i*batch_size):((i+1)*batch_size), :, :])
        assert lmlhd_samples_batch.shape == (n_samples, batch_size)
        lmlhd_samples[:, (i*batch_size):((i+1)*batch_size)] = lmlhd_samples_batch

        lmlhd_batch = np.mean(np.add(lmlhd_samples_batch, log_importance_weights_batch), axis=0)
        assert lmlhd_batch.shape == (batch_size,)
        lmlhd[(i*batch_size):((i+1)*batch_size)] = lmlhd_batch

    return lmlhd, np.add(lmlhd_samples, log_importance_weights), lmlhd_samples, log_importance_weights