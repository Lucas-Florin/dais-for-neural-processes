import math
import numpy as np
import torch

from bayesian_meta_learning.ais import ais_trajectory
from bayesian_meta_learning.dais_new import dais_new_trajectory
from eval_util.util import log_likelihood_mc_per_datapoint
from scipy.special import logsumexp
from torch.distributions.normal import Normal
from neural_process.dais import differentiable_annealed_importance_sampling


def get_task_batch_iterator(*args, batch_size=None):
    assert batch_size is not None
    assert all([type(a) is np.ndarray or type(a) is torch.Tensor for a in args])
    main_dimesion = args[0].shape[0]
    assert all([a.shape[0] == main_dimesion for a in args])
    processed_datapoints = 0
    while processed_datapoints < main_dimesion:
        this_batch_size = min(batch_size, main_dimesion - processed_datapoints)
        batch_list = [
            a[processed_datapoints:processed_datapoints+this_batch_size, ...]
            for a in args
        ]
        yield tuple(batch_list)
        processed_datapoints += this_batch_size
    assert processed_datapoints == main_dimesion    


def get_torch_rng(seed):
    if seed is None:
        rng = None
    else:
        rng = torch.Generator()
        rng.manual_seed(seed)
    return rng    


def lmlhd_mc(decode, distribution, task, n_samples = 100, batch_size = None, subbatch_size=None, seed=None): # -> tuple[np.ndarray, np.ndarray]
    task_x, task_y = task # (n_tsk, n_tst, d_x), (n_tsk, n_tst, d_y)
    mu_z, var_z = distribution
    assert mu_z.ndim == 2 # (n_tsk, d_z)
    assert var_z.ndim == 2 # (n_tsk, d_z)

    n_tsk = mu_z.shape[0]
    batch_size = n_tsk if batch_size is None else batch_size
    d_z = mu_z.shape[1]
    n_points = task_x.shape[1]
    d_x = task_x.shape[2]
    d_y = task_y.shape[2]

    lmlhd_list = list()    
    lmlhd_samples_list = list()    
    rng = get_torch_rng(seed)

    for mu_z_batch, var_z_batch, task_x_batch, task_y_batch in get_task_batch_iterator(mu_z, var_z, task_x, task_y, batch_size=batch_size):
        this_batch_size = mu_z_batch.shape[0]
        subbatch_size = n_samples if subbatch_size is None else subbatch_size
        assert n_samples % subbatch_size == 0
        lmlhd_batch_sum = np.zeros((batch_size, n_samples // subbatch_size))
        for i in range(n_samples // subbatch_size):
            current_sample = sample_normal(mu_z_batch, var_z_batch, subbatch_size, rng=rng)
            current_sample = torch.unsqueeze(torch.transpose(current_sample, dim0=0, dim1=1), dim=1)

            assert task_x_batch.ndim == 3  # (n_tsk, n_tst, d_x)
            assert current_sample.ndim == 4  # (n_tsk, n_ls, n_marg, d_z)
            assert current_sample.shape == (this_batch_size, 1, subbatch_size, d_z)

            mu_y, std_y = decode(torch.from_numpy(task_x_batch).float(), current_sample) # shape will be (n_tsk, n_ls, n_marg, n_tst, d_y)
            mu_y = torch.transpose(torch.squeeze(mu_y, dim=1), dim0=0, dim1=1).detach().numpy()
            std_y = torch.transpose(torch.squeeze(std_y, dim=1), dim0=0, dim1=1).detach().numpy()

            assert mu_y.shape == (subbatch_size, this_batch_size, n_points, d_y) # (n_samples, n_tasks, n_points, d_y)
            assert std_y.shape == (subbatch_size, this_batch_size, n_points, d_y)# (n_samples, n_tasks, n_points, d_y)

            lmlhd_batch, lmlhd_samples_batch = get_dataset_likelihood(mu_y, std_y, task_y_batch)
            lmlhd_batch_sum[:, i] = lmlhd_batch
        lmlhd_batch_sum = logsumexp(lmlhd_batch_sum, axis=1) - np.log(n_samples / subbatch_size)
        lmlhd_list.append(lmlhd_batch_sum)
        lmlhd_samples_list.append(lmlhd_samples_batch)
        
    lmlhd = np.concatenate(lmlhd_list)
    lmlhd_samples = np.concatenate(lmlhd_samples_list)
        
    assert lmlhd.shape == (n_tsk,) # check output

    return lmlhd, lmlhd_samples

def sample_normal(mu: torch.tensor, var: torch.tensor, num_samples: int, rng: torch.Generator = None): # -> tuple[torch.tensor, torch.tensor]
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
    shape = (num_samples, n_tsk, d_z)
    if rng is None:
        latent_distribution =  Normal(mu, torch.sqrt(var))
        samples = latent_distribution.sample((num_samples,)) # (number_samples, mu.shape)
    else: 
        samples = torch.normal(mu.expand(shape), var.sqrt().expand(shape), generator=rng)
    assert samples.shape == shape

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


def lmlhd_ais(decode, context_distribution, task, n_samples = 10, chain_length=500, device=None, num_leapfrog_steps=10, 
              step_size=0.01, adapt_step_size_to_std_z=False, scalar_step_size=False, random_time_direction=False, seed=None):
    task_x, task_y = task # (n_tsk, n_tst, d_x), (n_tsk, n_tst, d_y)
    assert task_x.ndim == 3
    assert task_y.ndim == 3
    assert isinstance(task_x, np.ndarray)
    assert isinstance(task_y, np.ndarray)

    task_x_torch = torch.from_numpy(task_x).float()
    task_y_torch = torch.from_numpy(task_y).float()

    
    forward_schedule = torch.linspace(0, 1, chain_length + 1, device=device)
    rng = get_torch_rng(seed)

    # initial state should have shape n_samples x n_task x d_z, last_agg_state has dimension n_task x d_z
    mu_z, var_z = context_distribution
    mu_z = mu_z.repeat(n_samples, 1, 1)
    var_z = var_z.repeat(n_samples, 1, 1)

    if adapt_step_size_to_std_z:
        if scalar_step_size:
            step_size = step_size * var_z.sqrt().mean()
        else:
            step_size = step_size * var_z.sqrt().mean(-1)
        step_size = step_size.detach()

    n_tsk = task_x.shape[0]
    d_z = mu_z.shape[-1]

    log_prior = construct_log_prior(mu_z, var_z)
    log_posterior = construct_log_posterior(decode, log_prior, task_x_torch, task_y_torch)

    initial_state = torch.normal(mu_z, torch.sqrt(var_z), generator=rng)
    assert initial_state.shape == (n_samples, n_tsk, d_z)

    return ais_trajectory(log_prior, log_posterior, initial_state, n_samples=n_samples, forward=True, 
                          schedule = forward_schedule, initial_step_size = step_size, device = device, 
                          num_leapfrog_steps=num_leapfrog_steps, random_time_direction=random_time_direction, 
                          rng=rng)


def lmlhd_dais_new(decode, context_distribution, task, n_samples = 10, chain_length=500, device=None, 
                   num_leapfrog_steps=10, step_size=0.01, step_size_update_factor = 0.98, target_accept_rate = 0.65, 
                   clip_grad=100.0, adapt_step_size=True, do_accept_reject_step=False, use_accept_hist=True,
                   adapt_step_size_to_std_z=False, scalar_step_size=False, seed=None):
    task_x, task_y = task # (n_tsk, n_tst, d_x), (n_tsk, n_tst, d_y)
    assert task_x.ndim == 3
    assert task_y.ndim == 3
    assert isinstance(task_x, np.ndarray)
    assert isinstance(task_y, np.ndarray)

    task_x_torch = torch.from_numpy(task_x).float()
    task_y_torch = torch.from_numpy(task_y).float()

    
    forward_schedule = torch.linspace(0, 1, chain_length + 1, device=device)
    rng = get_torch_rng(seed)

    # initial state should have shape n_samples x n_task x d_z, last_agg_state has dimension n_task x d_z
    mu_z, var_z = context_distribution
    mu_z = mu_z.repeat(n_samples, 1, 1)
    var_z = var_z.repeat(n_samples, 1, 1)
    
    if adapt_step_size_to_std_z:
        if scalar_step_size:
            step_size = step_size * var_z.sqrt().mean()
        else:
            step_size = step_size * var_z.sqrt().mean(-1)
        step_size = step_size.detach()

    n_tsk = task_x.shape[0]
    d_z = mu_z.shape[-1]

    log_prior = construct_log_prior(mu_z, var_z)
    log_posterior = construct_log_posterior(decode, log_prior, task_x_torch, task_y_torch)

    initial_state = torch.normal(mu_z, torch.sqrt(var_z), generator=rng)
    assert initial_state.shape == (n_samples, n_tsk, d_z)

    return dais_new_trajectory(log_prior, log_posterior, initial_state, n_samples=n_samples, forward=True, 
                          schedule = forward_schedule, initial_step_size = step_size, 
                          step_size_update_factor=step_size_update_factor, target_accept_rate=target_accept_rate, 
                          device = device, rng=rng,
                          num_leapfrog_steps=num_leapfrog_steps, clip_grad=clip_grad, adapt_step_size=adapt_step_size, 
                          do_accept_reject_step=do_accept_reject_step, use_accept_hist=use_accept_hist,
                          )

def construct_log_prior(mu_z, var_z):
    assert mu_z.shape == var_z.shape
    std_z = torch.sqrt(var_z)
    return lambda z : torch.sum(Normal(mu_z, std_z).log_prob(z), dim=-1)

def construct_log_posterior(decode, log_prior, test_set_x, test_set_y):
    return lambda z: log_prior(z) + log_likelihood_fn(decode, test_set_x, test_set_y, z)

def log_likelihood_fn(decode, test_set_x, test_set_y, z):
    assert test_set_x.ndim == 3  # (n_tsk, n_tst, d_x)
    assert test_set_y.ndim == 3  # (n_tsk, n_tst, d_y)
    assert z.ndim == 3  # (n_samples, n_tsk, d_z)
    n_tsk = test_set_x.shape[0]
    n_samples = z.shape[0]
    d_z = z.shape[-1]

    z_decoder = z
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
    assert result.shape == (n_samples, n_tsk)
    return result

def lmlhd_iwmc(decode, context_distribution, target_distribution, task, n_samples = 1000):
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

    target_z_samples = sample_normal(mu_target, var_target, n_samples)
    assert target_z_samples.shape == (n_samples, n_tsk, d_z)

    # Evaluate probabilities of z samples in both distributions
    log_context_probs = torch.sum(Normal(mu_context.repeat(n_samples, 1, 1), torch.sqrt(var_context.repeat(n_samples, 1, 1))).log_prob(target_z_samples), dim=-1)
    log_target_probs = torch.sum(Normal(mu_target.repeat(n_samples, 1, 1), torch.sqrt(var_target.repeat(n_samples, 1, 1))).log_prob(target_z_samples), dim=-1)
    assert log_context_probs.shape == (n_samples, n_tsk)
    assert log_target_probs.shape == (n_samples, n_tsk)

    log_importance_weights = (log_context_probs - log_target_probs).detach().numpy()
    assert log_importance_weights.shape == (n_samples, n_tsk)

    target_z_samples = torch.unsqueeze(torch.transpose(target_z_samples, dim0=0, dim1=1), dim=1)
    mu_y, std_y = decode(test_set_x, target_z_samples)
    assert mu_y.shape == (n_tsk, 1, n_samples, n_points, d_y)
    assert std_y.shape == (n_tsk, 1, n_samples, n_points, d_y)

    mu_y = torch.transpose(torch.squeeze(mu_y, dim=1), dim0=0, dim1=1).detach().numpy()
    std_y = torch.transpose(torch.squeeze(std_y, dim=1), dim0=0, dim1=1).detach().numpy()
    assert mu_y.shape == (n_samples, n_tsk, n_points, d_y)
    assert std_y.shape == (n_samples, n_tsk, n_points, d_y)
    
    _, lmlhd_samples = get_dataset_likelihood(mu_y, std_y, task_y)
    assert lmlhd_samples.shape == (n_samples, n_tsk)

    log_lhd = logsumexp(np.add(lmlhd_samples, log_importance_weights), axis=0) - np.log(n_samples)
    assert log_lhd.shape == (n_tsk,)
    return log_lhd, np.add(lmlhd_samples, log_importance_weights), lmlhd_samples, log_importance_weights


def lmlhd_dais(decode, context_distribution, task, n_samples = 10, chain_length=1000, device=None, 
               num_leapfrog_steps=10, step_size=0.01, partial=False, batch_size=None, clip_grad=1.0, 
               adapt_step_size_to_std_z=False, scalar_step_size=False, schedule='linear', seed=None):
    task_x, task_y = task # (n_tsk, n_tst, d_x), (n_tsk, n_tst, d_y)
    assert task_x.ndim == 3
    assert task_y.ndim == 3
    assert isinstance(task_x, np.ndarray)
    assert isinstance(task_y, np.ndarray)

    task_x = torch.from_numpy(task_x).float()
    task_y = torch.from_numpy(task_y).float()

    mu_z, var_z = context_distribution
    n_tsk = task_x.shape[0]
    d_z = mu_z.shape[-1]
    rng = None if seed is None else np.random.RandomState(seed)
    betas = np.linspace(0, 1, chain_length + 1) if schedule == 'linear' else None
    batch_size = n_tsk if batch_size is None else batch_size
    lmlhd_list = list()    
    for mu_z_batch, var_z_batch, task_x_batch, task_y_batch in get_task_batch_iterator(mu_z, var_z, task_x, task_y, batch_size=batch_size):

        # initial state should have shape n_samples x n_task x d_z, last_agg_state has dimension n_task x d_z
        mu_z_batch = mu_z_batch.repeat(n_samples, 1, 1)
        var_z_batch = var_z_batch.repeat(n_samples, 1, 1)
        step_size_batch = step_size
        if adapt_step_size_to_std_z:
            if scalar_step_size:
                step_size_batch = step_size_batch * var_z_batch.sqrt().mean()
            else:
                step_size_batch = step_size_batch * var_z_batch.sqrt().mean(-1)
            step_size_batch = step_size_batch.detach()
        log_prior = construct_log_prior(mu_z_batch, var_z_batch)
        log_posterior = construct_log_posterior(decode, log_prior, task_x_batch, task_y_batch)
        
        initial_state = torch.from_numpy(rng.normal(mu_z_batch.numpy(), torch.sqrt(var_z_batch).numpy())).float()
        # assert initial_state.shape == (n_samples, batch_size, d_z)

        ll, _ = differentiable_annealed_importance_sampling(
            initial_state, 
            log_posterior, 
            log_prior, 
            chain_length, 
            step_size = step_size_batch,
            partial=partial,
            clip_grad=clip_grad,
            betas=betas,
            is_train=False,
            rng=rng,
        )
        ll = ll.detach()
        ll = torch.logsumexp(ll, dim=0) - torch.log(torch.tensor(n_samples))
        lmlhd_list.append(ll.detach())
    
    lmlhd = torch.cat(lmlhd_list)
    assert lmlhd.shape == (n_tsk,) # check output

    return lmlhd