# from multiprocessing.context import assert_spawning
from typing import List
from typing import Optional
from typing import Union

import torch

from bayesian_meta_learning import hmc

# import logging

@torch.no_grad()
def dais_new_trajectory(
    proposal_log_prob_fn,
    target_log_prob_fn,
    initial_state,
    n_samples: int,
    forward: bool,
    schedule: Union[torch.Tensor, List],
    initial_step_size: Optional[int] = 0.01,
    step_size_update_factor = 0.98,
    target_accept_rate = 0.65,
    adapt_step_size=False,
    use_accept_hist=True,
    device: Optional[torch.device] = None,
    num_leapfrog_steps: Optional[int] = 10,
    clip_grad=100.0,
    do_accept_reject_step=False,
    rng: torch.Generator = None,
):
    """Compute annealed importance sampling trajectories for a batch of data.

    Could be used for *both* forward and reverse chain in BDMC.

    Args:
      proposal_log_prob_fn: Log-probability function initial distribution of AIS
      target_log_prob_fn: Log-probability function of target distribution of AIS
      initial_state: Initial sample from the initial distribution, shape is (n_samples * n_tsk) x d_z
      n_samples: Number of samples per task
      forward: indicate forward/backward chain
      schedule: temperature schedule, i.e. `p(z)p(x|z)^t`
      device: device to run all computation on
      initial_step_size: initial step size for leap-frog integration;
        the actual step size is adapted online based on accept-reject ratios

    Returns:
        a list where each element is a torch.Tensor that contains the
        log importance weights for a single batch of data
    """

    def log_f_i(z, t):
        """Unnormalized density for intermediate distribution `f_i`:
            f_i = p(z)^(1-t) p(x,z)^(t) = p(z) p(x|z)^t
        =>  log f_i = log p(z) + t * log p(x|z)
        """
        proposal = proposal_log_prob_fn(z) * (1 - t)
        target = target_log_prob_fn(z) * (t)
        return proposal + target

    B = initial_state.shape[:-1]
    if type(initial_step_size) is torch.Tensor and len(initial_step_size.shape) > 0:
        epsilon = initial_step_size
        assert epsilon.shape == B
    else:
        epsilon = torch.full(size=B, device=device, fill_value=initial_step_size)
    accept_hist = torch.zeros(size=B, device=device)
    logw = torch.zeros(size=B, device=device)

    # initial sample of z
    if forward:
        current_z = initial_state.to(device)
    
    else: # not implemented for now
        #current_z = utils.safe_repeat(post_z, n_samples).to(device)
        raise NotImplementedError

    for j, (t0, t1) in (enumerate(zip(schedule[:-1], schedule[1:]), 1)):
        # update log importance weight
        log_int_1 = log_f_i(current_z, t0)
        log_int_2 = log_f_i(current_z, t1)
        logw += log_int_2 - log_int_1

        def U(z):
            return -log_f_i(z, t1)

        @torch.enable_grad()
        def grad_U(z):
            z = z.clone().requires_grad_(True)
            grad, = torch.autograd.grad(U(z).sum(), z)
            if clip_grad is not None:
                max_ = torch.prod(torch.tensor(initial_state.shape)) * clip_grad # last dimension of mu_z
                grad = torch.clamp(grad, -max_, max_)
            return grad

        def normalized_kinetic(v):
            zeros = torch.zeros_like(v)
            ones = torch.ones_like(v)
            return - torch.distributions.Normal(zeros, ones).log_prob(v).sum(dim=-1)

        # resample velocity
        current_v = torch.normal(0.0, 1.0, size=current_z.shape, generator=rng)
        z, v = hmc.hmc_trajectory(current_z, current_v, grad_U, epsilon, L=num_leapfrog_steps)

        current_z, epsilon, accept_hist = hmc.accept_reject(
            current_z,
            current_v,
            z,
            v,
            epsilon,
            accept_hist,
            j,
            U=U,
            K=normalized_kinetic,
            acceptance_threshold=target_accept_rate,
            epsilon_update_factor=step_size_update_factor,
            adapt_step_size=adapt_step_size,
            do_accept_reject_step=do_accept_reject_step,
            use_accept_hist=use_accept_hist,
            rng=rng,
        )
        
    logw = torch.logsumexp(logw, dim=0) - torch.log(torch.tensor(n_samples))

    if not forward:
        logw = -logw
    
    return logw
