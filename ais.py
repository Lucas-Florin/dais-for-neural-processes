from multiprocessing.context import assert_spawning
from typing import List
from typing import Optional
from typing import Union

import torch
from tqdm import tqdm

import hmc
import utils

import logging

@torch.no_grad()
def ais_trajectory(
    proposal_log_prob_fn,
    target_log_prob_fn,
    initial_state,
    forward: bool,
    schedule: Union[torch.Tensor, List],
    initial_step_size: Optional[int] = 0.01,
    device: Optional[torch.device] = None,
):
    """Compute annealed importance sampling trajectories for a batch of data.

    Could be used for *both* forward and reverse chain in BDMC.

    Args:
      proposal_log_prob_fn: Log-probability function initial distribution of AIS
      target_log_prob_fn: Log-probability function of target distribution of AIS
      initial_state: Initial sample from the initial distribution, shape is n_samples x d_z
      forward: indicate forward/backward chain
      schedule: temperature schedule, i.e. `p(z)p(x|z)^t`
      device: device to run all computation on
      initial_step_size: initial step size for leap-frog integration;
        the actual step size is adapted online based on accept-reject ratios

    Returns:
        a list where each element is a torch.Tensor that contains the
        log importance weights for a single batch of data
    """

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    n_samples = initial_state.size()[0]

    def log_f_i(z, t):
        """Unnormalized density for intermediate distribution `f_i`:
            f_i = p(z)^(1-t) p(x,z)^(t) = p(z) p(x|z)^t
        =>  log f_i = log p(z) + t * log p(x|z)
        """
        #mu_z, var_z =  model.aggregator.last_agg_state

        #log_prior = utils.log_normal(z, mu_z, torch.log(var_z))



        
        # z modified to have shape 1 x 1 x B x dim_z
        #reshaped_z = torch.unsqueeze(torch.unsqueeze(z, dim=0), dim=0)
        
        #logger.warning("shape of reshaped z: " + str(reshaped_z.size()))
        #assert reshaped_z.type() == "torch.FloatTensor"
        '''
        reshaped_batch = torch.unsqueeze(batch, dim=0).float()
        #print("shape of reshaped batch: " + str(reshaped_batch.type()))
        #logger.warning("shape of reshaped batch: " + str(reshaped_batch.size()))
        assert reshaped_batch.type() == "torch.FloatTensor"
        mu, std = model.decoder.decode(reshaped_batch, reshaped_z)
        log_var = torch.log(std) * 0.5

        

        #print("shape of batch_labels: ", batch_labels.size(), "shape of mu: ", mu.size(), "shape of log_var: ", log_var.size())
        #logger.warning("shape of batch_labels: " + str(batch_labels.size()) + "shape of mu: " + str(mu.size()) + "shape of log_var: " + str(log_var.size()))
        
        # This will be modified, once we use more than one task
        batch_labels = batch_labels.squeeze(dim=0)
        batch_labels = batch_labels.squeeze(dim=0)

        mu = mu.squeeze(dim=0)
        mu = mu.squeeze(dim=0)

        log_var = log_var.squeeze(dim=0)
        log_var = log_var.squeeze(dim=0)
    	
        if(t == 1):
            print("mu: " + str(mu))
            print("log_var: " + str(log_var))
        
        #logger.warning("shape of squeezed batch_labels: " + str(batch_labels.size()) + "shape of squeezed mu: " + str(mu.size()) + "shape of squeezed log_var: " + str(log_var.size()))

        log_likelihood = log_likelihood_fn(batch_labels, mu, log_var).squeeze()   
        #logger.warning("shape of log_likelihood: " + str(log_likelihood.size()))
        '''
        proposal = proposal_log_prob_fn(z).mul_(1 - t)
        target = target_log_prob_fn(z).mul_(t)
        #logger.warning("shape of proposal: " + str(proposal.size()))
        #logger.warning("shape of target: " + str(target.size()))
        return proposal + target


    B = 1 * n_samples
    #batch = batch.to(device)

    #print("batch labels: " + str(batch_labels))

    #batch_labels = batch_labels[None, None, None, :, :]
    #batch_labels = batch_labels.expand(1, 1, n_samples, batch.size(0), batch.size(1))
    #batch = utils.safe_repeat(batch, n_samples)
    #batch_labels = utils.safe_repeat(batch_labels, n_samples)

    epsilon = torch.full(size=(B,), device=device, fill_value=initial_step_size)
    accept_hist = torch.zeros(size=(B,), device=device)
    logw = torch.zeros(size=(B,), device=device)

    # initial sample of z
    if forward:
        # This probably needs to change for NPs, because we want to start with samples of the prior conditioned on the context set
        #current_z = torch.randn(size=(B, model.settings["d_z"]), device=device, dtype=torch.float32)
        #mu_z, var_z =  initial_state
        # logger.warning("shape of mu_z: " + str(mu_z.size()) + ", shape of var_z: " + str(var_z.size()))

        #current_z = torch.normal(mu_z, torch.sqrt(var_z))
        current_z = initial_state
    
    else: # not implemented for now
        current_z = utils.safe_repeat(post_z, n_samples).to(device)

    for j, (t0, t1) in tqdm(enumerate(zip(schedule[:-1], schedule[1:]), 1)):
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
            max_ = B * initial_state[0].size()[-1] * 100. # last dimension of mu_z
            grad = torch.clamp(grad, -max_, max_)
            return grad

        def normalized_kinetic(v):
            zeros = torch.zeros_like(v)
            return -utils.log_normal(v, zeros, zeros)

        # resample velocity
        current_v = torch.randn_like(current_z)
        z, v = hmc.hmc_trajectory(current_z, current_v, grad_U, epsilon)
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
        )
    
    print(logw)
    logw = utils.logmeanexp(logw.view(n_samples, -1).transpose(0, 1))
    if not forward:
        logw = -logw
    
    print('Last batch stats %.4f' % (logw.mean().cpu().item()))

    return logw
