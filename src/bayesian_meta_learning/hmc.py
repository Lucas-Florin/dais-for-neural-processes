from typing import Callable
from typing import Optional

import torch


def hmc_trajectory(current_z: torch.Tensor,
                   current_v: torch.Tensor,
                   grad_U: Callable,
                   epsilon: torch.Tensor,
                   L: Optional[int] = 10,
                   forward_time: bool = True,
                   ):
    """Propose new state-velocity pair with leap-frog integrator.

    This function does not yet do the accept-reject step.
    Follows algo box in Figure 2 of https://arxiv.org/pdf/1206.1901.pdf.

    Args:
        current_z: current position
        current_v: current velocity/momentum
        grad_U: function to compute gradients w.r.t. U
        epsilon: step size
        L: number of leap-frog steps

    Returns:
        proposed state z and velocity v after the leap-frog steps
    """
    epsilon = epsilon[:, :, None]
    epsilon_rand = epsilon if forward_time else -epsilon
    z = current_z
    v = current_v - .5 * epsilon_rand * grad_U(z)

    for i in range(1, L + 1):
        z = z + epsilon_rand * v
        if i != L:
            v = v - epsilon_rand * grad_U(z)

    v = v - .5 * epsilon_rand * grad_U(z)
    v = -v

    return z, v


def accept_reject(current_z: torch.Tensor,
                  current_v: torch.Tensor,
                  z: torch.Tensor,
                  v: torch.Tensor,
                  epsilon: torch.Tensor,
                  accept_hist: torch.Tensor,
                  hist_len: int,
                  U: Callable,
                  K: Callable,
                  max_step_size: Optional[float] = 0.5,
                  min_step_size: Optional[float] = 1e-4,
                  acceptance_threshold: Optional[float] = 0.65,
                  epsilon_update_factor = 0.98,
                  adapt_step_size=True,
                  do_accept_reject_step=True,
                  use_accept_hist=True,
                  rng: torch.Generator = None,
                  ):
    """Accept/reject based on Hamiltonians for current and propose.

    Args:
        current_z: position *before* leap-frog steps
        current_v: speed *before* leap-frog steps
        z: position *after* leap-frog steps
        v: speed *after* leap-frog steps
        epsilon: step size of leap-frog.
        accept_hist: a tensor of size (batch_size,), each component of which is
            the number of time the trajectory is accepted
        hist_len: an int for the chain length after the current step
        U: function to compute potential energy
        K: function to compute kinetic energy
        max_step_size: maximum step size for leap-frog
        min_step_size: minimum step size for leap-frog
        acceptance_threshold: threshold acceptance rate; increase the step size
            if the chain is accepted more than this, and decrease otherwise

    Returns:
        the new state z, the adapted step size epsilon, and the updated
        accept-reject history
    """
    with torch.no_grad():
        current_Hamil = K(current_v) + U(current_z)
        propose_Hamil = K(v) + U(z)
        prob = torch.clamp_max(torch.exp(current_Hamil - propose_Hamil), 1.)
        accept = torch.gt(prob, torch.rand(prob.shape, generator=rng))
        if do_accept_reject_step:
            z = accept[..., None] * z + ~accept[..., None] * current_z
        accept_hist.add_(accept)
        # print(accept.float().mean())
        if adapt_step_size:
            accept_reference = accept_hist / hist_len if use_accept_hist else accept
            criteria = torch.gt(accept_reference, acceptance_threshold)
            adapt = criteria / epsilon_update_factor + ~criteria * epsilon_update_factor
            epsilon = (epsilon * adapt).clamp(min_step_size, max_step_size)

    return z, epsilon, accept_hist
