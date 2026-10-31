import math
from typing import Union

import torch


def pad_t_like_x(t, x):
    if isinstance(t, (float, int)):
        return t
    return t.reshape(-1, *([1] * (len(x.shape) - 1)))


# ----------------------------------------------------------------------------


class ConditionalFlowMatcher:
    """Independent Conditional Flow Matching

    Improving and Generalizing Flow-Based Generative Models
    with minibatch optimal transport, Tong et al. (2023)
    """

    def __init__(self, sigma: Union[float, int] = 0.0, name: str = "cfm"):
        self.sigma = sigma
        self.name = name

    def __repr__(self):
        return f"{self.name} ({self.sigma})"

    def mu_t(self, x0, x1, t):
        """mean of the probability path"""
        t = pad_t_like_x(t, x0)
        return t * x1 + (1 - t) * x0

    def sigma_t(self, t):
        """std of the probability path"""
        del t
        return self.sigma

    def sample_xt(self, x0, x1, t, epsilon):
        """sample from the probability path N(t * x1 + (1 - t) * x0, sigma),
        see (Eq.14)"""
        mu_t = self.mu_t(x0, x1, t)
        sigma_t = self.sigma_t(t)
        sigma_t = pad_t_like_x(sigma_t, x0)
        return mu_t + sigma_t * epsilon

    def conditional_flow(self, x0, x1, t, xt):
        """conditional vector field ut(x1|x0) = x1 - x0, see (Eq.15)"""
        del t, xt
        return x1 - x0

    def sample_noise_like(self, x):
        return torch.randn_like(x)

    def sample_location_and_conditional_flow(self, x0, x1, t=None, return_noise=False):
        """sample xt (drawn from N(t * x1 + (1 - t) * x0, sigma)) and
        the conditional vector field ut(x1|x0) = x1 - x0, see (Eq.15)"""
        if t is None:
            t = torch.rand(x0.shape[0], dtype=x0.dtype, device=x0.device)
        assert len(t) == x0.shape[0], "t has to have batch size dimension"

        eps = self.sample_noise_like(x0)
        xt = self.sample_xt(x0, x1, t, eps)
        ut = self.conditional_flow(x0, x1, t, xt)
        if return_noise:
            return t, xt, ut, eps
        else:
            return t, xt, ut

    def lambda_f(self, t):
        """weighting schedule"""
        return 2 * self.sigma_t(t) / (self.sigma**2 + 1e-8)


# ----------------------------------------------------------------------------


class VPConditionalFlowMatcher(ConditionalFlowMatcher):
    """Variance Preserving Stochastic Interpolant

    Stochastic Interpolants: A Unifying Framework for
    Flows and Diffusions, Albergo et al. (2023)
    """

    def __init__(self, sigma: Union[float, int] = 0.0):
        super().__init__(sigma, name="vp")

    def mu_t(self, x0, x1, t):
        """mean of the probability path (Eq.5)"""
        t = pad_t_like_x(t, x0)
        return torch.cos(math.pi / 2 * t) * x0 + torch.sin(math.pi / 2 * t) * x1

    def conditional_flow(self, x0, x1, t, xt):
        """conditional vector field ut(x1|x0) = pi/2 (cos(pi*t/2) x1 - sin(pi*t/2) x0),
        see (Eq.21)
        """
        del xt
        t = pad_t_like_x(t, x0)
        return (
            math.pi
            / 2
            * (torch.cos(math.pi / 2 * t) * x1 - torch.sin(math.pi / 2 * t) * x0)
        )


# ----------------------------------------------------------------------------


class TargetConditionalFlowMatcher(ConditionalFlowMatcher):
    """Target OT Conditional Flow Matching.

    Flow Matching for Generative Modeling, Lipman et al. (2023)
    """

    def __init__(self, sigma: Union[float, int] = 0.0):
        super().__init__(sigma, name="fm")

    def mu_t(self, x0, x1, t):
        """mean of the probability path"""
        del x0
        t = pad_t_like_x(t, x1)
        return t * x1

    def sigma_t(self, t):
        """std of the probability path"""
        return t * self.sigma - t + 1

    def conditional_flow(self, x0, x1, t, xt):
        """conditional vector field ut(x1|x0) = (x1 - (1 - sigma) xt) / (1 - (1 - sigma) t),"""
        del x0
        t = pad_t_like_x(t, x1)
        return (x1 - (1 - self.sigma) * xt) / (1 - (1 - self.sigma) * t)


# ----------------------------------------------------------------------------


CFM_DICT = {
    "cfm": ConditionalFlowMatcher,
    "vp": VPConditionalFlowMatcher,
    "fm": TargetConditionalFlowMatcher,
}


def str_to_cfm(cfm_name, sigma=0.0):
    if cfm_name not in CFM_DICT:
        raise ValueError(f"Invalid solver: {cfm_name}")
    solver = CFM_DICT[cfm_name]
    return solver(sigma)
