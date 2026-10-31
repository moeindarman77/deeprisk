import numpy as np
import torch
from arrgen.generating.conditioning_methods import *

class DiffusionSampler:
    def __init__(self, net):
        super().__init__()
        self.net = net

    @torch.no_grad()
    def edm_sampler(
        self,
        latents,
        condition=None,
        randn_like=torch.randn_like,
        num_steps=18,
        sigma_min=0.002,
        sigma_max=80,
        rho=7,
        S_churn=0,
        S_min=0,
        S_max=float("inf"),
        S_noise=1,
    ):
        """Proposed EDM sampler (Algorithm 2)."""
        # Adjust noise levels based on what's supported by the network.
        sigma_min = max(sigma_min, self.net.sigma_min)
        sigma_max = min(sigma_max, self.net.sigma_max)

        # Time step discretization.
        step_indices = torch.arange(
            num_steps, dtype=torch.float32, device=latents.device
        )
        t_steps = (
            sigma_max ** (1 / rho)
            + step_indices
            / (num_steps - 1)
            * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))
        ) ** rho
        t_steps = torch.cat(
            [self.net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])]
        )  # t_N = 0

        # Main sampling loop.
        x_next = latents.to(torch.float32) * t_steps[0]
        for i, (t_cur, t_next) in enumerate(
            zip(t_steps[:-1], t_steps[1:])
        ):  # 0, ..., N-1
            x_cur = x_next

            # Increase noise temporarily.
            gamma = (
                min(S_churn / num_steps, np.sqrt(2) - 1)
                if S_min <= t_cur <= S_max
                else 0
            )
            t_hat = self.net.round_sigma(t_cur + gamma * t_cur)
            x_hat = x_cur + (t_hat**2 - t_cur**2).sqrt() * S_noise * randn_like(x_cur)

            # Euler step.
            denoised = self.net(x_hat, t_hat, condition).to(torch.float32)
            d_cur = (x_hat - denoised) / t_hat
            x_next = x_hat + (t_next - t_hat) * d_cur

            # Apply 2nd order correction.
            if i < num_steps - 1:
                denoised = self.net(x_next, t_next, condition).to(torch.float32)
                d_prime = (x_next - denoised) / t_next
                x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)

        return x_next

    @torch.no_grad()
    def ablation_sampler(
        self,
        latents,
        condition=None,
        randn_like=torch.randn_like,
        num_steps=18,
        sigma_min=None,
        sigma_max=None,
        rho=7,
        solver="heun",
        discretization="edm",
        schedule="linear",
        scaling="none",
        epsilon_s=1e-3,
        C_1=0.001,
        C_2=0.008,
        M=1000,
        alpha=1,
        S_churn=0,
        S_min=0,
        S_max=float("inf"),
        S_noise=1,
    ):
        """Generalized ablation sampler, representing the superset
        of all sampling methods discussed in the paper."""
        assert solver in ["euler", "heun"]
        assert discretization in ["vp", "ve", "iddpm", "edm"]
        assert schedule in ["vp", "ve", "linear"]
        assert scaling in ["vp", "none"]

        # Helper functions for VP & VE noise level schedules.
        vp_sigma = (
            lambda beta_d, beta_min: lambda t: (
                np.e ** (0.5 * beta_d * (t**2) + beta_min * t) - 1
            )
            ** 0.5
        )
        vp_sigma_deriv = (
            lambda beta_d, beta_min: lambda t: 0.5
            * (beta_min + beta_d * t)
            * (sigma(t) + 1 / sigma(t))
        )
        vp_sigma_inv = (
            lambda beta_d, beta_min: lambda sigma: (
                (beta_min**2 + 2 * beta_d * (sigma**2 + 1).log()).sqrt() - beta_min
            )
            / beta_d
        )
        ve_sigma = lambda t: t.sqrt()
        ve_sigma_deriv = lambda t: 0.5 / t.sqrt()
        ve_sigma_inv = lambda sigma: sigma**2

        # Select default noise level range based on the specified time step discretization.
        if sigma_min is None:
            vp_def = vp_sigma(beta_d=19.9, beta_min=0.1)(t=epsilon_s)
            sigma_min = {"vp": vp_def, "ve": 0.02, "iddpm": 0.002, "edm": 0.002}[
                discretization
            ]
        if sigma_max is None:
            vp_def = vp_sigma(beta_d=19.9, beta_min=0.1)(t=1)
            sigma_max = {"vp": vp_def, "ve": 100, "iddpm": 81, "edm": 80}[
                discretization
            ]

        # Adjust noise levels based on what's supported by the network.
        sigma_min = max(sigma_min, self.net.sigma_min)
        sigma_max = min(sigma_max, self.net.sigma_max)

        # Compute corresponding betas for VP.
        vp_beta_d = (
            2
            * (np.log(sigma_min**2 + 1) / epsilon_s - np.log(sigma_max**2 + 1))
            / (epsilon_s - 1)
        )
        vp_beta_min = np.log(sigma_max**2 + 1) - 0.5 * vp_beta_d

        # Define time steps in terms of noise level.
        step_indices = torch.arange(
            num_steps, dtype=torch.float32, device=latents.device
        )
        if discretization == "vp":
            orig_t_steps = 1 + step_indices / (num_steps - 1) * (epsilon_s - 1)
            sigma_steps = vp_sigma(vp_beta_d, vp_beta_min)(orig_t_steps)
        elif discretization == "ve":
            orig_t_steps = (sigma_max**2) * (
                (sigma_min**2 / sigma_max**2) ** (step_indices / (num_steps - 1))
            )
            sigma_steps = ve_sigma(orig_t_steps)
        elif discretization == "iddpm":
            u = torch.zeros(M + 1, dtype=torch.float32, device=latents.device)
            alpha_bar = lambda j: (0.5 * np.pi * j / M / (C_2 + 1)).sin() ** 2
            for j in torch.arange(M, 0, -1, device=latents.device):  # M, ..., 1
                u[j - 1] = (
                    (u[j] ** 2 + 1) / (alpha_bar(j - 1) / alpha_bar(j)).clip(min=C_1)
                    - 1
                ).sqrt()
            u_filtered = u[torch.logical_and(u >= sigma_min, u <= sigma_max)]
            sigma_steps = u_filtered[
                ((len(u_filtered) - 1) / (num_steps - 1) * step_indices)
                .round()
                .to(torch.int32)
            ]
        else:
            assert discretization == "edm"
            sigma_steps = (
                sigma_max ** (1 / rho)
                + step_indices
                / (num_steps - 1)
                * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))
            ) ** rho

        # Define noise level schedule.
        if schedule == "vp":
            sigma = vp_sigma(vp_beta_d, vp_beta_min)
            sigma_deriv = vp_sigma_deriv(vp_beta_d, vp_beta_min)
            sigma_inv = vp_sigma_inv(vp_beta_d, vp_beta_min)
        elif schedule == "ve":
            sigma = ve_sigma
            sigma_deriv = ve_sigma_deriv
            sigma_inv = ve_sigma_inv
        else:
            assert schedule == "linear"
            sigma = lambda t: t
            sigma_deriv = lambda t: 1
            sigma_inv = lambda sigma: sigma

        # Define scaling schedule.
        if scaling == "vp":
            s = lambda t: 1 / (1 + sigma(t) ** 2).sqrt()
            s_deriv = lambda t: -sigma(t) * sigma_deriv(t) * (s(t) ** 3)
        else:
            assert scaling == "none"
            s = lambda t: 1
            s_deriv = lambda t: 0

        # Compute final time steps based on the corresponding noise levels.
        t_steps = sigma_inv(self.net.round_sigma(sigma_steps))
        t_steps = torch.cat([t_steps, torch.zeros_like(t_steps[:1])])  # t_N = 0

        # Main sampling loop.
        t_next = t_steps[0]
        x_next = latents.to(torch.float32) * (sigma(t_next) * s(t_next))
        for i, (t_cur, t_next) in enumerate(
            zip(t_steps[:-1], t_steps[1:])
        ):  # 0, ..., N-1
            x_cur = x_next

            # Increase noise temporarily.
            gamma = (
                min(S_churn / num_steps, np.sqrt(2) - 1)
                if S_min <= sigma(t_cur) <= S_max
                else 0
            )
            t_hat = sigma_inv(self.net.round_sigma(sigma(t_cur) + gamma * sigma(t_cur)))
            x_hat = s(t_hat) / s(t_cur) * x_cur + (
                sigma(t_hat) ** 2 - sigma(t_cur) ** 2
            ).clip(min=0).sqrt() * s(t_hat) * S_noise * randn_like(x_cur)

            # Euler step.
            h = t_next - t_hat
            denoised = self.net(x_hat / s(t_hat), sigma(t_hat), condition).to(
                torch.float32
            )
            d_cur = (
                sigma_deriv(t_hat) / sigma(t_hat) + s_deriv(t_hat) / s(t_hat)
            ) * x_hat - sigma_deriv(t_hat) * s(t_hat) / sigma(t_hat) * denoised
            x_prime = x_hat + alpha * h * d_cur
            t_prime = t_hat + alpha * h

            # Apply 2nd order correction.
            if solver == "euler" or i == num_steps - 1:
                x_next = x_hat + h * d_cur
            else:
                assert solver == "heun"
                denoised = self.net(x_prime / s(t_prime), sigma(t_prime), condition).to(
                    torch.float32
                )
                d_prime = (
                    sigma_deriv(t_prime) / sigma(t_prime)
                    + s_deriv(t_prime) / s(t_prime)
                ) * x_prime - sigma_deriv(t_prime) * s(t_prime) / sigma(
                    t_prime
                ) * denoised
                x_next = x_hat + h * (
                    (1 - 1 / (2 * alpha)) * d_cur + 1 / (2 * alpha) * d_prime
                )

        return x_next

class PosteriorDiffusionSampler:
    def __init__(self, net,):
        super().__init__()
        self.net = net
        self.sr = SuperResolutionOperator(in_shape = (48,96),target_shape=(720,1440)) #  #
        # self.sr = IGRAOperator()
        # self.sr = SR_IGRA_Operator()
        self.cm = ConditioningMethod(self.sr)
        
    def pos_edm_sampler(
        self,
        latents,
        measurement,
        condition=None,
        randn_like=torch.randn_like,
        num_steps=60,
        sigma_min=0.001,
        sigma_max=80,
        rho=7,
        S_churn=0,
        S_min=0,
        S_max=float("inf"),
        S_noise=1,
        
    ):  
        # print(num_steps,sigma_min,sigma_max,rho,S_churn,S_min,S_max,S_noise)
        """Proposed EDM sampler (Algorithm 2)."""
        # Adjust noise levels based on what's supported by the network.
        sigma_min = max(sigma_min, self.net.sigma_min)
        sigma_max = min(sigma_max, self.net.sigma_max)
        # Time step discretization.
        step_indices = torch.arange(
            num_steps, dtype=torch.float32, device=latents.device
        )
        t_steps = (
            sigma_max ** (1 / rho)
            + step_indices
            / (num_steps - 1)
            * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))
        ) ** rho
        t_steps = torch.cat(
            [self.net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])]
        )  # t_N = 0

        # Main sampling loop.
        x_next = latents.to(torch.float32) * t_steps[0]
        for i, (t_cur, t_next) in enumerate(
            zip(t_steps[:-1], t_steps[1:])
        ):  # 0, ..., N-1
            x_cur = x_next.requires_grad_() 
            # Increase noise temporarily.
            gamma = (
                min(S_churn / num_steps, np.sqrt(2) - 1)
                if S_min <= t_cur <= S_max
                else 0
            )
            t_hat = self.net.round_sigma(t_cur + gamma * t_cur)
            x_hat = (x_cur + (t_hat**2 - t_cur**2).sqrt() * S_noise * randn_like(x_cur)).float()
            # Euler step.
            with torch.autocast(
                                latents.device.type, enabled=True, dtype=torch.bfloat16
            ):
                denoised = self.net(x_hat, t_hat, condition)#.float()
            d_cur = (x_hat - denoised) / t_hat
            x_next = x_hat + (t_next - t_hat) * d_cur
            exp_x_0_x_t = denoised#/t_hat # 1/s^2(x_t + s^2 * sigma^2 * score(x_t))
            likelihood_score,err = self.cm.conditioning(x_prev = x_cur,x_t = x_next,x_0_hat=exp_x_0_x_t,
                                                measurement=measurement,sigma=t_hat)
            # if i==0 or (i+1)%5==0:
            #     print(t_hat.item(),err)
            # Apply 2nd order correction.
            if i < num_steps - 1:
                with torch.no_grad():
                    with torch.autocast(latents.device.type, enabled=True, dtype=torch.bfloat16
                        ):
                        denoised = self.net(x_next.float(), t_next.float(), condition)#.float()
                    d_prime = (x_next - denoised) / t_next
                    x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime - likelihood_score)
        return x_next
