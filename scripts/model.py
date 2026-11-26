"""
Gaussian-Bernoulli RBM with rectified hidden units.

Reference: Nair & Hinton (2010), "Rectified Linear Units Improve Restricted
Boltzmann Machines", ICML. Gaussian-visible stability follows Hinton (2010),
"A Practical Guide to Training Restricted Boltzmann Machines", and Liao et al.
(2022) for gradient clipping.

Visible units are Gaussian with fixed sigma = 1.0. Inputs are expected to be
pre-scaled to unit spread by the data layer (per-pixel median removed, divided
by global MAD; ledger step 10). The model does not rescale.

Hidden units switch on `unit_type`:
  GB  : Bernoulli hidden, biases yes        (sigmoid sampler)
  GR  : noisy rectified, biases yes         max(0, pre + N(0, softplus(pre)) * scale)
  GR0 : noisy rectified, hidden biases frozen at zero
  GL  : leaky rectified, biases yes         optional; no published recipe
                                              (flagged; not used in Stage 0)

Training samples the visible layer with Gaussian noise (Hinton 2010).
Measurement reconstructs noise-free. Those are two separate code paths and
must not be confused: `reconstruct(mode='noise_off')` is the measurement path.

Design commitment: no probes. Every readout in this project is a forward pass
through the model's own weights.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


UNIT_TYPES = ("GB", "GR", "GR0", "GL")


class GaussianBernoulliRBM(nn.Module):
    def __init__(
        self,
        n_visible: int,
        n_hidden: int,
        unit_type: str = "GR",
        sigma: float = 1.0,
        noise_scale: float = 1.0,
        leak: float = 0.01,
        init_std: float = 0.01,
    ):
        super().__init__()
        if unit_type not in UNIT_TYPES:
            raise ValueError(f"unit_type must be one of {UNIT_TYPES}, got {unit_type!r}")
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.unit_type = unit_type
        self.sigma = sigma
        self.noise_scale = noise_scale
        self.leak = leak

        self.W = nn.Parameter(torch.randn(n_visible, n_hidden) * init_std)
        self.b = nn.Parameter(torch.zeros(n_visible))   # visible bias = Gaussian mean
        self.c = nn.Parameter(torch.zeros(n_hidden))    # hidden bias

        # GR0: hidden biases exist as a buffer (so state_dict shapes match across
        # unit types for seed-matched comparison) but receive no gradient.
        if unit_type == "GR0":
            self.c.requires_grad_(False)

    # -- hidden ----------------------------------------------------------
    def hidden_preactivation(self, v: torch.Tensor) -> torch.Tensor:
        # v / sigma**2 is the Gaussian-visible scaling. sigma is scalar and
        # constant here, so dividing v before the matmul and dividing the
        # gradient after are algebraically identical; the comment exists so
        # this is not "fixed" later into a different (wrong) form.
        return (v / self.sigma**2) @ self.W + self.c

    def sample_hidden(self, v: torch.Tensor):
        """Return (h_activation, h_sample). For rectified units they are equal."""
        pre = self.hidden_preactivation(v)
        if self.unit_type == "GB":
            p = torch.sigmoid(pre)
            h_sample = torch.bernoulli(p)
            return p, h_sample
        if self.unit_type == "GR":
            noise = torch.randn_like(pre) * F.softplus(pre) * self.noise_scale
            h = torch.clamp(pre + noise, min=0.0)
            return h, h
        if self.unit_type == "GR0":
            noise = torch.randn_like(pre) * F.softplus(pre) * self.noise_scale
            h = torch.clamp(pre + noise, min=0.0)
            return h, h
        if self.unit_type == "GL":
            noise = torch.randn_like(pre) * F.softplus(pre) * self.noise_scale
            x = pre + noise
            h = torch.where(x > 0, x, self.leak * x)
            return h, h
        raise AssertionError("unreachable")

    def hidden_mean(self, v: torch.Tensor) -> torch.Tensor:
        """Deterministic hidden activation, no noise. For measurement / diagnostics."""
        pre = self.hidden_preactivation(v)
        if self.unit_type == "GB":
            return torch.sigmoid(pre)
        if self.unit_type == "GL":
            return torch.where(pre > 0, pre, self.leak * pre)
        return torch.clamp(pre, min=0.0)

    # -- visible ---------------------------------------------------------
    def visible_mean(self, h: torch.Tensor) -> torch.Tensor:
        """Noise-free visible mean. This is the measurement reconstruction path."""
        return self.b + self.sigma**2 * (h @ self.W.t())

    def sample_visible(self, h: torch.Tensor):
        """Training path: sample visible with Gaussian noise (Hinton 2010)."""
        mean = self.visible_mean(h)
        v = mean + torch.randn_like(mean) * self.sigma
        return mean, v

    # -- measurement -----------------------------------------------------
    @torch.no_grad()
    def reconstruct(self, v: torch.Tensor, mode: str = "noise_off", n_draws: int = 25) -> torch.Tensor:
        """
        Reconstruct inputs. Measurement path — no visible sampling.
          mode='noise_off' : h = hidden_mean(v); v_hat = visible_mean(h). One pass.
          mode='median'    : n_draws samples of noisy hidden -> visible_mean;
                             take elementwise median. Primary view.
          mode='sample'    : one noisy-hidden draw -> visible_mean. Illustration only.
        """
        if mode == "noise_off":
            return self.visible_mean(self.hidden_mean(v))
        if mode == "sample":
            h_act, _ = self.sample_hidden(v)
            return self.visible_mean(h_act)
        if mode == "median":
            draws = []
            for _ in range(n_draws):
                h_act, _ = self.sample_hidden(v)
                draws.append(self.visible_mean(h_act))
            return torch.stack(draws, dim=0).median(dim=0).values
        raise ValueError(f"mode must be 'noise_off', 'median', or 'sample', got {mode!r}")

    # -- free energy -----------------------------------------------------
    def free_energy(self, v: torch.Tensor) -> torch.Tensor:
        """
        Free energy for rectified / leaky / Bernoulli hidden units.

        For binary hidden, the hidden term is softplus(pre).
        For rectified hidden with no bias, the integral over h in [0, inf) of
        exp(c h + s h) also yields softplus(pre) up to constants, which is the
        form used here (and in Nair & Hinton). For GR with a learned bias c,
        softplus(pre) = softplus(W v / sigma^2 + c) is the energy term the
        code uses; this matches the implementation of the visible-to-hidden
        pass and is the quantity monitored for the held-out free-energy gap.
        """
        vbias_term = 0.5 * torch.sum((v - self.b) ** 2 / self.sigma**2, dim=1)
        pre = self.hidden_preactivation(v)
        hidden_term = torch.sum(F.softplus(pre), dim=1)
        return vbias_term - hidden_term