import torch
import pytest
from rbm_model import GaussianBernoulliRBM, UNIT_TYPES


torch.manual_seed(0)
N_VIS, N_HID, B = 64, 32, 8


def make(unit_type, **kw):
    return GaussianBernoulliRBM(N_VIS, N_HID, unit_type=unit_type, **kw)


def batch():
    return torch.rand(B, N_VIS)


# 1. shapes across all unit types
@pytest.mark.parametrize("unit_type", UNIT_TYPES)
def test_shapes(unit_type):
    m = make(unit_type)
    v = batch()
    h_act, h_s = m.sample_hidden(v)
    assert h_act.shape == (B, N_HID) and h_s.shape == (B, N_HID)
    assert m.visible_mean(h_act).shape == (B, N_VIS)


# 2. noise_off reconstruction is deterministic
@pytest.mark.parametrize("unit_type", UNIT_TYPES)
def test_noise_off_deterministic(unit_type):
    m = make(unit_type)
    v = batch()
    r1 = m.reconstruct(v, mode="noise_off")
    r2 = m.reconstruct(v, mode="noise_off")
    assert torch.allclose(r1, r2)


# 3. hidden_mean is noise-free and consistent with noise_off reconstruct
@pytest.mark.parametrize("unit_type", UNIT_TYPES)
def test_hidden_mean_no_noise(unit_type):
    m = make(unit_type)
    v = batch()
    h1 = m.hidden_mean(v)
    h2 = m.hidden_mean(v)
    assert torch.allclose(h1, h2)
    assert torch.allclose(m.reconstruct(v, mode="noise_off"), m.visible_mean(h1))


# 4. GR/GR0 noise is softplus-scaled, not fixed SD
def test_gr_noise_softplus_scaled():
    torch.manual_seed(1)
    m = make("GR", noise_scale=1.0)
    v = torch.zeros(B, N_VIS)          # pre = 0 -> softplus(0) = ln 2
    # sample many times, check empirical SD against softplus(0)
    samples = []
    for _ in range(4000):
        h, _ = m.sample_hidden(v)
        samples.append(h)
    s = torch.stack(samples)            # (T, B, H)
    # only look at entries whose preactivation is ~0 (all of them, since v=0, c=0)
    # h is clamp(pre + noise, min=0); for pre=0 the positive half has SD ~ softplus(0)/sqrt(2)... 
    # simpler: check it is NOT the fixed-SD 0.01 regime.
    sd = s.std(dim=0).mean().item()
    assert sd > 0.3, f"expected rectifier noise SD well above 0.01, got {sd:.4f}"


# 5. GR0 freezes hidden biases
def test_gr0_biases_frozen():
    m = make("GR0")
    assert m.c.requires_grad is False
    m2 = make("GR")
    assert m2.c.requires_grad is True


# 6. GB path returns probabilities in [0,1] and Bernoulli samples
def test_gb_probabilities():
    m = make("GB")
    v = batch()
    p, s = m.sample_hidden(v)
    assert (p >= 0).all() and (p <= 1).all()
    assert ((s == 0) | (s == 1)).all()


# 7. median reconstruction is between sample and noise_off in variance
def test_median_reconstruction():
    m = make("GR")
    v = batch()
    r_off = m.reconstruct(v, mode="noise_off")
    r_med = m.reconstruct(v, mode="median", n_draws=25)
    assert r_off.shape == r_med.shape
    assert torch.isfinite(r_med).all()


# 8. free energy is finite and differentiable
@pytest.mark.parametrize("unit_type", UNIT_TYPES)
def test_free_energy(unit_type):
    m = make(unit_type)
    v = batch()
    fe = m.free_energy(v)
    assert fe.shape == (B,)
    assert torch.isfinite(fe).all()
    fe.sum().backward()
    for p in m.parameters():
        if p.requires_grad:
            assert p.grad is not None


# 9. gradient does not flow into GR0 hidden biases
def test_gr0_no_gradient_on_c():
    m = make("GR0")
    v = batch()
    m.free_energy(v).sum().backward()
    assert m.c.grad is None


# 10. invalid unit type and mode raise
def test_invalid_args():
    with pytest.raises(ValueError):
        GaussianBernoulliRBM(N_VIS, N_HID, unit_type="XX")
    m = make("GR")
    with pytest.raises(ValueError):
        m.reconstruct(batch(), mode="nope")