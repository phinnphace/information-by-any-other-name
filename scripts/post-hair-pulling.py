#!/usr/bin/env python3
"""
Gaussian-Bernoulli RBM with rectified hidden units
Following Nair & Hinton (2010) "Rectified Linear Units Improve Restricted
Boltzmann Machines", with training modifications from the Gaussian-Bernoulli
RBM stability literature.

Purpose: train an RBM that preserves information about the input image,
then inspect reconstructions directly (no probes).

Diagnostics built in:
  - Reconstruction MSE per epoch
  - Gradient norm per epoch
  - Reconstruction image dumps every N epochs
  - Hidden unit activation statistics
"""
import os
import struct
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from codecs import decode
from collections import defaultdict
import json
import matplotlib.pyplot as plt

# ============================================================
# CONFIGURATION
# ============================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TARGET_SIZE = (100, 100)
IMAGE_DIM = TARGET_SIZE[0] * TARGET_SIZE[1]  # 10000 visible units

# RBM hyperparameters
N_HIDDEN = 1024              # hidden units (tune: 512-2048)
LEARNING_RATE = 0.001        # Gaussian layer lr (lower than binary; try 0.001-0.005)
MOMENTUM = 0.9
WEIGHT_DECAY = 0.0001
CD_K = 1                     # contrastive divergence steps
BATCH_SIZE = 32
EPOCHS = 100
NOISE_STD = 0.01             # noise added to rectified hidden units (Nair & Hinton)
GRAD_CLIP = 5.0              # gradient clipping for stability
RECON_DUMP_EVERY = 10        # save reconstructions every N epochs

# Data
TARGET = '的'
DISTRACTORS = ['地', '得', '也', '他', '她']
ALL_CHARS = [TARGET] + DISTRACTORS
TRAIN_GNT_DIR = '/mnt/c/Users/pmark/CASIA_data/HWDB1.1trn_gnt'

# Which condition to train (change to 'contextual' for the second RBM)
CONDITION = 'isolated'       # 'isolated' or 'contextual'

OUTPUT_DIR = f'models/rbm_{CONDITION}'
os.makedirs(OUTPUT_DIR, exist_ok=True)

np.random.seed(42)
torch.manual_seed(42)

print(f"Device: {DEVICE}")
print(f"Condition: {CONDITION}")
print(f"Visible units: {IMAGE_DIM}")
print(f"Hidden units: {N_HIDDEN}")
print("=" * 60)

# ============================================================
# GNT PARSER
# ============================================================
def parse_gnt_file(filepath):
    results = []
    with open(filepath, 'rb') as f:
        while True:
            packed_length = f.read(4)
            if packed_length == b'': break
            length = struct.unpack('<I', packed_length)[0]
            raw_label = struct.unpack('>cc', f.read(2))
            width = struct.unpack('<H', f.read(2))[0]
            height = struct.unpack('<H', f.read(2))[0]
            raw_bytes = f.read(height * width)
            photo_bytes = struct.unpack('{}B'.format(height * width), raw_bytes)
            label = decode(raw_label[0] + raw_label[1], encoding='gb2312')
            img_array = np.array(photo_bytes, dtype=np.uint8).reshape(height, width)
            results.append((Image.fromarray(img_array, mode='L'), label))
    return results

# ============================================================
# DATA LOADING
# ============================================================
def load_isolated_data():
    """Load isolated character images as flattened float vectors."""
    images = []
    char_to_label = {ch: i for i, ch in enumerate(ALL_CHARS)}
    gnt_files = sorted([f for f in os.listdir(TRAIN_GNT_DIR) if f.endswith('.gnt')])
    
    for gnt_file in gnt_files:
        samples = parse_gnt_file(os.path.join(TRAIN_GNT_DIR, gnt_file))
        for img, label in samples:
            if label in char_to_label:
                img = img.resize(TARGET_SIZE, Image.LANCZOS)
                arr = np.array(img, dtype=np.float32).flatten() / 255.0
                images.append(arr)
    
    return np.stack(images)


def load_contextual_data():
    """Load bigram crops as flattened float vectors."""
    from datasets import load_dataset
    
    images = []
    ds = load_dataset('Teklia/CASIA-HWDB2-line', split='validation')
    
    # Positives
    pos_count = 0
    for example in ds:
        text = example['text']
        if TARGET in text:
            img = example['image']
            w, h = img.size
            positions = [i for i, ch in enumerate(text) if ch == TARGET]
            for pos in positions:
                char_width = w / len(text)
                start_pos = max(0, pos - 1)
                end_pos = min(len(text), pos + 2)
                x1 = int(start_pos * char_width)
                x2 = int(end_pos * char_width)
                crop = img.crop((x1, 0, x2, h))
                crop = crop.resize(TARGET_SIZE, Image.LANCZOS)
                arr = np.array(crop, dtype=np.float32).flatten() / 255.0
                images.append(arr)
                pos_count += 1
                if pos_count >= 150:
                    break
        if pos_count >= 150:
            break
    
    # Negatives
    neg_count = 0
    for example in ds:
        text = example['text']
        if TARGET not in text and len(text) >= 2:
            img = example['image']
            w, h = img.size
            char_width = w / len(text)
            for i in range(len(text) - 1):
                x1 = int(i * char_width)
                x2 = int((i + 2) * char_width)
                crop = img.crop((x1, 0, x2, h))
                crop = crop.resize(TARGET_SIZE, Image.LANCZOS)
                arr = np.array(crop, dtype=np.float32).flatten() / 255.0
                images.append(arr)
                neg_count += 1
                if neg_count >= 150:
                    break
        if neg_count >= 150:
            break
    
    return np.stack(images)


# ============================================================
# GAUSSIAN-BERNOULLI RBM WITH RECTIFIED HIDDEN UNITS
# ============================================================
class GaussianBernoulliRBM(nn.Module):
    """
    Gaussian-Bernoulli RBM with rectified (noisy) hidden units.
    
    Visible units: continuous (Gaussian), representing pixel intensities.
    Hidden units: noisy rectified linear (Nair & Hinton 2010).
    
    Energy function:
        E(v, h) = sum_i (v_i - b_i)^2 / (2 sigma_i^2)
                  - sum_j c_j h_j
                  - sum_ij (v_i / sigma_i^2) W_ij h_j
    
    Where v are visible, h are hidden, W are weights, b and c are biases.
    """
    def __init__(self, n_visible, n_hidden, sigma=1.0, noise_std=0.01):
        super().__init__()
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.sigma = sigma
        self.noise_std = noise_std
        
        # Weights
        self.W = nn.Parameter(torch.randn(n_visible, n_hidden) * 0.01)
        # Visible bias (for Gaussian visible, this is the mean)
        self.b = nn.Parameter(torch.zeros(n_visible))
        # Hidden bias
        self.c = nn.Parameter(torch.zeros(n_hidden))
    
    def sample_hidden(self, v):
        """
        Given visible v, compute hidden activations with rectified units.
        Returns (hidden_activation, hidden_sample).
        
        Nair & Hinton: hidden pre-activation is W^T v / sigma^2 + c.
        Rectified: h = max(0, pre-activation + noise).
        """
        pre_activation = (v / self.sigma**2) @ self.W + self.c
        # Rectified linear with noise (Nair & Hinton)
        noise = torch.randn_like(pre_activation) * self.noise_std
        h_activation = torch.clamp(pre_activation + noise, min=0.0)
        return h_activation, h_activation  # For rectified units, activation = sample
    
    def sample_visible(self, h):
        """
        Given hidden h, reconstruct visible v.
        Gaussian: v = b + sigma^2 * (W @ h) + noise * sigma.
        """
        mean = self.b + self.sigma**2 * (h @ self.W.t())
        v = mean + torch.randn_like(mean) * self.sigma
        return mean, v  # Return both mean (for expected reconstruction) and sample
    
    def free_energy(self, v):
        """Free energy of visible configuration."""
        vbias_term = 0.5 * torch.sum((v - self.b)**2 / self.sigma**2, dim=1)
        wx_b = (v / self.sigma**2) @ self.W + self.c
        # For rectified units, the log partition over positive half-line is:
        # log(1 + exp(x)) approximated by softplus. Nair & Hinton use
        # the stepped sigmoid approximation. For energy, we use softplus.
        hidden_term = torch.sum(torch.log1p(torch.exp(wx_b)), dim=1)
        return vbias_term - hidden_term


def contrastive_divergence(model, v0, k=1):
    """
    k-step contrastive divergence for Gaussian-Bernoulli RBM.
    Returns gradient estimates for W, b, c and reconstruction error.
    """
    batch_size = v0.size(0)
    
    # Positive phase
    h0_act, h0 = model.sample_hidden(v0)
    
    # Negative phase (k steps of Gibbs)
    v_k = v0
    for _ in range(k):
        h_k_act, h_k = model.sample_hidden(v_k)
        _, v_k = model.sample_visible(h_k)
    
    h_k_act, h_k = model.sample_hidden(v_k)
    
    # Gradient estimates
    # For Gaussian visible: grad_W = (v0^T h0 / sigma^2) - (v_k^T h_k / sigma^2)
    # scaled by 1/batch_size
    pos_grad_W = (v0.t() @ h0_act) / batch_size
    neg_grad_W = (v_k.t() @ h_k_act) / batch_size
    
    pos_grad_b = v0.mean(dim=0) / model.sigma**2
    neg_grad_b = v_k.mean(dim=0) / model.sigma**2
    
    pos_grad_c = h0_act.mean(dim=0)
    neg_grad_c = h_k_act.mean(dim=0)
    
    grad_W = (pos_grad_W - neg_grad_W) / model.sigma**2
    grad_b = pos_grad_b - neg_grad_b
    grad_c = pos_grad_c - neg_grad_c
    
    # Reconstruction error (MSE between input and mean reconstruction)
    with torch.no_grad():
        mean_recon, _ = model.sample_visible(h0_act)
        recon_mse = torch.mean((v0 - mean_recon)**2).item()
    
    return grad_W, grad_b, grad_c, recon_mse


# ============================================================
# TRAINING WITH DIAGNOSTICS
# ============================================================
def train_rbm(data, model, epochs=EPOCHS, batch_size=BATCH_SIZE, lr=LEARNING_RATE):
    """
    Train the RBM with contrastive divergence, using momentum and gradient clipping.
    Returns per-epoch diagnostic history.
    """
    n_samples = data.shape[0]
    data_tensor = torch.tensor(data, dtype=torch.float32).to(DEVICE)
    
    # Momentum buffers
    vel_W = torch.zeros_like(model.W)
    vel_b = torch.zeros_like(model.b)
    vel_c = torch.zeros_like(model.c)
    
    history = {
        'epoch': [],
        'recon_mse': [],
        'grad_norm': [],
        'hidden_mean_activation': [],
        'hidden_sparsity': [],  # fraction of hidden units with activation > 0
    }
    
    for epoch in range(epochs):
        # Shuffle
        perm = torch.randperm(n_samples)
        epoch_recon_mse = []
        epoch_grad_norms = []
        epoch_hidden_mean = []
        epoch_hidden_sparsity = []
        
        for i in range(0, n_samples, batch_size):
            batch_idx = perm[i:i+batch_size]
            v0 = data_tensor[batch_idx]
            
            if v0.size(0) < 2:
                continue
            
            grad_W, grad_b, grad_c, recon_mse = contrastive_divergence(
                model, v0, k=CD_K
            )
            
            # Gradient clipping
            grad_norm = torch.sqrt(
                (grad_W**2).sum() + (grad_b**2).sum() + (grad_c**2).sum()
            ).item()
            if grad_norm > GRAD_CLIP:
                scale = GRAD_CLIP / grad_norm
                grad_W = grad_W * scale
                grad_b = grad_b * scale
                grad_c = grad_c * scale
                grad_norm = GRAD_CLIP
            
            # Momentum update
            vel_W = MOMENTUM * vel_W + lr * (grad_W - WEIGHT_DECAY * model.W)
            vel_b = MOMENTUM * vel_b + lr * grad_b
            vel_c = MOMENTUM * vel_c + lr * grad_c
            
            # Apply
            with torch.no_grad():
                model.W += vel_W
                model.b += vel_b
                model.c += vel_c
            
            epoch_recon_mse.append(recon_mse)
            epoch_grad_norms.append(grad_norm)
            
            with torch.no_grad():
                h_act, _ = model.sample_hidden(v0)
                epoch_hidden_mean.append(h_act.mean().item())
                epoch_hidden_sparsity.append((h_act > 0).float().mean().item())
        
        # Record epoch stats
        if epoch_recon_mse:
            history['epoch'].append(epoch + 1)
            history['recon_mse'].append(np.mean(epoch_recon_mse))
            history['grad_norm'].append(np.mean(epoch_grad_norms))
            history['hidden_mean_activation'].append(np.mean(epoch_hidden_mean))
            history['hidden_sparsity'].append(np.mean(epoch_hidden_sparsity))
            
            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"  Epoch {epoch+1:3d}: "
                      f"recon_mse={np.mean(epoch_recon_mse):.5f}, "
                      f"grad_norm={np.mean(epoch_grad_norms):.3f}, "
                      f"hidden_mean={np.mean(epoch_hidden_mean):.3f}, "
                      f"sparsity={np.mean(epoch_hidden_sparsity):.3f}")
        
        # Save reconstruction dump periodically
        if (epoch + 1) % RECON_DUMP_EVERY == 0:
            save_reconstruction_dump(model, data_tensor, epoch + 1)
    
    return history


def save_reconstruction_dump(model, data_tensor, epoch, n_images=8):
    """Save a grid of original vs. reconstructed images."""
    with torch.no_grad():
        indices = torch.randperm(len(data_tensor))[:n_images]
        originals = data_tensor[indices]
        h_act, _ = model.sample_hidden(originals)
        recon_mean, _ = model.sample_visible(h_act)
    
    fig, axes = plt.subplots(2, n_images, figsize=(n_images * 1.5, 3.5))
    for i in range(n_images):
        orig = originals[i].cpu().numpy().reshape(TARGET_SIZE)
        recon = recon_mean[i].cpu().numpy().reshape(TARGET_SIZE)
        
        axes[0, i].imshow(orig, cmap='gray', vmin=0, vmax=1)
        axes[0, i].axis('off')
        axes[0, i].set_title('Original', fontsize=8)
        
        axes[1, i].imshow(recon, cmap='gray', vmin=0, vmax=1)
        axes[1, i].axis('off')
        axes[1, i].set_title('Reconstruction', fontsize=8)
    
    plt.suptitle(f'Epoch {epoch}', fontsize=10)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/recon_epoch{epoch:03d}.png', dpi=100, bbox_inches='tight')
    plt.close()


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    print("\nLoading data...")
    if CONDITION == 'isolated':
        data = load_isolated_data()
    else:
        data = load_contextual_data()
    print(f"  Loaded {data.shape[0]} images, shape {data.shape[1]}")
    
    print(f"\nInitializing RBM ({IMAGE_DIM} visible, {N_HIDDEN} hidden)...")
    model = GaussianBernoulliRBM(IMAGE_DIM, N_HIDDEN, sigma=1.0, noise_std=NOISE_STD).to(DEVICE)
    
    print(f"\nTraining for {EPOCHS} epochs...")
    print(f"  lr={LEARNING_RATE}, momentum={MOMENTUM}, CD_k={CD_K}")
    print(f"  grad_clip={GRAD_CLIP}, noise_std={NOISE_STD}")
    print()
    
    history = train_rbm(data, model)
    
    # Save model and history
    torch.save(model.state_dict(), f'{OUTPUT_DIR}/rbm_final.pth')
    with open(f'{OUTPUT_DIR}/history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    # Final reconstruction dump (larger grid)
    final_data_tensor = torch.tensor(data[:64], dtype=torch.float32).to(DEVICE)
    save_reconstruction_dump(model, final_data_tensor, EPOCHS, n_images=8)
    
    print(f"\nDone. Model saved to {OUTPUT_DIR}/rbm_final.pth")
    print(f"Reconstruction history saved to {OUTPUT_DIR}/history.json")
    
    # Diagnostic summary
    print("\n" + "=" * 60)
    print("TRAINING DIAGNOSTIC SUMMARY")
    print("=" * 60)
    if len(history['recon_mse']) >= 2:
        initial_mse = history['recon_mse'][0]
        final_mse = history['recon_mse'][-1]
        reduction = (initial_mse - final_mse) / initial_mse * 100
        print(f"  Reconstruction MSE: {initial_mse:.5f} → {final_mse:.5f} "
              f"({reduction:.1f}% reduction)")
    
    if len(history['hidden_sparsity']) >= 2:
        final_sparsity = history['hidden_sparsity'][-1]
        print(f"  Hidden unit sparsity (fraction active): {final_sparsity:.3f}")
        if final_sparsity < 0.01:
            print(f"  WARNING: Very low sparsity — hidden units may be dying.")
        elif final_sparsity > 0.99:
            print(f"  WARNING: Very high sparsity — hidden units may be saturated.")
    
    print("\n  Next steps:")
    print(f"  1. Inspect {OUTPUT_DIR}/recon_epoch*.png to see if reconstructions are legible.")
    print(f"  2. If reconstructions are noise or uniform, try: lower lr (0.0005), "
          f"more CD steps, more epochs, or lower noise_std.")
    print(f"  3. Once reconstructions are legible on isolated data, "
          f"change CONDITION to 'contextual' and repeat.")