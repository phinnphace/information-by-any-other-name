"""
Stage 0 Training Loop for the Gaussian-Bernoulli RBM.
Loads ISO and CTX data, applies median/MAD scaling, splits, and trains.
"""
import csv
import random
import numpy as np
import torch
import torch.optim as optim
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt

from model import GaussianBernoulliRBM
from seeds import pull_batch, assign, write_seeds_used

# --- Configuration ---
ISO_TARGET_DIR = Path('/home/phinnm/Desktop/character_context_project/notebooks/data/isolated/target')
ISO_DISTRACTOR_DIR = Path('/home/phinnm/Desktop/character_context_project/notebooks/data/isolated/distractors')
CTX_CSV = Path('ledger/manual_crop_coords.csv')
CTX_CROP_DIR = Path('ledger/manual_crops')

N_VISIBLE = 32 * 32
N_HIDDEN = 128
BATCH_SIZE = 10
EPOCHS = 150
LEARNING_RATE = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_images():
    images = []
    writers = []
    is_ctx = []
    
    # 1. ISO
    for img_path in ISO_TARGET_DIR.glob('*.png'):
        img = Image.open(img_path).convert('L').resize((32, 32), Image.LANCZOS)
        images.append(np.array(img, dtype=np.float32).flatten() / 255.0)
        writers.append("ISO")
        is_ctx.append(False)

    for char_dir in ISO_DISTRACTOR_DIR.iterdir():
        if char_dir.is_dir():
            for img_path in char_dir.glob('*.png'):
                img = Image.open(img_path).convert('L').resize((32, 32), Image.LANCZOS)
                images.append(np.array(img, dtype=np.float32).flatten() / 255.0)
                writers.append("ISO")
                is_ctx.append(False)

    # 2. CTX
    with open(CTX_CSV, newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            safe_name = row['image_file'].replace('/', '_').replace('.png', '')
            img_path = CTX_CROP_DIR / f"{safe_name}.png"
            if img_path.exists():
                img = Image.open(img_path).convert('L').resize((32, 32), Image.LANCZOS)
                images.append(np.array(img, dtype=np.float32).flatten() / 255.0)
                writers.append(row['writer'])
                is_ctx.append(True)

    return np.array(images), np.array(writers), np.array(is_ctx)

def preprocess_data(X):
    pixel_median = np.median(X, axis=0)
    X_centered = X - pixel_median
    global_mad = np.median(np.abs(X_centered))
    if global_mad == 0: global_mad = 1e-6
    return X_centered / global_mad

def main():
    print(f"Device: {DEVICE}")
    
    X_raw, writers, is_ctx = load_images()
    print(f"Loaded {len(X_raw)} total images.")
    
    X_scaled = preprocess_data(X_raw)
    
    # --- FIXED SPLIT LOGIC ---
    # For ISO, we split 80/20 randomly (since we lack writer IDs).
    # For CTX, we split by writer.
    iso_idx = np.where(~is_ctx)[0]
    ctx_idx = np.where(is_ctx)[0]
    
    # Shuffle ISO
    rng = np.random.default_rng(42)
    rng.shuffle(iso_idx)
    split = int(len(iso_idx) * 0.8)
    iso_train, iso_val = iso_idx[:split], iso_idx[split:]
    
    # Split CTX by writer (just use last writer as val for Stage 0)
    unique_ctx_writers = np.unique(writers[ctx_idx])
    val_writer = unique_ctx_writers[-1]
    ctx_train = ctx_idx[writers[ctx_idx] != val_writer]
    ctx_val = ctx_idx[writers[ctx_idx] == val_writer]
    
    # Combine
    train_idx = np.concatenate([iso_train, ctx_train])
    val_idx = np.concatenate([iso_val, ctx_val])
    
    X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
    print(f"Train size: {len(X_train)} (ISO: {len(iso_train)}, CTX: {len(ctx_train)})")
    print(f"Val size: {len(X_val)} (ISO: {len(iso_val)}, CTX: {len(ctx_val)})")
    
    # --- SEED & MODEL ---
    seed = pull_batch(1)[0]
    assign(seed, "stage0_run_1")
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    model = GaussianBernoulliRBM(N_VISIBLE, N_HIDDEN, unit_type="GR").to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    v0_train = torch.tensor(X_train, dtype=torch.float32).to(DEVICE)
    
    print("Starting training...")
    for epoch in range(EPOCHS):
        model.train()
        optimizer.zero_grad()
        
        # CD-1
        h0_act, _ = model.sample_hidden(v0_train)
        v_k = v0_train
        h_k_act, _ = model.sample_hidden(v_k)
        _, v_k = model.sample_visible(h_k_act)
        h_k_act, _ = model.sample_hidden(v_k)
        
        loss = model.free_energy(v0_train).mean() - model.free_energy(v_k).mean()
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {loss.item():.4f}")
            
    # --- SAVE & VISUALIZE ---
    model_dir = Path('models/stage0')
    model_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), model_dir / 'rbm_stage0.pth')
    write_seeds_used(model_dir, 1, [seed], ["stage0_run_1"])
    
    # Reconstruct 3 random val images
    print("Generating reconstructions...")
    model.eval()
    v_val = torch.tensor(X_val[:3], dtype=torch.float32).to(DEVICE)
    with torch.no_grad():
        recons = model.reconstruct(v_val, mode="noise_off")
        
    fig, axes = plt.subplots(2, 3, figsize=(9, 6))
    for i in range(3):
        axes[0, i].imshow(v_val[i].cpu().numpy().reshape(32, 32), cmap='gray')
        axes[0, i].set_title("Original")
        axes[0, i].axis('off')
        axes[1, i].imshow(recons[i].cpu().numpy().reshape(32, 32), cmap='gray')
        axes[1, i].set_title("Recon")
        axes[1, i].axis('off')
    plt.tight_layout()
    plt.savefig(model_dir / 'recon_stage0.png')
    print("Model and reconstructions saved to models/stage0/")

if __name__ == "__main__":
    main()