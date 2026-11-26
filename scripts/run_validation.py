import csv
import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path

# --- Paths ---
ISO_TARGET_DIR = Path('/home/phinnm/Desktop/character_context_project/notebooks/data/isolated/target')
ISO_DISTRACTOR_DIR = Path('/run/media/phinnm/WSLsave/character_context_project/character_context_project/data/isolated/distractors')
CTX_CSV = Path('ledger/manual_crop_coords.csv')
CTX_CROP_DIR = Path('ledger/manual_crops')

# Output folder
OUT_DIR = Path('internal_validation_output')
OUT_DIR.mkdir(parents=True, exist_ok=True)

def load_all_images():
    """Loads the full pool of 644 images."""
    images = []
    
    # 1. ISO Target
    for img_path in ISO_TARGET_DIR.glob('*.png'):
        images.append((img_path, 'isolated', 'target'))
        
    # 2. ISO Distractors
    for char_dir in ISO_DISTRACTOR_DIR.iterdir():
        if char_dir.is_dir():
            char = char_dir.name
            for img_path in char_dir.glob('*.png'):
                images.append((img_path, 'isolated', char))
                
    # 3. Contextual
    with open(CTX_CSV, newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            safe_name = row['image_file'].replace('/', '_').replace('.png', '')
            img_path = CTX_CROP_DIR / f"{safe_name}.png"
            if img_path.exists():
                images.append((img_path, 'contextual', 'target'))
                
    return images

def generate_plot(img_path, condition, label):
    """Generates the paired plot for a single image."""
    # Load and resize to 32x32 to match the pipeline
    img = Image.open(img_path).convert('L').resize((32, 32), Image.LANCZOS)
    arr = np.array(img, dtype=np.float32)
    
    # Distance transform (Pure NumPy)
    dark_mask = arr < 128
    h, w = arr.shape
    dark_coords = np.argwhere(dark_mask)
    distances = np.zeros((h, w))
    
    for y in range(h):
        for x in range(w):
            if len(dark_coords) > 0:
                diffs = dark_coords - [y, x]
                dists = np.sqrt(np.sum(diffs**2, axis=1))
                distances[y, x] = np.min(dists)
            else:
                distances[y, x] = 0
                
    # Flatten for scatter
    intensities = arr.flatten()
    flat_distances = distances.flatten()
    
    # Create the figure
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Panel 1: The Image
    axes[0].imshow(arr, cmap='gray')
    axes[0].set_title(f'{condition} | {label}\n{img_path.name}')
    axes[0].axis('off')
    
    # Panel 2: The Scatter Plot
    axes[1].scatter(flat_distances, intensities, c=intensities, cmap='gray_r', alpha=0.5, s=10)
    axes[1].set_title('Pixel Intensity vs. Distance to Nearest Ink')
    axes[1].set_xlabel('Distance to nearest dark pixel (pixels)')
    axes[1].set_ylabel('Pixel Intensity (0=black, 255=white)')
    axes[1].invert_yaxis()
    axes[1].grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    
    # Save the plot
    out_name = f"{condition}_{label}_{img_path.stem}_plot.png"
    plt.savefig(OUT_DIR / out_name, dpi=150)
    plt.close(fig) # Close to free memory
    print(f"Saved {out_name}")

def main():
    all_images = load_all_images()
    print(f"Loaded {len(all_images)} images total.")
    
    # Randomly sample 5 images for the pilot
    random.seed(42)
    sample = random.sample(all_images, 5)
    
    print(f"Generating plots for {len(sample)} images...")
    for img_path, condition, label in sample:
        generate_plot(img_path, condition, label)
        
    print(f"\nDone. Check the {OUT_DIR}/ folder.")

if __name__ == "__main__":
    main()