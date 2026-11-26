import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path

# Target: image index 122 (1-based)
ISO_DISTRACTOR_DIR = Path('/run/media/phinnm/WSLsave/character_context_project/character_context_project/data/isolated/distractors')

# Find the 122nd image
all_images = []
for char_dir in sorted(ISO_DISTRACTOR_DIR.iterdir()):
    if char_dir.is_dir():
        all_images.extend(sorted(list(char_dir.glob('*.png'))))

target_img = all_images[122]
print(f"Pilot Image: {target_img}")

# Load and resize to 32x32 to match our pipeline
img = Image.open(target_img).convert('L').resize((32, 32), Image.LANCZOS)
arr = np.array(img, dtype=np.float32)

# --- Pure NumPy Distance Transform ---

# Find all dark pixels (ink) and light pixels (paper)
dark_mask = arr < 128

# We need the distance from every pixel to the nearest dark pixel.
# Instead of scipy, we can use a simple grid search.
h, w = arr.shape
dark_coords = np.argwhere(dark_mask)

distances = np.zeros((h, w))

# For each pixel, find the minimum distance to any dark pixel
for y in range(h):
    for x in range(w):
        # Calculate distance to all dark pixels
        if len(dark_coords) > 0:
            diffs = dark_coords - [y, x]
            dists = np.sqrt(np.sum(diffs**2, axis=1))
            distances[y, x] = np.min(dists)
        else:
            distances[y, x] = 0

# --- Scatter Plot ---

# Flatten the arrays to get individual pixels
intensities = arr.flatten()
flat_distances = distances.flatten()

# Create the scatter plot
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Panel 1: The Image
axes[0].imshow(arr, cmap='gray')
axes[0].set_title(f'Pilot Image: {target_img.name}')
axes[0].axis('off')

# Panel 2: The Scatter Plot (Every Pixel)
# We use alpha=0.5 to see overlapping pixels
scatter = axes[1].scatter(flat_distances, intensities, c=intensities, cmap='gray_r', alpha=0.5, s=10)
axes[1].set_title('Pixel Intensity vs. Distance to Nearest Ink')
axes[1].set_xlabel('Distance to nearest dark pixel (pixels)')
axes[1].set_ylabel('Pixel Intensity (0=black, 255=white)')
axes[1].invert_yaxis() # Invert so black ink (0) is at the top, paper (255) at bottom
axes[1].grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig('figures/pilot_decay_122.png', dpi=150)
print("Saved pilot scatter plot to figures/pilot_decay_122.png")
