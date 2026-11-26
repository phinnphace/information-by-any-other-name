import csv
import random
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt

QUEUE_CSV = Path('ledger/ctx_audit_sheet.csv')
LINE_IMG_DIR = Path('/home/phinnm/Desktop/capstone-characters/HWDB2.1Train-lines')
OUT_FIG = Path('figures/raw_calligraphy_sample.png')

OUT_FIG.parent.mkdir(parents=True, exist_ok=True)

def main():
    rows = []
    with open(QUEUE_CSV, newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            rows.append(row)

    if not rows:
        print("No rows found.")
        return

    # Randomly sample 6 lines
    random.seed(42) # Reproducible random sample
    samples = random.sample(rows, min(6, len(rows)))

    # Grid layout: 6 rows, 1 column
    fig, axes = plt.subplots(len(samples), 1, figsize=(12, 2 * len(samples)))
    
    if len(samples) == 1:
        axes = [axes]

    for i, row in enumerate(samples):
        img_path = LINE_IMG_DIR / row['image_file']
        
        if img_path.exists():
            img = Image.open(img_path)
            axes[i].imshow(img, cmap='gray')
        else:
            axes[i].text(0.5, 0.5, "Image Missing", ha='center', va='center')
            
        axes[i].axis('off')  # No axes, no titles, just the image

    plt.tight_layout()
    plt.savefig(OUT_FIG, dpi=300, bbox_inches='tight', pad_inches=0.1)
    print(f"Figure saved to {OUT_FIG}")

if __name__ == "__main__":
    main()