import csv
from pathlib import Path

CSV_PATH = Path('ledger/manual_crop_coords.csv')
CROP_DIR = Path('ledger/manual_crops')

# 1. Load the expected safe_names from the CSV
expected_names = set()
with open(CSV_PATH, newline='', encoding='utf-8') as f:
    for row in csv.DictReader(f):
        safe_name = row['image_file'].replace('/', '_').replace('.png', '')
        expected_names.add(safe_name)

# 2. Scan the folder for old names and rename them
renamed = 0
for img_path in CROP_DIR.glob('*.png'):
    if img_path.stem in expected_names:
        continue # Already correct
        
    # Old format: 1016-P16_10.png -> parts = ['1016-P16', '10']
    if '_' in img_path.stem:
        parts = img_path.stem.split('_')
        if len(parts) == 2 and '-' in parts[0]:
            stem, line = parts
            # Reconstruct the expected safe_name: HWDB2.1Train_1016-P16_L10
            # Note: the CSV uses L02, L10 (zero-padded to 2 digits)
            expected_stem = f"HWDB2.1Train_{stem}_L{int(line):02d}"
            
            if expected_stem in expected_names:
                new_path = CROP_DIR / f"{expected_stem}.png"
                img_path.rename(new_path)
                print(f"Renamed {img_path.name} -> {new_path.name}")
                renamed += 1

print(f"\nFinished. Renamed {renamed} files.")