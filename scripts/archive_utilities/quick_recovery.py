import csv
import os
from pathlib import Path
from PIL import Image

CSV_PATH = Path('ledger/manual_crop_coords.csv')
CROP_DIR = Path('ledger/manual_crops')
LINES_CSV = Path('/home/phinnm/Desktop/capstone-characters/HWDB2.1Train-lines/lines.csv')

# 1. Load what's already in the CSV
existing = set()
with open(CSV_PATH, newline='', encoding='utf-8') as f:
    for row in csv.DictReader(f):
        existing.add(row['image_file'])

# 2. Load the line metadata so we can fill in the text/writer columns
line_data = {}
with open(LINES_CSV, newline='', encoding='utf-8') as f:
    for row in csv.DictReader(f):
        line_data[row['image_file']] = row

# 3. Find missing crops in the folder
missing_rows = []
for img_file in CROP_DIR.glob('*.png'):
    # Reconstruct original image_file name (e.g., HWDB2.1Train_1070-P19_L03.png -> HWDB2.1Train/1070-P19_L03.png)
    raw_name = img_file.name.replace('.png', '')
    parts = raw_name.split('_', 1) # Split on first underscore
    if len(parts) == 2:
        original_path = f"{parts[0]}/{parts[1]}"
    else:
        original_path = raw_name
        
    if original_path not in existing and original_path in line_data:
        row = line_data[original_path]
        # We don't have coords, so we use dummy 0s for x1, y1, x2, y2
        missing_rows.append([original_path, row['page_stem'].split('-')[0], row['text'], 0, 0, 0, 0])

# 4. Append to CSV
if missing_rows:
    with open(CSV_PATH, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        for r in missing_rows:
            writer.writerow(r)
    print(f"Recovered and appended {len(missing_rows)} missing rows!")
else:
    print("No missing rows found.")