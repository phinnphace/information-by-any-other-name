python -c "
import csv
from pathlib import Path

CTX_CSV = Path('ledger/manual_crop_coords.csv')
CTX_CROP_DIR = Path('ledger/manual_crops')

missing = []
found = 0

with open(CTX_CSV, newline='', encoding='utf-8') as f:
    for row in csv.DictReader(f):
        safe_name = row['image_file'].replace('/', '_').replace('.png', '')
        expected = CTX_CROP_DIR / f'{safe_name}.png'
        if expected.exists():
            found += 1
        else:
            missing.append(safe_name)

print(f'Found: {found}')
print(f'Missing: {len(missing)}')
if missing:
    print('First 5 missing:')
    for m in missing[:5]:
        print(f'  {m}.png')
"