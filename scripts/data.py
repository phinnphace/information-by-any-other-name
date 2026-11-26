"""
Data layer for the Gaussian-Bernoulli RBM.

Builds the ISO and CTX conditions from raw CASIA data.
- ISO: HWDB1.1 GNT files (parsed via struct).
- CTX: HWDB2.1 lines (parsed via lines.csv), cropped via recorded
  proportional division.

Enforces writer-grouped 5-fold splits.
Generates the audit queue for proportional crop failures.
"""
import os
import csv
import struct
import numpy as np
import torch
from PIL import Image
from pathlib import Path

# --- Configuration ---
TARGET_CHAR = '打'
DISTRACTORS = ['扔', '扛', '扫', '托', '扣']
GNT_DIR = Path('/home/phinnm/Desktop/capstone-characters/HWDB1.1trn_gnt')
LINES_CSV = Path('/home/phinnm/Desktop/capstone-characters/HWDB2.1Train-lines/lines.csv')
LINE_IMG_DIR = Path('/home/phinnm/Desktop/capstone-characters/HWDB2.1Train-lines/HWDB2.1Train/HWDB2.1Train')

TARGET_SIZE = (32, 32)  # Ledger step 8
N_FOLDS = 5
AUDIT_REASON_CODES = [
    "pass", "clipped_edge", "neighbor_intrusion",
    "diagonal_baseline", "touching_strokes", "line_box_error"
]

# ============================================================
# ISO CONDITION
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

def extract_writer_from_gnt(filename):
    # e.g., "1001-c.gnt" -> "1001"
    return filename.split('-')[0]

def load_iso_data():
    """Returns list of dicts: {'image': 32x32 float tensor, 'writer': str, 'char': str}"""
    data = []
    char_set = [TARGET_CHAR] + DISTRACTORS
    gnt_files = sorted(GNT_DIR.glob('*.gnt'))
    
    for gnt_path in gnt_files:
        writer = extract_writer_from_gnt(gnt_path.name)
        samples = parse_gnt_file(gnt_path)
        for img, label in samples:
            if label in char_set:
                # Resize to 32x32 (Ledger step 8)
                img = img.resize(TARGET_SIZE, Image.LANCZOS)
                arr = np.array(img, dtype=np.float32).flatten() / 255.0
                # TODO: Apply median/MAD scaling here (Ledger step 10)
                data.append({'image': arr, 'writer': writer, 'char': label})
    return data

# ============================================================
# CTX CONDITION
# ============================================================

def get_ctx_crops():
    """
    Reads lines.csv, generates proportional crops, and returns a queue for audit.
    Does NOT return images yet; returns crop metadata and the line image path.
    """
    crops_to_audit = []
    
    with open(LINES_CSV, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            text = row['text']
            if TARGET_CHAR in text:
                # B.3: Two-character crops (target + neighbour)
                # We generate the crop box using proportional division.
                # The actual pixel crop happens during the audit/save phase.
                for pos in [i for i, ch in enumerate(text) if ch == TARGET_CHAR]:
                    start_pos = max(0, pos - 1)
                    end_pos = min(len(text), pos + 2)
                    
                    crops_to_audit.append({
                        'image_file': row['image_file'],
                        'full_text': text,
                        'target_pos': pos,
                        'crop_slice': text[start_pos:end_pos],
                        'writer': row['page_stem'].split('-')[0],
                        'status': 'PENDING_AUDIT'
                    })
    return crops_to_audit

def generate_audit_queue():
    """Writes the audit queue to disk for manual review."""
    crops = get_ctx_crops()
    out_path = Path('ledger/ctx_audit_queue.csv')
    out_path.parent.mkdir(exist_ok=True)
    
    with open(out_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['image_file', 'full_text', 'target_pos', 'crop_slice', 'writer', 'status'])
        writer.writeheader()
        writer.writerows(crops)
    print(f"Wrote {len(crops)} crops to {out_path} for audit.")

# ============================================================
# FOLDS
# ============================================================

def assign_folds(data_items, seed=42):
    """
    Assigns writer-grouped 5-folds. 
    data_items: list of dicts with 'writer' key.
    """
    rng = np.random.default_rng(seed)
    unique_writers = sorted(list(set(item['writer'] for item in data_items)))
    rng.shuffle(unique_writers)
    
    fold_map = {}
    for i, w in enumerate(unique_writers):
        fold_map[w] = i % N_FOLDS
        
    for item in data_items:
        item['fold'] = fold_map[item['writer']]
    return data_items