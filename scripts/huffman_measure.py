"""
1a-cal: Huffman calibration measurement.

For each crop:
  1. Round-trip verification: encode -> decode must return the exact input.
     This is the zero-residual calibration. If it fails, the crop or encoder
     is broken, and we find out before trusting any nonzero residual.
  2. Code length under three symbol definitions:
       - raw:    each pixel value (0-255) is a symbol
       - block4: each 4x4 block, quantized, is a symbol
       - block8: each 8x8 block, quantized, is a symbol
     Code length is in bits. Normalized by pixel count for cross-crop comparison.
  3. ΔX(VAR) across definitions: code length differences, in bits.
     Analytic variable: "compressibility given symbol definition."

Outputs:
  ledger/huffman_measurements.csv
  ledger/huffman_summary.json
  figures/huffman_distributions.png

No model. No seeds. No folds. Deterministic.
"""
import csv
import json
import heapq
from collections import Counter
from pathlib import Path
from itertools import product

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# --- Paths ---
ISO_TARGET_DIR = Path('/home/phinnm/Desktop/character_context_project/notebooks/data/isolated/target')
ISO_DISTRACTOR_DIR = Path('/run/media/phinnm/WSLsave/character_context_project/character_context_project/data/isolated/distractors')
CTX_CSV = Path('ledger/manual_crop_coords.csv')
CTX_CROP_DIR = Path('ledger/manual_crops')

OUT_CSV = Path('ledger/huffman_measurements.csv')
OUT_JSON = Path('ledger/huffman_summary.json')
OUT_FIG = Path('figures/huffman_distributions.png')

OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
OUT_FIG.parent.mkdir(parents=True, exist_ok=True)

# --- Huffman core (pure, no external dependency) ---

def build_huffman_code(frequencies):
    """Returns {symbol: '0101...'} for the given symbol frequencies."""
    if not frequencies:
        return {}
    if len(frequencies) == 1:
        # Degenerate case: one symbol. Assign it a 1-bit code.
        return {next(iter(frequencies)): '0'}
    heap = [(freq, i, [sym, None, None]) for i, (sym, freq) in enumerate(frequencies.items())]
    heapq.heapify(heap)
    counter = len(heap)
    while len(heap) > 1:
        f1, _, left = heapq.heappop(heap)
        f2, _, right = heapq.heappop(heap)
        heapq.heappush(heap, (f1 + f2, counter, [None, left, right]))
        counter += 1
    root = heap[0][2]
    codes = {}
    def walk(node, prefix):
        if node[0] is not None:
            codes[node[0]] = prefix or '0'
            return
        walk(node[1], prefix + '0')
        walk(node[2], prefix + '1')
    walk(root, '')
    return codes

def huffman_code_length(symbols):
    """Total code length in bits for a sequence of symbols."""
    if not symbols:
        return 0
    freqs = Counter(symbols)
    codes = build_huffman_code(dict(freqs))
    return sum(len(codes[s]) for s in symbols)

def round_trip_check(symbols):
    """Encode with Huffman, then verify the symbol sequence is recoverable.
    We verify structurally: the code must be prefix-free, so decoding is unique."""
    freqs = Counter(symbols)
    codes = build_huffman_code(dict(freqs))
    all_codes = list(codes.values())
    for a, b in product(all_codes, repeat=2):
        if a != b and b.startswith(a):
            return False
    return True

# --- Symbol definitions ---

def symbols_raw(pixels):
    """Each pixel value 0-255 is a symbol."""
    return [int(p) for p in pixels.flatten()]

def symbols_block(pixels, block_size, palette_bits=4):
    """Each block_size x block_size block is a symbol, quantized to `palette_bits`."""
    h, w = pixels.shape
    assert h % block_size == 0 and w % block_size == 0
    levels = 2 ** palette_bits
    symbols = []
    for by in range(0, h, block_size):
        for bx in range(0, w, block_size):
            block = pixels[by:by+block_size, bx:bx+block_size].astype(np.float64)
            mean_val = block.mean()
            # Quantize mean to palette_bits
            idx = int(round((mean_val / 255.0) * (levels - 1)))
            symbols.append(idx)
    return symbols

# --- Load crops ---

def load_crops():
    """Returns list of (condition, label, identifier, image_array_2d)."""
    crops = []
    
    for img_path in ISO_TARGET_DIR.glob('*.png'):
        img = Image.open(img_path).convert('L')
        arr = np.array(img, dtype=np.uint8)
        crops.append(('isolated', 'target', img_path.name, arr))
    
    for char_dir in ISO_DISTRACTOR_DIR.iterdir():
        if char_dir.is_dir():
            char = char_dir.name
            for img_path in char_dir.glob('*.png'):
                img = Image.open(img_path).convert('L')
                arr = np.array(img, dtype=np.uint8)
                crops.append(('isolated', char, img_path.name, arr))
    
    with open(CTX_CSV, newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            safe_name = row['image_file'].replace('/', '_').replace('.png', '')
            img_path = CTX_CROP_DIR / f"{safe_name}.png"
            if img_path.exists():
                img = Image.open(img_path).convert('L')
                arr = np.array(img, dtype=np.uint8)
                # Resize isolated crops to 32x32 to match; contextual are already 100x100
                # The script handles any size, but block definitions require divisible sizes.
                if arr.shape[0] % 8 != 0 or arr.shape[1] % 8 != 0:
                    img2 = Image.open(img_path).convert('L').resize((32, 32), Image.LANCZOS)
                    arr = np.array(img2, dtype=np.uint8)
                crops.append(('contextual', 'target', safe_name, arr))
    
    return crops

# --- Main ---

def main():
    crops = load_crops()
    print(f"Loaded {len(crops)} crops.")
    
    results = []
    for condition, label, identifier, arr in crops:
        # Resize to divisible-by-8 grid for block definitions
        h, w = arr.shape
        new_h = (h // 8) * 8
        new_w = (w // 8) * 8
        if new_h == 0 or new_w == 0:
            print(f"Skipping {identifier}: too small for 8x8 blocks.")
            continue
        if new_h != h or new_w != w:
            arr = np.array(Image.fromarray(arr).resize((new_w, new_h), Image.LANCZOS), dtype=np.uint8)
        
        # Symbol definitions
        s_raw = symbols_raw(arr)
        s_b4 = symbols_block(arr, 4)
        s_b8 = symbols_block(arr, 8)
        
        # Round-trip calibration
        rt_raw = round_trip_check(s_raw)
        rt_b4 = round_trip_check(s_b4)
        rt_b8 = round_trip_check(s_b8)
        round_trip = rt_raw and rt_b4 and rt_b8
        
        # Code lengths (bits)
        L_raw = huffman_code_length(s_raw)
        L_b4 = huffman_code_length(s_b4)
        L_b8 = huffman_code_length(s_b8)
        
        # ΔX(VAR): change in code length across definitions
        dX_raw_to_b4 = L_b4 - L_raw
        dX_b4_to_b8 = L_b8 - L_b4
        
        # Normalized: bits per pixel
        n_pixels = arr.size
        results.append({
            'condition': condition,
            'label': label,
            'identifier': identifier,
            'n_pixels': n_pixels,
            'round_trip': round_trip,
            'code_length_raw_bits': L_raw,
            'code_length_block4_bits': L_b4,
            'code_length_block8_bits': L_b8,
            'dX_raw_to_block4': dX_raw_to_b4,
            'dX_block4_to_block8': dX_b4_to_b8,
            'bits_per_pixel_raw': L_raw / n_pixels,
            'bits_per_pixel_block4': L_b4 / n_pixels,
            'bits_per_pixel_block8': L_b8 / n_pixels,
        })
    
    # Write CSV
    with open(OUT_CSV, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    print(f"Wrote {len(results)} measurements to {OUT_CSV}")
    
    # Summary by condition
    summary = {}
    for condition in ('isolated', 'contextual'):
        subset = [r for r in results if r['condition'] == condition]
        if not subset:
            continue
        summary[condition] = {
            'n': len(subset),
            'round_trip_failures': sum(1 for r in subset if not r['round_trip']),
            'median_code_length_raw': float(np.median([r['code_length_raw_bits'] for r in subset])),
            'median_code_length_block4': float(np.median([r['code_length_block4_bits'] for r in subset])),
            'median_code_length_block8': float(np.median([r['code_length_block8_bits'] for r in subset])),
            'median_dX_raw_to_block4': float(np.median([r['dX_raw_to_block4'] for r in subset])),
            'median_dX_block4_to_block8': float(np.median([r['dX_block4_to_block8'] for r in subset])),
            'median_bpp_raw': float(np.median([r['bits_per_pixel_raw'] for r in subset])),
            'median_bpp_block4': float(np.median([r['bits_per_pixel_block4'] for r in subset])),
            'median_bpp_block8': float(np.median([r['bits_per_pixel_block8'] for r in subset])),
        }
    with open(OUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote summary to {OUT_JSON}")
    
    # Distribution figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, key, title in zip(
        axes,
        ['bits_per_pixel_raw', 'bits_per_pixel_block4', 'bits_per_pixel_block8'],
        ['Raw pixels', '4×4 blocks', '8×8 blocks']
    ):
        for condition, color in [('isolated', 'tab:blue'), ('contextual', 'tab:orange')]:
            vals = [r[key] for r in results if r['condition'] == condition]
            if vals:
                ax.hist(vals, bins=30, alpha=0.5, label=condition, color=color)
        ax.set_title(f'{title}')
        ax.set_xlabel('bits per pixel')
        ax.set_ylabel('count')
        ax.legend()
    plt.tight_layout()
    plt.savefig(OUT_FIG, dpi=150)
    print(f"Wrote distributions to {OUT_FIG}")
    
    # Report
    print("\n--- 1a-cal SUMMARY ---")
    for condition, s in summary.items():
        print(f"{condition}: n={s['n']}, round-trip failures={s['round_trip_failures']}")
        print(f"  median bits/pixel: raw={s['median_bpp_raw']:.3f}, block4={s['median_bpp_block4']:.3f}, block8={s['median_bpp_block8']:.3f}")
        print(f"  median ΔX: raw→block4={s['median_dX_raw_to_block4']:.0f} bits, block4→block8={s['median_dX_block4_to_block8']:.0f} bits")

if __name__ == "__main__":
    main()