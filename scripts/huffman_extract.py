import csv
import heapq
from collections import Counter
from pathlib import Path
import numpy as np
from PIL import Image

# --- Paths ---
ISO_TARGET_DIR = Path('/home/phinnm/Desktop/character_context_project/notebooks/data/isolated/target')
ISO_DISTRACTOR_DIR = Path('/run/media/phinnm/WSLsave/character_context_project/character_context_project/data/isolated/distractors')
CTX_CSV = Path('ledger/manual_crop_coords.csv')
CTX_CROP_DIR = Path('ledger/manual_crops')

OUT_DIR = Path('huffman_extractions')
OUT_DIR.mkdir(parents=True, exist_ok=True)

def load_all_images():
    images = []
    for img_path in ISO_TARGET_DIR.glob('*.png'):
        images.append((img_path, 'isolated', 'target'))
    for char_dir in ISO_DISTRACTOR_DIR.iterdir():
        if char_dir.is_dir():
            char = char_dir.name
            for img_path in char_dir.glob('*.png'):
                images.append((img_path, 'isolated', char))
    with open(CTX_CSV, newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            safe_name = row['image_file'].replace('/', '_').replace('.png', '')
            img_path = CTX_CROP_DIR / f"{safe_name}.png"
            if img_path.exists():
                images.append((img_path, 'contextual', 'target'))
    return images

# --- Huffman Core ---

def build_huffman_tree(frequencies):
    if not frequencies:
        return None
    if len(frequencies) == 1:
        return (next(iter(frequencies)), None, None)
    
    heap = [(freq, i, [sym, None, None]) for i, (sym, freq) in enumerate(frequencies.items())]
    heapq.heapify(heap)
    counter = len(heap)
    
    while len(heap) > 1:
        f1, _, left = heapq.heappop(heap)
        f2, _, right = heapq.heappop(heap)
        heapq.heappush(heap, (f1 + f2, counter, [None, left, right]))
        counter += 1
        
    return heap[0][2]

def build_codes(tree, prefix="", codebook=None):
    if codebook is None:
        codebook = {}
    if tree is None:
        return codebook
    
    sym, left, right = tree
    if sym is not None:
        codebook[sym] = prefix or "0"
    else:
        build_codes(left, prefix + "0", codebook)
        build_codes(right, prefix + "1", codebook)
        
    return codebook

# --- Extraction ---

def process_image(img_path, condition, label):
    # Load image as 32x32 grayscale
    img = Image.open(img_path).convert('L').resize((32, 32), Image.LANCZOS)
    arr = np.array(img, dtype=np.uint8)
    pixels = arr.flatten().tolist()
    
    # Build Huffman code
    freqs = Counter(pixels)
    tree = build_huffman_tree(freqs)
    codes = build_codes(tree)
    
    # Encode to binary string
    bitstring = "".join(codes[p] for p in pixels)
    
    # Verify round-trip (decode)
    reverse_codes = {v: k for k, v in codes.items()}
    decoded_pixels = []
    current_code = ""
    for bit in bitstring:
        current_code += bit
        if current_code in reverse_codes:
            decoded_pixels.append(reverse_codes[current_code])
            current_code = ""
            
    # Check lossless
    if decoded_pixels != pixels:
        print(f"ROUND-TRIP FAILURE: {img_path.name}")
        return
        
    # Save binary payload
    # We prepend a small header with the codebook size so the model can parse it
    # For now, we just save the raw bits as bytes.
    # To handle bit-level writing cleanly, we pad the string to a multiple of 8
    padding = (8 - len(bitstring) % 8) % 8
    padded_bitstring = bitstring + "0" * padding
    
    bytes_out = bytearray()
    for i in range(0, len(padded_bitstring), 8):
        byte_val = int(padded_bitstring[i:i+8], 2)
        bytes_out.append(byte_val)
        
    bin_name = f"{condition}_{label}_{img_path.stem}.bin"
    with open(OUT_DIR / bin_name, "wb") as f:
        f.write(bytes_out)
        
    # We also need to save the Huffman codebook (dictionary) so the model can decode
    # This is the "key" to the compression.
    import json
    dict_name = f"{condition}_{label}_{img_path.stem}_dict.json"
    with open(OUT_DIR / dict_name, "w", encoding="utf-8") as f:
        json.dump(codes, f)

def main():
    all_images = load_all_images()
    print(f"Starting Huffman extraction for {len(all_images)} images...")
    
    for i, (img_path, condition, label) in enumerate(all_images, 1):
        process_image(img_path, condition, label)
        if i % 50 == 0:
            print(f"  Processed {i}/{len(all_images)}")
            
    print("Done. Extraction and round-trip verification complete.")

if __name__ == "__main__":
    main()