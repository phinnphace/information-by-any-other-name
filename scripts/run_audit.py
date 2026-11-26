"""
Generates crops for the manual audit queue.
Reads ledger/ctx_audit_queue.csv, finds the source line image,
applies the proportional division math, and saves the crop.
"""
import csv
import os
from pathlib import Path
from PIL import Image

# --- Configuration ---
LINES_CSV = Path('/home/phinnm/Desktop/capstone-characters/HWDB2.1Train-lines/lines.csv')
LINE_IMG_DIR = Path('/home/phinnm/Desktop/capstone-characters/HWDB2.1Train-lines/HWDB2.1Train')
QUEUE_CSV = Path('ledger/ctx_audit_queue.csv')
CROP_DIR = Path('ledger/ctx_crops_to_audit')
TARGET_SIZE = (100, 100)

def main():
    CROP_DIR.mkdir(parents=True, exist_ok=True)
    
    # 1. Load lines.csv into a dict for fast lookup
    print("Loading lines.csv...")
    line_data = {}
    with open(LINES_CSV, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            line_data[row['image_file']] = row
            
    # 2. Process the queue
    print("Generating crops...")
    updated_queue = []
    
    with open(QUEUE_CSV, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            image_file = row['image_file']
            text = row['full_text']
            pos = int(row['target_pos'])
            
            # Get line geometry
            line = line_data.get(image_file)
            if not line:
                print(f"Warning: {image_file} not found in lines.csv")
                continue
                
            img_width = int(line['width'])
            img_height = int(line['height'])
            n_chars = len(text)
            
            # Proportional division math (B.3)
            char_width = img_width / n_chars
            start_pos = max(0, pos - 1)
            end_pos = min(n_chars, pos + 2)
            
            x_start = int(start_pos * char_width)
            x_end = int(end_pos * char_width)
            
            # Load the line image
            img_path = LINE_IMG_DIR / image_file
            if not img_path.exists():
                # Sometimes the CSV path has a subfolder prefix
                img_path = LINE_IMG_DIR / image_file.split('/')[-1]
                
            if not img_path.exists():
                print(f"Warning: Image not found {img_path}")
                continue
                
            img = Image.open(img_path)
            
            # Crop and resize
            crop = img.crop((x_start, 0, x_end, img_height))
            crop_resized = crop.resize(TARGET_SIZE, Image.LANCZOS)
            
            # Save crop with a predictable name for the audit
            out_name = f"audit_{i:03d}_{image_file.replace('/', '_')}_{pos}.png"
            out_path = CROP_DIR / out_name
            crop_resized.save(out_path)
            
            # Prepare row for the audit sheet
            row['audit_image'] = str(out_path)
            row['status'] = '' # Blank for human to fill
            updated_queue.append(row)
            
            if (i + 1) % 50 == 0:
                print(f"  Processed {i + 1} crops...")

    # 3. Write the updated queue with the image paths
    out_csv = Path('ledger/ctx_audit_sheet.csv')
    with open(out_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=updated_queue[0].keys())
        writer.writeheader()
        writer.writerows(updated_queue)
        
    print(f"\nDone. {len(updated_queue)} crops saved to {CROP_DIR}")
    print(f"Audit sheet: {out_csv}")
    print("Open the sheet, review the images, and fill the 'status' column with:")
    print("  pass, clipped_edge, neighbor_intrusion, diagonal_baseline, touching_strokes, line_box_error")

if __name__ == "__main__":
    main()