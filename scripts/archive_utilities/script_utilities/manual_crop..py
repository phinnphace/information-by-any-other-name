import csv
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.widgets import RectangleSelector
from PIL import Image
from pathlib import Path

# --- Configuration ---
LINES_CSV = Path('/home/phinnm/Desktop/capstone-characters/HWDB2.1Train-lines/lines.csv')
LINE_IMG_DIR = Path('/home/phinnm/Desktop/capstone-characters/HWDB2.1Train-lines/HWDB2.1Train')
OUT_DIR = Path('ledger/manual_crops')
OUT_CSV = Path('ledger/manual_crop_coords.csv')
TARGET_CHAR = '打'

OUT_DIR.mkdir(parents=True, exist_ok=True)

def main():
    # Load CSV to find lines with 打
    lines_to_crop = []
    with open(LINES_CSV, newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            if TARGET_CHAR in row['text']:
                lines_to_crop.append(row)

    print(f"Found {len(lines_to_crop)} lines containing {TARGET_CHAR}.")
    
    # If you need to resume later, check what's already done
    done = set()
    if OUT_CSV.exists():
        with open(OUT_CSV, newline='', encoding='utf-8') as f:
            for row in csv.DictReader(f):
                done.add(row['image_file'])
        print(f"Resuming... {len(done)} already cropped.")

    # Prepare CSV
    if not OUT_CSV.exists():
        with open(OUT_CSV, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['image_file', 'writer', 'text', 'x1', 'y1', 'x2', 'y2'])

    # Setup interactive plot
    fig, ax = plt.subplots(figsize=(10, 6))
    plt.subplots_adjust(bottom=0.2)
    
    state = {'x1': 0, 'y1': 0, 'x2': 0, 'y2': 0, 'index': 0}
    
    def onselect(eclick, erelease):
        state['x1'], state['y1'] = int(eclick.xdata), int(eclick.ydata)
        state['x2'], state['y2'] = int(erelease.xdata), int(erelease.ydata)
        print(f"Selected: ({state['x1']}, {state['y1']}) to ({state['x2']}, {state['y2']})")

    def next_image(event=None):
        # Save the current selection if it exists
        if state['x2'] > state['x1'] and state['y2'] > state['y1']:
            row = lines_to_crop[state['index']]
            img_path = LINE_IMG_DIR / row['image_file']
            
            if img_path.exists():
                img = Image.open(img_path)
                # Crop and save
                crop = img.crop((state['x1'], state['y1'], state['x2'], state['y2']))
                crop_resized = crop.resize((100, 100), Image.LANCZOS)
                out_name = f"{row['page_stem']}_{row['line_index']}.png"
                crop_resized.save(OUT_DIR / out_name)
                
                # Log coords
                with open(OUT_CSV, 'a', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow([row['image_file'], row['page_stem'].split('-')[0], row['text'], state['x1'], state['y1'], state['x2'], state['y2']])
                print(f"Saved crop for {out_name}")

        # Move to next
        state['index'] += 1
        state['x1'], state['y1'], state['x2'], state['y2'] = 0, 0, 0, 0
        
        if state['index'] >= len(lines_to_crop):
            print("Finished all images!")
            plt.close()
            return

        # Load next image
        row = lines_to_crop[state['index']]
        img_path = LINE_IMG_DIR / row['image_file']
        if not img_path.exists():
            print(f"Missing image: {img_path}")
            next_image() # skip
            return
            
        img = Image.open(img_path)
        ax.clear()
        ax.imshow(img, cmap='gray')
        ax.set_title(f"Image {state['index']+1}/{len(lines_to_crop)}: Draw a box around {TARGET_CHAR} + neighbor\n(Press 'Space' to save and go to next)")
        fig.canvas.draw()

    # Setup selector
    selector = RectangleSelector(ax, onselect, useblit=True, button=[1], minspanx=5, minspany=5, spancoords='pixels', interactive=True)
    
    # Connect spacebar
    fig.canvas.mpl_connect('key_press_event', lambda event: next_image() if event.key == ' ' else None)

    # Load first image
    if state['index'] < len(lines_to_crop):
        row = lines_to_crop[state['index']]
        img_path = LINE_IMG_DIR / row['image_file']
        if img_path.exists():
            ax.imshow(Image.open(img_path), cmap='gray')
            ax.set_title(f"Image {state['index']+1}/{len(lines_to_crop)}: Draw a box around {TARGET_CHAR} + neighbor\n(Press 'Space' to save and go to next)")
        else:
            print(f"First image missing: {img_path}")
    
    plt.show()

if __name__ == "__main__":
    main()