import csv
from pathlib import Path

LEDGER_PATH = Path('ledger/seeds.csv')

def pull_batch(batch_num):
    """Returns the 20 seeds for a given batch number."""
    seeds = []
    with open(LEDGER_PATH, newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            if int(row['batch']) == batch_num:
                seeds.append(int(row['seed']))
    if len(seeds) != 20:
        raise ValueError(f"Batch {batch_num} does not have exactly 20 seeds.")
    return seeds

def assign(seed, use_label):
    """Updates the `use` column for a specific seed."""
    rows = []
    with open(LEDGER_PATH, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        for row in reader:
            if int(row['seed']) == seed:
                if row['use']:
                    print(f"Warning: Seed {seed} already assigned to {row['use']}. Overwriting with {use_label}.")
                row['use'] = use_label
            rows.append(row)
            
    with open(LEDGER_PATH, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

def write_seeds_used(run_dir, batch_num, seeds, labels):
    """Writes a json log of the seeds used in this run to the models folder."""
    import json
    run_dir.mkdir(parents=True, exist_ok=True)
    data = {
        "batch": batch_num,
        "seeds": seeds,
        "labels": labels
    }
    with open(run_dir / "seeds_used.json", 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4)