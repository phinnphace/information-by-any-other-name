#!/usr/bin/env python3
"""
Build ledger/seeds.csv from the raw random.org TRG draw.

Raw input: tab-separated, no headers, one seed per cell.
  20 columns x 50 rows  ->  column = batch (1..50), row = index (1..20)
  (or 50 columns x 20 rows if --transpose is passed)

Checks (refuse to emit on any failure):
  - exactly 1000 integer cells
  - every seed in [1, 100000000]
  - no duplicates
  - no empty cells, no non-integer tokens

Output schema:
  batch, draw_date, draw_settings, index, seed, use
  `use` is empty at build time.
"""
import argparse
import csv
import sys
from pathlib import Path


def parse_cells(path: Path, transpose: bool):
    rows = []
    for ln_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        cells = line.split(",")
        if any(not c.strip() for c in cells):
            sys.exit(f"empty cell on line {ln_no}")
        rows.append([c.strip() for c in cells])
    widths = {len(r) for r in rows}
    if len(widths) != 1:
        sys.exit(f"ragged rows, widths seen: {widths}")
    w = widths.pop()
    # orient to (n_batches=50, n_per_batch=20)
    n_cells = len(rows) * w
    if n_cells != 1000:
        sys.exit(f"expected 1000 cells, got {n_cells} ({len(rows)}x{w})")
    if w == 20 and len(rows) == 50 and not transpose:
        grid = rows
        n_batches, per_batch = len(rows), w  # 50 batches, 20 per batch
        get = lambda b, i: grid[b][i]        # batch = row, index = col
    elif w == 50 and len(rows) == 20 and not transpose:
        grid = rows
        n_batches, per_batch = w, len(rows)  # 20 batches, 50 per batch
        get = lambda b, i: grid[i][b]        # batch = col, index = row
    elif transpose:
        # caller says the file is transposed relative to the default
        grid = rows
        if w == 50 and len(rows) == 20:
            n_batches, per_batch = len(rows), w
            get = lambda b, i: grid[b][i]
        elif w == 20 and len(rows) == 50:
            n_batches, per_batch = w, len(rows)
            get = lambda b, i: grid[i][b]
        else:
            sys.exit(f"--transpose: unexpected shape {len(rows)}x{w}")
    else:
        sys.exit(f"unexpected shape {len(rows)}x{w}; expected 50x20 or 20x50")
    return n_batches, per_batch, get


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("raw_csv", type=Path)
    ap.add_argument("out_csv", type=Path, default=Path("ledger/seeds.csv"))
    ap.add_argument("--transpose", action="store_true")
    ap.add_argument("--draw-date", required=True)
    ap.add_argument("--draw-settings",
                    default="random.org Integer Generator; TRG; "
                            "num=1000; min=1; max=100000000; col=20; base=10")
    args = ap.parse_args()

    n_batches, per_batch, get = parse_cells(args.raw_csv, args.transpose)
    seeds = [int(get(b, i)) for b in range(n_batches) for i in range(per_batch)]

    if len(seeds) != 1000:
        sys.exit(f"expected 1000 seeds, got {len(seeds)}")
    bad = [s for s in seeds if not (1 <= s <= 100_000_000)]
    if bad:
        sys.exit(f"{len(bad)} seeds out of range [1, 1e8]; first: {bad[0]}")
    if len(set(seeds)) != len(seeds):
        from collections import Counter
        dupes = [s for s, n in Counter(seeds).items() if n > 1]
        sys.exit(f"duplicate seeds: {dupes[:10]} (total {len(dupes)})")

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["batch", "draw_date", "draw_settings", "index", "seed", "use"])
        for b in range(n_batches):
            for i in range(per_batch):
                w.writerow([b + 1, args.draw_date, args.draw_settings, i + 1, get(b, i), ""])
    print(f"wrote {args.out_csv}: {n_batches} batches x {per_batch} = {len(seeds)} seeds")


if __name__ == "__main__":
    main()