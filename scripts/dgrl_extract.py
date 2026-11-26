#!/usr/bin/env python3
"""
CASIA-HWDB2 .dgrl extractor
===========================
Decodes CASIA's page-level .dgrl files into one grayscale PNG per text line,
at original resolution, plus a metadata CSV. No resizing, no colour conversion,
no cropping beyond CASIA's own line boxes.

DGRL layout (per CASIA's format description; verified at runtime, see CHECKS):
  header_size      uint32 LE
  format_code      8 bytes, b"DGR" padded with nulls in HWDB2.0 files
  illustration     header_size - 36 bytes (free text)
  code_type        20 bytes (e.g. "ASCII", "GB")
  code_length      int16 LE  (bytes per character label, normally 2)
  bits_per_pixel   int16 LE  (expected 8)
  page_height      int32 LE
  page_width       int32 LE
  line_count       int32 LE
  for each line:
    char_count     int32 LE
    label          char_count * code_length bytes (GBK)
    top, left      int32 LE each
    height, width  int32 LE each
    bitmap         height * width bytes, uint8 grayscale

CHECKS (the script stops on the first failure):
  - format_code begins with DGR
  - bits_per_pixel == 8
  - every line box has positive size and lies inside the page
  - every byte of the file is consumed, no more and no less

Usage:
  python3 dgrl_extract.py ~/datasets/casia-hwdb2-raw  ~/datasets/casia-hwdb2-lines
  python3 dgrl_extract.py IN OUT --limit 3      # try a few files first
"""
import argparse
import csv
import struct
import sys
from pathlib import Path

import numpy as np
from PIL import Image


class DGRLFormatError(Exception):
    pass


def _read(f, n, what):
    b = f.read(n)
    if len(b) != n:
        raise DGRLFormatError(f"unexpected end of file reading {what} ({len(b)} of {n} bytes)")
    return b


def parse_dgrl(path):
    """Yield one dict per text line. Raises DGRLFormatError on any check failure."""
    size_on_disk = path.stat().st_size
    with open(path, "rb") as f:
        header_size = struct.unpack("<I", _read(f, 4, "header_size"))[0]
        format_code = _read(f, 8, "format_code")
        # Real HWDB2.0 files carry "DGR" padded with nulls, not "DGRL"
        if not format_code.startswith(b"DGR"):
            raise DGRLFormatError(f"format code is {format_code!r}, expected DGR")
        illustration = _read(f, header_size - 36, "illustration")
        code_type = _read(f, 20, "code_type").rstrip(b"\x00").decode("ascii", "replace")
        code_length, bpp = struct.unpack("<hh", _read(f, 4, "code_length/bpp"))
        if bpp != 8:
            raise DGRLFormatError(f"bits_per_pixel is {bpp}, expected 8")

        page_h, page_w, n_lines = struct.unpack("<iii", _read(f, 12, "page record"))

        for line_idx in range(n_lines):
            n_chars = struct.unpack("<i", _read(f, 4, f"line {line_idx} char count"))[0]
            raw_label = _read(f, n_chars * code_length, f"line {line_idx} label")
            chars = []
            for k in range(n_chars):
                code = raw_label[k * code_length:(k + 1) * code_length]
                chars.append(code.decode("gbk", errors="replace"))
            top, left, h, w = struct.unpack("<iiii", _read(f, 16, f"line {line_idx} box"))
            if h <= 0 or w <= 0:
                raise DGRLFormatError(f"line {line_idx} has non-positive size {h}x{w}")
            if top < 0 or left < 0 or top + h > page_h or left + w > page_w:
                raise DGRLFormatError(
                    f"line {line_idx} box ({top},{left},{h},{w}) outside page {page_h}x{page_w}")
            bitmap = np.frombuffer(_read(f, h * w, f"line {line_idx} bitmap"), dtype=np.uint8)
            yield {
                "line_index": line_idx,
                "text": "".join(chars),
                "n_chars": n_chars,
                "replaced_chars": sum(c == "\ufffd" for c in chars),
                "top": top, "left": left, "height": h, "width": w,
                "page_height": page_h, "page_width": page_w,
                "code_type": code_type,
                "image": bitmap.reshape(h, w),
            }

        leftover = f.read()
        consumed = size_on_disk - len(leftover)
        if leftover:
            raise DGRLFormatError(
                f"{len(leftover)} bytes left after {n_lines} lines "
                f"(read {consumed} of {size_on_disk}); layout assumption is wrong")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input_dir", type=Path, help="folder containing .dgrl files (searched recursively)")
    ap.add_argument("output_dir", type=Path)
    ap.add_argument("--limit", type=int, default=None, help="only process the first N files")
    args = ap.parse_args()

    files = sorted(args.input_dir.rglob("*.dgrl"))
    if not files:
        sys.exit(f"No .dgrl files under {args.input_dir}")
    if args.limit:
        files = files[:args.limit]
    print(f"{len(files)} .dgrl files")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "lines.csv"
    fields = ["image_file", "release", "source_dgrl", "page_stem", "line_index", "text",
              "n_chars", "replaced_chars", "top", "left", "height", "width",
              "page_height", "page_width", "code_type"]

    total_lines = total_chars = total_replaced = 0
    failures = []
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for i, path in enumerate(files, 1):
            # e.g. .../HWDB2.0Train/001-P16.dgrl -> release HWDB2.0Train, stem 001-P16
            release = path.parent.name
            stem = path.stem
            out_dir = args.output_dir / release
            out_dir.mkdir(parents=True, exist_ok=True)
            try:
                lines = list(parse_dgrl(path))
            except DGRLFormatError as e:
                failures.append((str(path), str(e)))
                print(f"  FAIL {path.name}: {e}")
                continue
            for ln in lines:
                img_name = f"{stem}_L{ln['line_index']:02d}.png"
                Image.fromarray(ln["image"], mode="L").save(out_dir / img_name)
                row = {k: ln[k] for k in fields if k in ln}
                row.update(image_file=f"{release}/{img_name}", release=release,
                           source_dgrl=str(path), page_stem=stem)
                writer.writerow(row)
                total_lines += 1
                total_chars += ln["n_chars"]
                total_replaced += ln["replaced_chars"]
            if i % 200 == 0 or i == len(files):
                print(f"  {i}/{len(files)} files, {total_lines} lines so far")

    print("\nSummary")
    print(f"  files parsed:      {len(files) - len(failures)} of {len(files)}")
    print(f"  lines written:     {total_lines}")
    print(f"  characters:        {total_chars}")
    print(f"  undecodable chars: {total_replaced}")
    print(f"  metadata:          {csv_path}")
    if failures:
        print(f"\n  {len(failures)} files failed; first: {failures[0]}")
        sys.exit(1)


if __name__ == "__main__":
    main()