#!/usr/bin/env python3
"""
Convert old YOLO detection labels (axis-aligned boxes) to YOLOv8 OBB labels.

OLD format per line:  class  x_center  y_center  width  height        (all normalized 0-1)
NEW format per line:  class  x1 y1 x2 y2 x3 y3 x4 y4                   (4 corner points, normalized)

Corner order is clockwise starting from the top-left:
    (x1,y1) top-left -> (x2,y2) top-right -> (x3,y3) bottom-right -> (x4,y4) bottom-left

NOTE: Because the source boxes are axis-aligned, every converted box has an angle of 0 degrees.
This changes the file FORMAT only. It does not add real rotation information.
"""

import argparse
from pathlib import Path


def clamp(v, lo=0.0, hi=1.0):
    return max(lo, min(hi, v))


def convert_line(line):
    """Convert a single label line. Returns the new line, or None to skip."""
    parts = line.split()
    if not parts:
        return None  # blank line

    # Already in OBB format (class + 8 coords) -> leave as-is.
    if len(parts) == 9:
        return line.strip()

    if len(parts) != 5:
        raise ValueError(f"Expected 5 values (class xc yc w h), got {len(parts)}: {line!r}")

    cls = parts[0]
    xc, yc, w, h = map(float, parts[1:5])
    hw, hh = w / 2.0, h / 2.0

    corners = [
        (xc - hw, yc - hh),  # top-left
        (xc + hw, yc - hh),  # top-right
        (xc + hw, yc + hh),  # bottom-right
        (xc - hw, yc + hh),  # bottom-left
    ]

    coords = " ".join(f"{clamp(x):.6f} {clamp(y):.6f}" for x, y in corners)
    return f"{cls} {coords}"


def convert_file(src_path, dst_path):
    """Returns number of boxes converted in this file."""
    boxes = 0
    out_lines = []
    for raw in src_path.read_text().splitlines():
        new = convert_line(raw)
        if new is None:
            continue
        out_lines.append(new)
        boxes += 1

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    dst_path.write_text("\n".join(out_lines) + ("\n" if out_lines else ""))
    return boxes


def main():
    ap = argparse.ArgumentParser(description="Convert YOLO detection labels to YOLOv8 OBB labels.")
    ap.add_argument("input", help="Folder containing the old .txt label files")
    ap.add_argument("-o", "--output", help="Output folder. Omit to overwrite IN PLACE.")
    ap.add_argument("-r", "--recursive", action="store_true",
                    help="Recurse into subfolders (keeps folder structure in output)")
    args = ap.parse_args()

    in_dir = Path(args.input)
    if not in_dir.is_dir():
        raise SystemExit(f"Not a folder: {in_dir}")

    out_dir = Path(args.output) if args.output else in_dir
    in_place = args.output is None

    pattern = "**/*.txt" if args.recursive else "*.txt"
    files = sorted(in_dir.glob(pattern))
    # Don't treat a dataset config like classes.txt as a label file by accident.
    files = [f for f in files if f.name.lower() not in {"classes.txt", "readme.txt"}]

    if not files:
        raise SystemExit(f"No .txt files found in {in_dir}")

    total_files = total_boxes = errors = 0
    for f in files:
        dst = f if in_place else out_dir / f.relative_to(in_dir)
        try:
            total_boxes += convert_file(f, dst)
            total_files += 1
        except ValueError as e:
            errors += 1
            print(f"  SKIPPED {f.name}: {e}")

    where = "in place" if in_place else f"-> {out_dir}"
    print(f"\nDone {where}")
    print(f"  files converted: {total_files}")
    print(f"  boxes converted: {total_boxes}")
    if errors:
        print(f"  files with errors (left untouched in output): {errors}")


if __name__ == "__main__":
    main()