"""
Convert Standard YOLO labels → YOLOv8 OBB (Oriented Bounding Box) format

Standard YOLO (input):
    <class> <cx> <cy> <w> <h>          (all normalized 0–1)

YOLOv8 OBB (output):
    <class> <x1> <y1> <x2> <y2> <x3> <y3> <x4> <y4>   (normalized, clockwise from top-left)

Since the input boxes are axis-aligned (no rotation), the 4 corners are computed
directly from the center + width/height. The angle is implicitly 0°.
"""

import os
import argparse
from pathlib import Path


def xywh_to_obb(cx, cy, w, h):
    """
    Convert normalized (cx, cy, w, h) to 4 corner points in clockwise order:
        top-left → top-right → bottom-right → bottom-left
    """
    half_w = w / 2
    half_h = h / 2

    x1, y1 = cx - half_w, cy - half_h  # top-left
    x2, y2 = cx + half_w, cy - half_h  # top-right
    x3, y3 = cx + half_w, cy + half_h  # bottom-right
    x4, y4 = cx - half_w, cy + half_h  # bottom-left

    return x1, y1, x2, y2, x3, y3, x4, y4


def convert_file(src_path: Path, dst_path: Path):
    """Convert a single YOLO label file to OBB format."""
    lines_out = []

    with open(src_path, "r") as f:
        for line_num, raw in enumerate(f, 1):
            line = raw.strip()
            if not line:
                continue  # skip blank lines

            parts = line.split()
            if len(parts) != 5:
                print(f"  [WARN] {src_path.name} line {line_num}: "
                      f"expected 5 values, got {len(parts)} — skipping")
                continue

            try:
                cls = int(parts[0])
                cx, cy, w, h = map(float, parts[1:])
            except ValueError as e:
                print(f"  [WARN] {src_path.name} line {line_num}: parse error ({e}) — skipping")
                continue

            # Clamp to [0, 1] just in case of floating-point drift
            cx = max(0.0, min(1.0, cx))
            cy = max(0.0, min(1.0, cy))
            w  = max(0.0, min(1.0, w))
            h  = max(0.0, min(1.0, h))

            x1, y1, x2, y2, x3, y3, x4, y4 = xywh_to_obb(cx, cy, w, h)

            coords = " ".join(f"{v:.6f}" for v in (x1, y1, x2, y2, x3, y3, x4, y4))
            lines_out.append(f"{cls} {coords}\n")

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    with open(dst_path, "w") as f:
        f.writelines(lines_out)

    return len(lines_out)


def convert_dataset(input_dir: str, output_dir: str):
    input_path  = Path(input_dir)
    output_path = Path(output_dir)

    if not input_path.exists():
        raise FileNotFoundError(f"Input directory not found: {input_path}")

    label_files = list(input_path.rglob("*.txt"))
    if not label_files:
        print("No .txt label files found in the input directory.")
        return

    print(f"Found {len(label_files)} label file(s) in '{input_path}'")
    print(f"Output → '{output_path}'\n")

    total_boxes = 0
    for src in label_files:
        # Preserve subdirectory structure (e.g. train/labels/, val/labels/)
        rel = src.relative_to(input_path)
        dst = output_path / rel

        n = convert_file(src, dst)
        total_boxes += n
        print(f"  ✓  {rel}  ({n} box{'es' if n != 1 else ''})")

    print(f"\nDone — converted {total_boxes} bounding box(es) across {len(label_files)} file(s).")


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Convert standard YOLO labels to YOLOv8 OBB format."
    )
    parser.add_argument(
        "input_dir",
        help="Root directory containing your existing YOLO .txt label files "
             "(subdirectories like train/labels/ are preserved automatically)."
    )
    parser.add_argument(
        "output_dir",
        help="Destination directory for the converted OBB label files."
    )
    args = parser.parse_args()

    convert_dataset(args.input_dir, args.output_dir)


if __name__ == "__main__":
    main()