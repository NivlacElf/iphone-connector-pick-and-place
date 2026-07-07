#!/usr/bin/env python3
"""
YOLOv8-OBB label bootstrapper (manual-config edition).

Hand-annotate ONE reference image (two adjacent corners + the perpendicular
side length) and this script propagates an oriented bounding box to every
other image using the object's known world position (mm) encoded in each
filename, together with the z-dependent millimetre-per-pixel scale.

You no longer pass anything on the command line. Edit the EDIT-ME block and
run:  python obb_propagate.py

----------------------------------------------------------------------------
HOW IT WORKS
----------------------------------------------------------------------------
1. From the reference image you give two corners P1=(X1,Y1), P2=(X2,Y2) in
   pixels. P1->P2 is ONE edge of the rectangle (the "width" side). HEIGHT is
   the perpendicular side length in pixels. Because the box is a rectangle the
   two sides are always 90 deg apart, so no angle input is needed - the
   orientation comes straight from the P1->P2 line you drew.

2. For every other image the script reads (x_mm, y_mm, z_mm) from the filename
   and rebuilds the box:
       - physical size is constant  -> pixel size = phys_mm / mm_per_pixel(z)
       - centre moves by (delta_mm / mm_per_pixel(z)) from the reference centre
       - orientation = reference edge angle (constant), unless you turn on
         USE_FILENAME_ANGLE (see note in CONFIG).

3. It writes:
       <OUT>/dataset/images/*           copies of the source images
       <OUT>/dataset/labels/*.txt       YOLOv8-OBB labels (normalised, 4 corners)
       <OUT>/visualizations/*           images with the predicted box drawn on
       <OUT>/dataset/data.yaml          starter dataset config

----------------------------------------------------------------------------
FILENAME FORMAT (example)
----------------------------------------------------------------------------
    -51.9_img_x198_y138.5_z200.jpg
        (-51.9 is the world angle; it is IGNORED unless USE_FILENAME_ANGLE)
        x = 198   (mm)
        y = 138.5 (mm)
        z = 200   (mm)
"""

import math
import os
import re
import shutil
import sys

import cv2
import numpy as np

# ===========================================================================
# EDIT ME  --  everything you normally change lives here
# ===========================================================================

# ---- paths ----
# Folder names are relative to wherever this .py file lives.
_HERE = os.path.dirname(os.path.abspath(__file__))
IMAGES_DIR = os.path.join(_HERE, "127.3")
OUT_DIR    = os.path.join(_HERE, "127.3output")

# ---- reference image (must be a file inside IMAGES_DIR) ----
REF_IMAGE  = "127.3_img_x197.5_y139.5_z200.jpg"

# ---- the one hand-labelled box on the reference image ----
# P1 -> P2 is ONE edge of the rectangle (the "width" side), in pixels.
X1, Y1 = 911, 1142
X2, Y2 = 776, 963
# Perpendicular side length of the rectangle (the "height"), in pixels.
HEIGHT = 775

# ===========================================================================
# CONFIG  --  flip these after looking at the visualizations if boxes are wrong
# ===========================================================================

# mm-per-pixel as a function of z (mm).
def mm_per_pixel(z):
    return (z - 121.8) / 8150.556

# Which way the perpendicular (height) side is built from the P1->P2 edge.
# +1 = rotate the edge +90 deg, -1 = rotate -90 deg. Flip if P3/P4 land on the
# wrong side of the P1->P2 line.
PERP_SIGN = +1

# World-mm -> image-pixel axis mapping. Image x grows right, image y grows DOWN.
# If the object moving in +x_mm makes the box move LEFT in the image, set -1.
# If the object moving in +y_mm makes the box move UP in the image, set -1.
SIGN_X = -1
SIGN_Y = -1

# Should each box be rotated by the object's world angle read from the filename?
#   False -> every box keeps the reference orientation. Use this when the object
#            looks the SAME in every frame and only its position changes. This
#            is the "angle not needed" case you described.
#   True  -> box orientation follows the filename angle. Use this ONLY if the
#            object actually rotates from frame to frame (the boxes will come out
#            misaligned for rotated parts if you leave this off).
USE_FILENAME_ANGLE = False
SIGN_ROT = +1   # only used when USE_FILENAME_ANGLE is True

# Which z-scale to use when converting a position delta (mm) into pixels.
#   "per_image"  -> use the current image's z (pinhole-consistent default)
#   "reference"  -> use the reference frame's z (use if z is constant per batch)
POSITION_SCALE_MODE = "per_image"

# Clamp normalised corner coords into [0, 1].
CLAMP_TO_IMAGE = True

# Class id written to every label.
CLASS_ID = 0

# Visualization style
BOX_COLOR = (0, 255, 0)      # BGR
BOX_THICKNESS = 2
DRAW_CORNER_LABELS = True

# Image extensions to process
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

# Position is ALWAYS required (x, y, z). Matches "_x198_y138.5_z200".
POS_RE = re.compile(
    r"_x(?P<x>-?\d+(?:\.\d+)?)"
    r"_y(?P<y>-?\d+(?:\.\d+)?)"
    r"_z(?P<z>-?\d+(?:\.\d+)?)",
    re.IGNORECASE,
)

# Optional leading angle. Matches "-51.9_..." and also the old "30.1deg_...".
ANGLE_RE = re.compile(r"^\s*(?P<angle>-?\d+(?:\.\d+)?)(?:deg)?_", re.IGNORECASE)

# ===========================================================================
# Geometry helpers
# ===========================================================================

def parse_filename(name):
    """Return dict(x, y, z, angle) from a filename, or None if no x/y/z found."""
    pos = POS_RE.search(name)
    if not pos:
        return None
    info = {
        "x": float(pos.group("x")),
        "y": float(pos.group("y")),
        "z": float(pos.group("z")),
        "angle": None,
    }
    am = ANGLE_RE.search(name)
    if am:
        info["angle"] = float(am.group("angle"))
    return info


def corners_from_center(cx, cy, edge_angle_deg, w_px, l_px, perp_sign):
    """
    Build the 4 rectangle corners from its centre, the edge (width) angle in
    pixels, and the width and length (height) in pixels.

    P1 -> P2 is the width edge (direction = edge_angle).
    P2 -> P3 is the perpendicular (height) edge.
    Returns [(x,y) * 4] in order P1, P2, P3, P4.
    """
    a = math.radians(edge_angle_deg)
    ex, ey = math.cos(a), math.sin(a)                            # unit width dir
    px, py = -math.sin(a) * perp_sign, math.cos(a) * perp_sign   # +/-90 deg

    hw_x, hw_y = ex * w_px / 2.0, ey * w_px / 2.0  # half-width vector
    hl_x, hl_y = px * l_px / 2.0, py * l_px / 2.0  # half-length vector

    p1 = (cx - hw_x - hl_x, cy - hw_y - hl_y)
    p2 = (cx + hw_x - hl_x, cy + hw_y - hl_y)
    p3 = (cx + hw_x + hl_x, cy + hw_y + hl_y)
    p4 = (cx - hw_x + hl_x, cy - hw_y + hl_y)
    return [p1, p2, p3, p4]


def build_reference_model(ref_info, x1, y1, x2, y2, height_px):
    """Derive the constant physical box and the pixel-space reference state."""
    mmpp_ref = mm_per_pixel(ref_info["z"])
    if mmpp_ref <= 0:
        sys.exit(f"ERROR: reference z={ref_info['z']} gives non-positive mm/px "
                 f"({mmpp_ref:.5f}). Check the z value / equation.")

    edge_angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
    w_px = math.hypot(x2 - x1, y2 - y1)
    l_px = float(height_px)

    # centre = P1 + half width along edge + half height along perpendicular
    a = math.radians(edge_angle)
    ex, ey = math.cos(a), math.sin(a)
    px, py = -math.sin(a) * PERP_SIGN, math.cos(a) * PERP_SIGN
    cx = x1 + ex * w_px / 2.0 + px * l_px / 2.0
    cy = y1 + ey * w_px / 2.0 + py * l_px / 2.0

    return {
        "center_px": (cx, cy),
        "edge_angle_deg": edge_angle,
        "w_phys_mm": w_px * mmpp_ref,
        "l_phys_mm": l_px * mmpp_ref,
        "x_ref_mm": ref_info["x"],
        "y_ref_mm": ref_info["y"],
        "angle_ref_deg": ref_info["angle"],
        "mmpp_ref": mmpp_ref,
    }


def predict_corners(model, info):
    """Predict the 4 pixel corners for an image given the reference model."""
    mmpp = mm_per_pixel(info["z"])
    if mmpp <= 0:
        return None  # invalid scale; skip

    # Pixel size of the (physically constant) box at this z.
    w_px = model["w_phys_mm"] / mmpp
    l_px = model["l_phys_mm"] / mmpp

    # Scale used to convert a position delta (mm) into pixels.
    mmpp_pos = mmpp if POSITION_SCALE_MODE == "per_image" else model["mmpp_ref"]

    dx_mm = info["x"] - model["x_ref_mm"]
    dy_mm = info["y"] - model["y_ref_mm"]
    cx = model["center_px"][0] + SIGN_X * dx_mm / mmpp_pos
    cy = model["center_px"][1] + SIGN_Y * dy_mm / mmpp_pos

    edge_angle = model["edge_angle_deg"]
    if USE_FILENAME_ANGLE:
        edge_angle += SIGN_ROT * (info["angle"] - model["angle_ref_deg"])

    return corners_from_center(cx, cy, edge_angle, w_px, l_px, PERP_SIGN)


# ===========================================================================
# Output writers
# ===========================================================================

def write_label(path, corners, img_w, img_h):
    """Write one YOLOv8-OBB label line: class x1 y1 x2 y2 x3 y3 x4 y4 (norm)."""
    parts = [str(CLASS_ID)]
    for (x, y) in corners:
        nx = x / img_w
        ny = y / img_h
        if CLAMP_TO_IMAGE:
            nx = min(max(nx, 0.0), 1.0)
            ny = min(max(ny, 0.0), 1.0)
        parts.append(f"{nx:.6f}")
        parts.append(f"{ny:.6f}")
    with open(path, "w") as f:
        f.write(" ".join(parts) + "\n")


def draw_visualization(img, corners):
    pts = np.array([[int(round(x)), int(round(y))] for (x, y) in corners],
                   dtype=np.int32)
    cv2.polylines(img, [pts], isClosed=True, color=BOX_COLOR,
                  thickness=BOX_THICKNESS)
    if DRAW_CORNER_LABELS:
        for i, (x, y) in enumerate(corners, start=1):
            cv2.circle(img, (int(round(x)), int(round(y))), 4, BOX_COLOR, -1)
            cv2.putText(img, f"P{i}", (int(round(x)) + 5, int(round(y)) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, BOX_COLOR, 1, cv2.LINE_AA)
    return img


def write_data_yaml(out_dir):
    yaml_path = os.path.join(out_dir, "dataset", "data.yaml")
    content = (
        "# Starter config for YOLOv8-OBB. Split images/labels into train/val\n"
        "# yourself, then point these paths at the split folders.\n"
        f"path: {os.path.abspath(os.path.join(out_dir, 'dataset'))}\n"
        "train: images\n"
        "val: images\n"
        "names:\n"
        "  0: object\n"
    )
    with open(yaml_path, "w") as f:
        f.write(content)


# ===========================================================================
# Main
# ===========================================================================

def main():
    images_dir = IMAGES_DIR
    out_dir = OUT_DIR

    if not os.path.isdir(images_dir):
        sys.exit(f"ERROR: IMAGES_DIR does not exist: {images_dir}")

    # ---- Output folders ----
    img_out = os.path.join(out_dir, "dataset", "images")
    lbl_out = os.path.join(out_dir, "dataset", "labels")
    viz_out = os.path.join(out_dir, "visualizations")
    for d in (img_out, lbl_out, viz_out):
        os.makedirs(d, exist_ok=True)

    # ---- Reference ----
    if not os.path.isfile(os.path.join(images_dir, REF_IMAGE)):
        print(f"ERROR: reference image not found in IMAGES_DIR: {REF_IMAGE}")
        print(f"  looked in: {os.path.abspath(images_dir)}")
        print("  files actually present (between the brackets):")
        for f in sorted(os.listdir(images_dir)):
            print(f"    [{f}]")
        sys.exit(1)

    ref_info = parse_filename(REF_IMAGE)
    if ref_info is None:
        sys.exit(f"ERROR: reference filename '{REF_IMAGE}' has no _x.._y.._z.. "
                 f"part (e.g. -51.9_img_x198_y138.5_z200.jpg).")

    if USE_FILENAME_ANGLE and ref_info["angle"] is None:
        sys.exit("ERROR: USE_FILENAME_ANGLE is True but no leading angle was "
                 f"found in the reference filename '{REF_IMAGE}'.")

    model = build_reference_model(ref_info, X1, Y1, X2, Y2, HEIGHT)

    print("Reference model:")
    print(f"  centre (px)      : ({model['center_px'][0]:.1f}, {model['center_px'][1]:.1f})")
    print(f"  edge angle (deg) : {model['edge_angle_deg']:.2f}")
    print(f"  width  (mm)      : {model['w_phys_mm']:.3f}")
    print(f"  height (mm)      : {model['l_phys_mm']:.3f}")
    print(f"  mm/px @ ref z    : {model['mmpp_ref']:.5f}")
    print(f"  use angle        : {USE_FILENAME_ANGLE}")
    print()

    # ---- Process every image ----
    files = sorted(f for f in os.listdir(images_dir)
                   if f.lower().endswith(IMAGE_EXTS))
    if not files:
        sys.exit(f"ERROR: no images found in {images_dir}")

    processed, skipped = 0, 0
    for name in files:
        info = parse_filename(name)
        if info is None:
            print(f"  SKIP (no x/y/z in name): {name}")
            skipped += 1
            continue
        if USE_FILENAME_ANGLE and info["angle"] is None:
            print(f"  SKIP (no angle in name): {name}")
            skipped += 1
            continue

        src = os.path.join(images_dir, name)
        img = cv2.imread(src)
        if img is None:
            print(f"  SKIP (unreadable): {name}")
            skipped += 1
            continue
        h, w = img.shape[:2]

        corners = predict_corners(model, info)
        if corners is None:
            print(f"  SKIP (bad z scale): {name}")
            skipped += 1
            continue

        # dataset image copy
        shutil.copy2(src, os.path.join(img_out, name))
        # label
        stem = os.path.splitext(name)[0]
        write_label(os.path.join(lbl_out, stem + ".txt"), corners, w, h)
        # visualization
        viz = draw_visualization(img.copy(), corners)
        cv2.imwrite(os.path.join(viz_out, name), viz)

        processed += 1

    write_data_yaml(out_dir)

    print()
    print(f"Done. processed={processed}  skipped={skipped}")
    print(f"  images : {img_out}")
    print(f"  labels : {lbl_out}")
    print(f"  viz    : {viz_out}")
    print()
    print("Now open the visualizations folder and verify the boxes.")
    print("If they are wrong, adjust PERP_SIGN / SIGN_X / SIGN_Y in the CONFIG")
    print("block and re-run.")


if __name__ == "__main__":
    main()