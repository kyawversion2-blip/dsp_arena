#!/usr/bin/env python3
"""
Batch update spectrogram title labels directly on image pixels.

Mapping rules:
    frog    -> A
    chicken -> B
    pig     -> C
    snake   -> D

Requirements:
    1. Target/detect the title text area at the top of each spectrogram ("Result (...) Name (...)").
    2. Cover the original class name with black background and redraw mapped label (A, B, C, D)
       in matching white font.
    3. Save modified images into output_spectrograms/ without overwriting originals.
"""

import os
import re
import sys
import glob
import argparse
import numpy as np
from PIL import Image, ImageFont, ImageDraw

CLASS_MAPPING = {
    "frog": "A",
    "chicken": "B",
    "pig": "C",
    "snake": "D",
}

FONT_CANDIDATES = [
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/System/Library/Fonts/Arial.ttf",
    "/Library/Fonts/Arial.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
]


def resolve_font(size: int = 34) -> ImageFont.FreeTypeFont:
    """Resolve an available system font matching the original spectrogram styling."""
    for path in FONT_CANDIDATES:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                continue
    # Fallback to PIL default
    return ImageFont.load_default()


def detect_title_bbox(arr: np.ndarray, max_row: int = 65):
    """
    Detect the bounding box and horizontal center of the title text in the top rows.
    Returns (min_x, max_x, min_y, max_y, center_x).
    """
    # Look for bright text pixels (R, G, B > 100) in the top rows
    mask = (arr[:max_row, :, 0] > 100) & (arr[:max_row, :, 1] > 100) & (arr[:max_row, :, 2] > 100)
    coords = np.argwhere(mask)
    if len(coords) == 0:
        return None
    min_y, min_x = coords.min(axis=0)
    max_y, max_x = coords.max(axis=0)
    center_x = (min_x + max_x) / 2.0
    return int(min_x), int(max_x), int(min_y), int(max_y), float(center_x)


def parse_filename(filename: str):
    """
    Extract (index, result_number, class_name) from filename pattern.
    Example: 0_Result_1__Name_Chicken.png -> ("0", "1", "Chicken")
    """
    base = os.path.basename(filename)
    m = re.match(r"^(\d+)_Result_(\d+)__Name_([A-Za-z]+)\.png$", base, re.IGNORECASE)
    if m:
        return m.group(1), m.group(2), m.group(3)
    return None, None, None


def process_spectrogram_image(
    input_path: str,
    output_path: str,
    font: ImageFont.FreeTypeFont,
    font_size: int = 34,
    y_pos: int = 15,
) -> dict:
    """
    Load image, detect title area, mask and redraw new mapped title, then save.
    """
    img = Image.open(input_path)
    is_rgba = img.mode == "RGBA"
    working_img = img.convert("RGBA" if is_rgba else "RGB")
    arr = np.array(working_img)

    idx, res_num, cls_name = parse_filename(input_path)
    if not cls_name:
        raise ValueError(f"Could not parse filename format: {input_path}")

    mapped_letter = CLASS_MAPPING.get(cls_name.lower())
    if not mapped_letter:
        raise ValueError(f"Unknown class '{cls_name}' in {input_path}")

    # Detect title bounding box
    bbox_info = detect_title_bbox(arr, max_row=65)
    if bbox_info is not None:
        min_x, max_x, min_y, max_y, center_x = bbox_info
    else:
        # Default fallback center for standard plot dimensions
        center_x = 655.5

    # New title text format
    new_text = f"Result ({res_num})  Name ({mapped_letter})"

    # Clear title area with solid black (RGB: 0, 0, 0, Alpha: 255 if RGBA)
    draw = ImageDraw.Draw(working_img)
    bg_fill = (0, 0, 0, 255) if is_rgba else (0, 0, 0)
    text_fill = (255, 255, 255, 255) if is_rgba else (255, 255, 255)

    # Black out the title band safely above the axis ticks (y: 5 to 60)
    # Spanning across title bounds with margin
    clear_left = max(0, int(center_x - 350))
    clear_right = min(working_img.width, int(center_x + 350))
    draw.rectangle([(clear_left, 5), (clear_right, 62)], fill=bg_fill)

    # Measure rendered text to center it accurately
    text_bbox = font.getbbox(new_text)
    text_w = text_bbox[2] - text_bbox[0]
    draw_x = center_x - (text_w / 2.0)

    # Redraw new mapped title
    draw.text((draw_x, y_pos), new_text, fill=text_fill, font=font)

    # Save output
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    working_img.save(output_path, format="PNG")

    return {
        "file": os.path.basename(input_path),
        "result_num": res_num,
        "original_class": cls_name,
        "mapped_label": mapped_letter,
        "output_path": output_path,
        "center_x": center_x,
    }


def main():
    parser = argparse.ArgumentParser(description="Batch update spectrogram titles.")
    parser.add_argument(
        "--input-dir",
        default="/Users/kyawnandar/Downloads/RippleSpectrogramsAG",
        help="Source directory containing spectrogram images.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Destination directory. Defaults to input-dir/output_spectrograms.",
    )
    args = parser.parse_args()

    input_dir = os.path.abspath(args.input_dir)
    if not os.path.isdir(input_dir):
        print(f"Error: Input directory does not exist: {input_dir}", file=sys.stderr)
        sys.exit(1)

    output_dir = args.output_dir
    if not output_dir:
        output_dir = os.path.join(input_dir, "output_spectrograms")
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    png_files = sorted(glob.glob(os.path.join(input_dir, "*.png")))
    if not png_files:
        print(f"No PNG files found in {input_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(png_files)} spectrograms in {input_dir}")
    print(f"Output directory: {output_dir}")

    font = resolve_font(size=34)
    results = []

    for f in png_files:
        fname = os.path.basename(f)
        out_path = os.path.join(output_dir, fname)
        res = process_spectrogram_image(f, out_path, font=font)
        results.append(res)
        print(f"  [OK] {fname} -> Result ({res['result_num']}) Name ({res['mapped_label']})")

    print("\nBatch update complete.")
    print(f"Total processed: {len(results)}/{len(png_files)}")


if __name__ == "__main__":
    main()
