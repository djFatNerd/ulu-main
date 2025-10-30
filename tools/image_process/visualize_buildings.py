#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Visualize building footprints from a JSONL produced by convert_building_coordinates.py (LOCAL-aligned)

Two modes:
  - bbox:    draw red rectangles from "bbox_px"
  - polygon: draw red outlines from "polygon_px" rings (exterior + holes)

Usage examples:
  python visualize_buildings.py \
    --jsonl buildings_pixels.jsonl \
    --base_image city.png \
    --out viz_bbox.png \
    --mode bbox \
    --num 10 \
    --line_width 2 \
    --seed 42

  python visualize_buildings.py \
    --jsonl buildings_pixels.jsonl \
    --base_image city.png \
    --out viz_poly.png \
    --mode polygon \
    --num 10 \
    --line_width 2
"""

import argparse
import json
import random
from typing import List, Tuple, Any

from PIL import Image, ImageDraw


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--jsonl", required=True, help="Path to JSONL exported by convert_building_coordinates.py")
    p.add_argument("--base_image", required=True, help="Path to base image (png/jpg) converted from the raster array")
    p.add_argument("--out", required=True, help="Output image path")
    p.add_argument("--mode", choices=["bbox", "polygon"], required=True, help="Visualization mode")
    p.add_argument("--num", type=int, default=10, help="Number of buildings to visualize")
    p.add_argument("--line_width", type=int, default=2, help="Outline line width in pixels")
    p.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    p.add_argument("--pixel_origin", choices=["edge", "center"], default="edge",
                   help="If your renderer draws at pixel centers, choose 'center' to subtract 0.5 px.")
    return p.parse_args()


def load_jsonl(path: str) -> List[dict]:
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                items.append(obj)
            except json.JSONDecodeError:
                continue
    return items


def bbox_intersects_image(bbox: List[Any], w: int, h: int) -> bool:
    if bbox is None or any(v is None for v in bbox):
        return False
    xmin, ymin, xmax, ymax = map(float, bbox)
    return not (xmax < 0 or ymax < 0 or xmin >= w or ymin >= h)


def polygon_intersects_image(rings: List[List[List[Any]]], w: int, h: int) -> bool:
    if not rings:
        return False
    for ring in rings:
        for x, y in ring:
            x = float(x); y = float(y)
            if 0 <= x < w and 0 <= y < h:
                return True
    # Fallback: ring bbox overlaps
    for ring in rings:
        xs = [float(p[0]) for p in ring]
        ys = [float(p[1]) for p in ring]
        if not xs or not ys:
            continue
        xmin, xmax = min(xs), max(xs)
        ymin, ymax = min(ys), max(ys)
        if not (xmax < 0 or ymax < 0 or xmin >= w or ymin >= h):
            return True
    return False


def draw_bbox(draw: ImageDraw.ImageDraw, bbox: List[Any], color: Tuple[int, int, int], width: int, pixel_origin: str):
    xmin, ymin, xmax, ymax = map(float, bbox)
    if pixel_origin == "center":
        xmin -= 0.5; ymin -= 0.5; xmax -= 0.5; ymax -= 0.5
    draw.rectangle([xmin, ymin, xmax, ymax], outline=color, width=width)


def draw_polygon_rings(draw: ImageDraw.ImageDraw, rings: List[List[List[Any]]], color: Tuple[int, int, int], width: int, pixel_origin: str):
    for ring in rings:
        if len(ring) < 2:
            continue
        if pixel_origin == "center":
            pts = [(float(x) - 0.5, float(y) - 0.5) for x, y in ring]
        else:
            pts = [(float(x), float(y)) for x, y in ring]
        if pts[0] != pts[-1]:
            pts.append(pts[0])
        draw.line(pts, fill=color, width=width, joint="curve")


def main():
    args = parse_args()
    if args.seed is not None:
        random.seed(args.seed)

    data = load_jsonl(args.jsonl)
    if not data:
        raise SystemExit("No records found in JSONL.")

    img = Image.open(args.base_image).convert("RGB")
    W, H = img.size
    draw = ImageDraw.Draw(img)
    red = (255, 0, 0)

    # Strong size check: the JSONL was produced for a specific canvas size.
    js_w, js_h = None, None
    if "image_size_px" in data[0] and isinstance(data[0]["image_size_px"], list) and len(data[0]["image_size_px"]) == 2:
        js_w, js_h = int(data[0]["image_size_px"][0]), int(data[0]["image_size_px"][1])
        if (js_w, js_h) != (W, H):
            print(f"[WARN] Base image size {W}x{H} differs from JSONL {js_w}x{js_h}. Overlay may misalign.")

    candidates = []
    for rec in data:
        if args.mode == "bbox":
            bbox = rec.get("bbox_px")
            if bbox_intersects_image(bbox, W, H):
                candidates.append(("bbox", rec["id"], bbox))
        else:
            rings = rec.get("polygon_px", [])
            if polygon_intersects_image(rings, W, H):
                candidates.append(("polygon", rec["id"], rings))

    if not candidates:
        raise SystemExit("No candidates intersect the image area.")

    k = min(args.num, len(candidates))
    chosen = random.sample(candidates, k)

    for kind, fid, payload in chosen:
        if kind == "bbox":
            draw_bbox(draw, payload, red, args.line_width, args.pixel_origin)
        else:
            draw_polygon_rings(draw, payload, red, args.line_width, args.pixel_origin)

    img.save(args.out)
    print(f"Saved visualization with {k} buildings to: {args.out}")


if __name__ == "__main__":
    main()
