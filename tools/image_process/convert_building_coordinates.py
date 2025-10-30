#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GeoJSON (WGS84) -> image pixel polygons + bbox exporter (LOCAL projector aligned)

Goal:
- Guarantee pixel-perfect alignment with the OSM raster produced by tools.osm.rasterize_semantics,
  which uses a local projector anchored at (center_lat, center_lon), square canvas of side 2*radius_m,
  and resolution (meters-per-pixel).

Key assumptions (MUST match your Stage-1 generation):
- Metadata JSON (e.g., metadata_osm_overture.json) contains:
    {
      "center": {"lat": <float>, "lon": <float>},
      "radius_m": <float>,              # half side length in meters
      "resolution_m": <float>,          # meters per pixel used for rasterization
      ...
    }
- Pixel (0,0) is the top-left corner of the canvas.
- Local coordinates (meters) come from tools.osm.latlon_to_local_projector(center_lat, center_lon).
- Pixel mapping:
    x_px = (x_m + radius_m) / mpp
    y_px = (radius_m - y_m) / mpp

CLI overrides:
- You may override mpp or width/height, but for perfect alignment, they must be consistent with
  the raster's (2*radius_m / mpp) relation. If width/height are omitted, they are derived as:
    W = H = ceil(2 * radius_m / mpp)
"""

import argparse
import json
import math
from typing import List, Tuple, Union, Optional

import geopandas as gpd
from shapely.geometry import Polygon, MultiPolygon, Point

# Import the SAME projector used by your raster pipeline
from tools.osm import (
    latlon_to_local_projector,
    local_to_latlon_projector,
)

# ------------------------------
# Pixel mapping helpers
# ------------------------------
def coords_to_px_ring_local(coords_ll, to_local, radius_m: float, mpp: float, round_pix: bool):
    """
    coords_ll: iterable of (lon, lat)
    to_local: function (shapely.geometry) -> shapely.geometry
    """
    ring = []
    for lon, lat in coords_ll:
        # ✅ FIXED: convert to shapely Point before projection
        pt_m = to_local(Point(lon, lat))  # returns Point(x_m, y_m)
        x_m, y_m = pt_m.x, pt_m.y

        x_px = (x_m + radius_m) / mpp
        y_px = (radius_m - y_m) / mpp  # pixel y grows downward

        if round_pix:
            x_px = int(round(x_px))
            y_px = int(round(y_px))
        ring.append([x_px, y_px])
    return ring


def geom_ll_to_rings_px_local(
    geom_ll,
    to_local,
    radius_m: float,
    mpp: float,
    round_pix: bool = False,
    exterior_only: bool = False,
) -> List[List[List[Union[float, int]]]]:
    """
    Convert a lon/lat Polygon/MultiPolygon into pixel rings using the LOCAL projector and center/radius anchor.
    Returns a flattened list of rings (exterior + holes).
    """
    rings_px: List[List[List[Union[float, int]]]] = []

    if isinstance(geom_ll, Polygon):
        rings_px.append(coords_to_px_ring_local(list(geom_ll.exterior.coords), to_local, radius_m, mpp, round_pix))
        if not exterior_only:
            for interior in geom_ll.interiors:
                rings_px.append(coords_to_px_ring_local(list(interior.coords), to_local, radius_m, mpp, round_pix))
    elif isinstance(geom_ll, MultiPolygon):
        for poly in geom_ll.geoms:
            rings_px.append(coords_to_px_ring_local(list(poly.exterior.coords), to_local, radius_m, mpp, round_pix))
            if not exterior_only:
                for interior in poly.interiors:
                    rings_px.append(coords_to_px_ring_local(list(interior.coords), to_local, radius_m, mpp, round_pix))
    return rings_px


def rings_bbox(rings_px: List[List[List[Union[float, int]]]]) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
    xs, ys = [], []
    for ring in rings_px:
        for x, y in ring:
            xs.append(float(x))
            ys.append(float(y))
    if not xs:
        return (None, None, None, None)
    return (min(xs), min(ys), max(xs), max(ys))


# ------------------------------
# Metadata
# ------------------------------
def parse_meta(meta_path: str) -> dict:
    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f)


# ------------------------------
# Main
# ------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--geojson", required=True, help="Input GeoJSON path (WGS84)")
    ap.add_argument("--meta", required=True, help="Metadata JSON path (must contain center/radius_m/resolution_m)")
    ap.add_argument("--out", required=True, help="Output JSONL path")

    # Optional overrides (must remain consistent with raster to keep perfect alignment)
    ap.add_argument("--mpp", type=float, default=None, help="Override meters per pixel; defaults to metadata['resolution_m']")
    ap.add_argument("--size", type=int, default=None, help="Override square image size; W=H=size")
    ap.add_argument("--width", type=int, default=None, help="Override image width")
    ap.add_argument("--height", type=int, default=None, help="Override image height")
    ap.add_argument("--exterior_only", action="store_true", help="Export exterior ring only")
    ap.add_argument("--round_pix", action="store_true", help="Round pixel coordinates to integers")
    args = ap.parse_args()

    # --- Load metadata ---
    meta = parse_meta(args.meta)
    if "center" not in meta or "radius_m" not in meta:
        raise SystemExit("Metadata must include 'center':{'lat','lon'} and 'radius_m'.")

    center_lat = float(meta["center"]["lat"])
    center_lon = float(meta["center"]["lon"])
    radius_m = float(meta["radius_m"])
    mpp = float(args.mpp if args.mpp is not None else meta.get("resolution_m") or meta.get("mpp") or 1.0)

    # --- Image size ---
    if args.width is not None and args.height is not None:
        W, H = int(args.width), int(args.height)
    elif args.size is not None:
        W = H = int(args.size)
    else:
        W = H = int(math.ceil((2.0 * radius_m) / mpp))

    ideal_size = (2.0 * radius_m) / mpp
    if not (abs(W - ideal_size) < 1.0 and abs(H - ideal_size) < 1.0):
        print("[WARN] Canvas size may not perfectly align with raster resolution/radius.")

    # --- Load GeoJSON ---
    gdf_ll = gpd.read_file(args.geojson).set_crs(4326, allow_override=True)

    # --- Build local projector (SAME as raster pipeline) ---
    to_local = latlon_to_local_projector(center_lat, center_lon)
    to_lonlat = local_to_latlon_projector(center_lat, center_lon)

    # --- Canvas bbox in lon/lat (for reference only) ---
    left_m, right_m = -radius_m, +radius_m
    top_m, bottom_m = +radius_m, -radius_m

    def xy_to_lonlat(x_m: float, y_m: float):
        pt_ll = to_lonlat(Point(x_m, y_m))
        return float(pt_ll.x), float(pt_ll.y)

    (min_lon, min_lat) = xy_to_lonlat(left_m, bottom_m)
    (max_lon, max_lat) = xy_to_lonlat(right_m, top_m)

    canvas_bbox_ll = {
        "min_lon": float(min_lon),
        "min_lat": float(min_lat),
        "max_lon": float(max_lon),
        "max_lat": float(max_lat),
    }

    # --- Export per-feature pixel geometry ---
    with open(args.out, "w", encoding="utf-8") as fout:
        for idx, row in gdf_ll.iterrows():
            geom_ll = row.geometry
            if geom_ll is None or geom_ll.is_empty:
                continue

            # Feature ID
            fid = None
            fid_candidate = getattr(row, "id", None)
            if fid_candidate not in (None, ""):
                fid = str(fid_candidate)
            elif hasattr(row, "osm_id"):
                osm_id = getattr(row, "osm_id", None)
                if isinstance(osm_id, (int, str)):
                    fid = str(osm_id)
            if fid is None:
                fid = f"feature_{idx}"

            # Compute pixel coordinates (LOCAL projection)
            rings_px = geom_ll_to_rings_px_local(
                geom_ll=geom_ll,
                to_local=to_local,
                radius_m=radius_m,
                mpp=mpp,
                round_pix=args.round_pix,
                exterior_only=args.exterior_only,
            )
            xmin, ymin, xmax, ymax = rings_bbox(rings_px)

            # Original lon/lat bbox
            minx, miny, maxx, maxy = geom_ll.bounds
            bbox_lonlat = [float(minx), float(miny), float(maxx), float(maxy)]

            rec = {
                "id": fid,
                "image_size_px": [int(W), int(H)],
                "mpp": float(mpp),
                "crs": ["LOCAL", f"{center_lat:.6f},{center_lon:.6f}"],
                "canvas_anchor_local_m": {"left": float(left_m), "top": float(top_m)},
                "canvas_bbox_lonlat": canvas_bbox_ll,
                "bbox_px": [xmin, ymin, xmax, ymax],
                "bbox_lonlat": bbox_lonlat,
                "polygon_px": rings_px,
            }
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"✅ Done. Wrote JSONL to: {args.out}")
    print(f"Canvas: {W}x{H} px @ {mpp} m/px | LOCAL projector centered at ({center_lat}, {center_lon})")
    print(f"Anchor (local meters) at pixel (0,0): left={left_m:.3f}, top={top_m:.3f}")
    print("Canvas bbox (lon/lat):",
          f"[{canvas_bbox_ll['min_lon']}, {canvas_bbox_ll['min_lat']}, {canvas_bbox_ll['max_lon']}, {canvas_bbox_ll['max_lat']}]")
    print("NOTE: For pixel-perfect overlay, use the same base raster produced with this metadata.")


if __name__ == "__main__":
    main()
