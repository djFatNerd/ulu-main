"""Generate semantic maps and building taxonomies for a square area.

This script queries OpenStreetMap via the Overpass API, rasterizes semantic
classes at 1 m resolution (configurable), and produces a GeoJSON file with
multi-level building labels.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from semantic_palette import SEMANTIC_COLOR_PALETTE


# ---------------------------------------------------------------------------
# Network configuration
# ---------------------------------------------------------------------------

# Public Overpass API mirrors that we can fall back to when the primary
# endpoint is unavailable. Keeping the list close to the top of the module
# makes it easy to maintain and prevents NameError issues when referenced in
# helper functions further below.
DEFAULT_OVERPASS_FALLBACKS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass.openstreetmap.ru/api/interpreter",
    "https://overpass.osm.ch/api/interpreter",
]

# ---------------------------------------------------------------------------
# Semantic configuration
# ---------------------------------------------------------------------------
CLASS_TO_ID = {
    "ground": 0,
    "vegetation": 1,
    "water": 2,
    "building": 3,
    "road": 4,
    "traffic_road": 5,
}

# Draw order decides which class overwrites previous pixels.
DRAW_ORDER = ["vegetation", "water", "road", "traffic_road", "building"]

# Default drawing widths in meters for linear features.
ROAD_WIDTHS = {
    "default": 6.0,
    "traffic": 12.0,
}

# Minimum hole size (in pixel units) that we preserve inside buildings.
# Smaller voids are assumed to be artifacts and will be filled.
MIN_BUILDING_HOLE_PIXELS = 16


# ---------------------------------------------------------------------------
# Building classification helpers
# ---------------------------------------------------------------------------
AMENITY_MAPPING = {
    "school": ("education", "school"),
    "university": ("education", "university"),
    "college": ("education", "college"),
    "kindergarten": ("education", "kindergarten"),
    "hospital": ("healthcare", "hospital"),
    "clinic": ("healthcare", "clinic"),
    "doctors": ("healthcare", "medical_office"),
    "dentist": ("healthcare", "dental_clinic"),
    "pharmacy": ("healthcare", "pharmacy"),
    "library": ("public_service", "library"),
    "police": ("public_service", "police_station"),
    "fire_station": ("public_service", "fire_station"),
    "townhall": ("public_service", "town_hall"),
    "courthouse": ("public_service", "courthouse"),
    "community_centre": ("public_service", "community_center"),
    "arts_centre": ("culture", "arts_center"),
    "theatre": ("culture", "theatre"),
    "cinema": ("culture", "cinema"),
    "place_of_worship": ("religious", "place_of_worship"),
    "church": ("religious", "church"),
    "mosque": ("religious", "mosque"),
    "temple": ("religious", "temple"),
    "synagogue": ("religious", "synagogue"),
    "bank": ("commercial", "bank"),
    "atm": ("commercial", "atm"),
    "post_office": ("public_service", "post_office"),
    "restaurant": ("hospitality", "restaurant"),
    "cafe": ("hospitality", "cafe"),
    "fast_food": ("hospitality", "fast_food"),
    "bar": ("hospitality", "bar"),
    "pub": ("hospitality", "pub"),
    "hotel": ("hospitality", "hotel"),
    "motel": ("hospitality", "motel"),
    "guest_house": ("hospitality", "guest_house"),
    "hostel": ("hospitality", "hostel"),
    "parking": ("transport", "parking"),
    "bus_station": ("transport", "bus_station"),
    "ferry_terminal": ("transport", "ferry_terminal"),
    "train_station": ("transport", "train_station"),
    "aerodrome": ("transport", "airport_terminal"),
    "marketplace": ("commercial", "marketplace"),
    "sports_centre": ("recreation", "sports_center"),
    "stadium": ("recreation", "stadium"),
    "gym": ("recreation", "gym"),
    "swimming_pool": ("recreation", "swimming_pool"),
    "kindergarten": ("education", "kindergarten"),
}

SHOP_MAPPING = {
    "supermarket": ("commercial", "supermarket"),
    "convenience": ("commercial", "convenience_store"),
    "mall": ("commercial", "shopping_mall"),
    "department_store": ("commercial", "department_store"),
    "bakery": ("commercial", "bakery"),
    "butcher": ("commercial", "butcher"),
    "greengrocer": ("commercial", "greengrocer"),
    "clothes": ("commercial", "clothing_store"),
    "fashion": ("commercial", "fashion_store"),
    "electronics": ("commercial", "electronics_store"),
    "furniture": ("commercial", "furniture_store"),
    "hardware": ("commercial", "hardware_store"),
    "car": ("commercial", "car_dealership"),
    "car_repair": ("industrial", "auto_repair"),
    "fuel": ("transport", "fuel_station"),
    "bicycle": ("commercial", "bicycle_shop"),
    "sports": ("commercial", "sports_shop"),
    "books": ("commercial", "bookstore"),
    "pharmacy": ("healthcare", "pharmacy"),
}

BUILDING_MAPPING = {
    "apartments": ("residential", "apartment"),
    "residential": ("residential", "residential"),
    "house": ("residential", "house"),
    "detached": ("residential", "detached_house"),
    "semidetached_house": ("residential", "semi_detached_house"),
    "terrace": ("residential", "terrace"),
    "bungalow": ("residential", "bungalow"),
    "static_caravan": ("residential", "mobile_home"),
    "dormitory": ("residential", "dormitory"),
    "hotel": ("hospitality", "hotel"),
    "commercial": ("commercial", "commercial"),
    "retail": ("commercial", "retail"),
    "office": ("commercial", "office"),
    "industrial": ("industrial", "industrial"),
    "warehouse": ("industrial", "warehouse"),
    "manufacture": ("industrial", "manufacturing"),
    "factory": ("industrial", "factory"),
    "supermarket": ("commercial", "supermarket"),
    "kindergarten": ("education", "kindergarten"),
    "school": ("education", "school"),
    "college": ("education", "college"),
    "university": ("education", "university"),
    "hospital": ("healthcare", "hospital"),
    "clinic": ("healthcare", "clinic"),
    "church": ("religious", "church"),
    "mosque": ("religious", "mosque"),
    "synagogue": ("religious", "synagogue"),
    "cathedral": ("religious", "cathedral"),
    "chapel": ("religious", "chapel"),
    "temple": ("religious", "temple"),
    "hangar": ("transport", "hangar"),
    "train_station": ("transport", "train_station"),
    "transportation": ("transport", "transportation"),
    "sports_hall": ("recreation", "sports_hall"),
    "stadium": ("recreation", "stadium"),
    "grandstand": ("recreation", "grandstand"),
    "civic": ("public_service", "civic"),
    "government": ("public_service", "government"),
}

LANDUSE_MAPPING = {
    "residential": ("residential", "residential_zone"),
    "retail": ("commercial", "retail_zone"),
    "commercial": ("commercial", "commercial_zone"),
    "industrial": ("industrial", "industrial_zone"),
    "military": ("public_service", "military"),
    "railway": ("transport", "railway"),
    "recreation_ground": ("recreation", "recreation_ground"),
    "cemetery": ("religious", "cemetery"),
}


def classify_building(tags: Dict[str, str]) -> Tuple[str, Optional[str], Optional[str], List[str]]:
    """Return (primary, secondary, tertiary, sources) for a building."""

    sources: List[str] = []

    amenity = tags.get("amenity")
    if amenity and amenity in AMENITY_MAPPING:
        primary, secondary = AMENITY_MAPPING[amenity]
        tertiary = f"amenity:{amenity}"
        sources.append(f"amenity={amenity}")
        return primary, secondary, tertiary, sources

    shop = tags.get("shop")
    if shop and shop in SHOP_MAPPING:
        primary, secondary = SHOP_MAPPING[shop]
        tertiary = f"shop:{shop}"
        sources.append(f"shop={shop}")
        return primary, secondary, tertiary, sources

    building = tags.get("building")
    if building and building in BUILDING_MAPPING:
        primary, secondary = BUILDING_MAPPING[building]
        tertiary = f"building:{building}"
        sources.append(f"building={building}")
        return primary, secondary, tertiary, sources

    landuse = tags.get("landuse")
    if landuse and landuse in LANDUSE_MAPPING:
        primary, secondary = LANDUSE_MAPPING[landuse]
        tertiary = f"landuse:{landuse}"
        sources.append(f"landuse={landuse}")
        return primary, secondary, tertiary, sources

    building_use = tags.get("building:use")
    if building_use:
        sources.append(f"building:use={building_use}")
        return "mixed_use", "building_use", f"building_use:{building_use}", sources

    return "unknown", None, None, sources


# ---------------------------------------------------------------------------
# Geospatial helpers
# ---------------------------------------------------------------------------

def meters_per_degree_lat(lat_deg: float) -> float:
    lat_rad = math.radians(lat_deg)
    return (
        111132.92
        - 559.82 * math.cos(2 * lat_rad)
        + 1.175 * math.cos(4 * lat_rad)
        - 0.0023 * math.cos(6 * lat_rad)
    )


def meters_per_degree_lon(lat_deg: float) -> float:
    lat_rad = math.radians(lat_deg)
    return (
        111412.84 * math.cos(lat_rad)
        - 93.5 * math.cos(3 * lat_rad)
        + 0.118 * math.cos(5 * lat_rad)
    )


def get_numpy():
    """Import numpy lazily to avoid hard dependency during CLI discovery."""

    import numpy

    return numpy


def get_requests():
    """Import requests lazily so --help works without dependencies installed."""

    import requests

    return requests


def get_pillow():
    """Import Pillow lazily for rasterization utilities."""

    from PIL import Image, ImageDraw

    return Image, ImageDraw


def get_shapely_geometry():
    """Import shapely geometry helpers lazily."""

    from shapely.geometry import LineString, Point, Polygon, mapping

    return LineString, Point, Polygon, mapping


def get_shapely_transform():
    """Import shapely transform lazily."""

    from shapely.ops import transform

    return transform


def latlon_to_local_projector(lat0: float, lon0: float):
    meters_lat = meters_per_degree_lat(lat0)
    meters_lon = meters_per_degree_lon(lat0)
    transform_fn = get_shapely_transform()

    def _project(lon, lat, z=None):
        np = get_numpy()
        lon_arr = np.asarray(lon)
        lat_arr = np.asarray(lat)
        x = (lon_arr - lon0) * meters_lon
        y = (lat_arr - lat0) * meters_lat
        return x, y

    return lambda geom: transform_fn(_project, geom)


# ---------------------------------------------------------------------------
# OSM extraction helpers
# ---------------------------------------------------------------------------

def compute_bbox(lat: float, lon: float, radius_m: float) -> Tuple[float, float, float, float]:
    lat_margin = radius_m / meters_per_degree_lat(lat)
    lon_margin = radius_m / meters_per_degree_lon(lat)
    south = lat - lat_margin
    north = lat + lat_margin
    west = lon - lon_margin
    east = lon + lon_margin
    return south, west, north, east


def build_overpass_query(bbox: Tuple[float, float, float, float]) -> str:
    south, west, north, east = bbox
    bbox_str = f"{south},{west},{north},{east}"
    return f"""
    [out:json][timeout:180];
    (
      way["building"]({bbox_str});
      way["landuse"]({bbox_str});
      way["natural"]({bbox_str});
      way["leisure"]({bbox_str});
      way["waterway"]({bbox_str});
      way["highway"]({bbox_str});
    );
    out geom;
    """


def _build_overpass_candidates(primary_url: str) -> List[str]:
    urls = [primary_url]
    for candidate in DEFAULT_OVERPASS_FALLBACKS:
        if candidate not in urls:
            urls.append(candidate)
    return urls


def download_osm(
    bbox: Tuple[float, float, float, float], overpass_url: str, requests_module, retries: int = 3
) -> List[dict]:
    query = build_overpass_query(bbox)
    last_error: Optional[Exception] = None
    urls = _build_overpass_candidates(overpass_url)
    RequestException = requests_module.exceptions.RequestException

    for url_index, url in enumerate(urls):
        for attempt in range(1, retries + 1):
            try:
                response = requests_module.post(url, data=query.encode("utf-8"), timeout=300)
                response.raise_for_status()
                payload = response.json()
                logging.info("Downloaded OSM data from %s", url)
                return payload.get("elements", [])
            except RequestException as exc:
                last_error = exc
                logging.warning(
                    "Attempt %d/%d to download OSM data from %s failed: %s",
                    attempt,
                    retries,
                    url,
                    exc,
                )
                if attempt < retries:
                    # Basic linear backoff to give the Overpass API time to recover.
                    sleep_seconds = min(30, 5 * attempt)
                    time.sleep(sleep_seconds)
        if url_index < len(urls) - 1:
            logging.info("Falling back to alternate Overpass endpoint %s", urls[url_index + 1])

    if last_error is not None:
        raise last_error
    raise RuntimeError("Failed to download OSM data and no error was captured")


def element_to_geometry(element: dict) -> Optional[Any]:
    LineString, _, Polygon, _ = get_shapely_geometry()
    geometry = element.get("geometry")
    if not geometry:
        return None
    coords = [(point["lon"], point["lat"]) for point in geometry]
    if len(coords) < 2:
        return None

    tags = element.get("tags", {})
    is_area = False
    if coords[0] == coords[-1] and len(coords) >= 4:
        is_area = True
    if tags.get("area") == "yes":
        is_area = True
    if tags.get("building") or tags.get("landuse") or tags.get("natural") in {
        "water",
        "wood",
        "scrub",
        "grassland",
        "wetland",
    } or tags.get("leisure"):
        is_area = True

    if is_area:
        if coords[0] != coords[-1]:
            coords.append(coords[0])
        polygon = Polygon(coords)
        if not polygon.is_valid:
            polygon = polygon.buffer(0)
        return polygon if polygon.is_valid and not polygon.is_empty else None

    line = LineString(coords)
    return line if line.is_valid and not line.is_empty else None


def classify_semantic(tags: Dict[str, str]) -> Optional[str]:
    if "building" in tags:
        return "building"

    natural = tags.get("natural")
    if natural in {"water", "wetland", "bay", "river", "stream"}:
        return "water"
    if natural in {"wood", "scrub", "grassland", "heath", "tree", "tree_row"}:
        return "vegetation"

    landuse = tags.get("landuse")
    if landuse in {"forest", "meadow", "grass", "vineyard", "orchard", "allotments"}:
        return "vegetation"
    if landuse in {"reservoir", "pond", "basin"}:
        return "water"

    leisure = tags.get("leisure")
    if leisure in {"park", "garden", "golf_course", "nature_reserve"}:
        return "vegetation"

    highway = tags.get("highway")
    if highway:
        if highway in {"motorway", "trunk", "motorway_link", "trunk_link", "primary", "primary_link"}:
            return "traffic_road"
        if highway in {
            "secondary",
            "secondary_link",
            "tertiary",
            "tertiary_link",
            "residential",
            "living_street",
            "service",
            "unclassified",
            "road",
        }:
            return "road"
        if highway in {"footway", "path", "cycleway", "bridleway", "steps"}:
            return "ground"

    waterway = tags.get("waterway")
    if waterway in {"river", "stream", "canal"}:
        return "water"

    return None


def explode_geometries(geom) -> Iterable[Any]:
    LineString, _, Polygon, _ = get_shapely_geometry()
    if geom.is_empty:
        return []
    geom_type = geom.geom_type
    if geom_type in {"Polygon", "LineString"}:
        return [geom]
    if geom_type == "MultiPolygon" or geom_type == "GeometryCollection":
        return [g for g in geom.geoms if isinstance(g, Polygon) and not g.is_empty]
    if geom_type == "MultiLineString":
        return [g for g in geom.geoms if isinstance(g, LineString) and not g.is_empty]
    return []


# ---------------------------------------------------------------------------
# Rasterization
# ---------------------------------------------------------------------------

def rasterize_semantics(
    class_geometries: Dict[str, List[Tuple[Any, Dict]]],
    resolution: float,
    radius_m: float,
    output_path: Path,
) -> Path:
    Image, ImageDraw = get_pillow()
    LineString, _, Polygon, _ = get_shapely_geometry()
    size = int(math.ceil((radius_m * 2) / resolution))
    if size <= 0:
        raise ValueError("Computed raster size is non-positive")

    x_min = -radius_m
    y_min = -radius_m
    x_max = radius_m
    y_max = radius_m

    image = Image.new("L", (size, size), CLASS_TO_ID["ground"])
    draw = ImageDraw.Draw(image)

    def to_pixel(pt: Tuple[float, float]) -> Tuple[float, float]:
        x, y = pt
        col = (x - x_min) / resolution
        row = (y_max - y) / resolution
        return col, row

    clipping_square = Polygon([(x_min, y_min), (x_max, y_min), (x_max, y_max), (x_min, y_max)])

    for class_name in DRAW_ORDER:
        entries = class_geometries.get(class_name, [])
        if not entries:
            continue
        class_id = CLASS_TO_ID[class_name]
        for geom, metadata in entries:
            clipped = geom.intersection(clipping_square)
            if clipped.is_empty:
                continue
            if isinstance(clipped, LineString):
                coords = [to_pixel(pt) for pt in clipped.coords]
                width_m = metadata.get("width_m", ROAD_WIDTHS["default"])
                if class_name == "traffic_road":
                    width_m = metadata.get("width_m", ROAD_WIDTHS["traffic"])
                width_px = max(1, int(round(width_m / resolution)))
                draw.line(coords, fill=class_id, width=width_px)
            else:
                for part in explode_geometries(clipped):
                    if isinstance(part, LineString):
                        coords = [to_pixel(pt) for pt in part.coords]
                        width_px = max(1, int(round(metadata.get("width_m", ROAD_WIDTHS["default"]) / resolution)))
                        draw.line(coords, fill=class_id, width=width_px)
                        continue
                    if not isinstance(part, Polygon):
                        continue
                    exterior = [to_pixel(pt) for pt in part.exterior.coords]
                    draw.polygon(exterior, fill=class_id)
                    min_hole_area = (resolution ** 2) * MIN_BUILDING_HOLE_PIXELS
                    for interior in part.interiors:
                        if class_name == "building":
                            hole_polygon = Polygon(interior)
                            if not hole_polygon.is_valid:
                                hole_polygon = hole_polygon.buffer(0)
                            hole_area = abs(hole_polygon.area) if not hole_polygon.is_empty else 0.0
                            if hole_area < min_hole_area:
                                continue
                        hole = [to_pixel(pt) for pt in interior.coords]
                        draw.polygon(hole, fill=CLASS_TO_ID["ground"])

    np = get_numpy()
    array = np.array(image, dtype=np.uint8)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, array)
    Image.fromarray(array, mode="L").save(output_path.with_suffix(".png"))

    # Create a colored visualization using the fixed palette.
    max_class_id = max(CLASS_TO_ID.values())
    palette = np.zeros((max_class_id + 1, 3), dtype=np.uint8)
    for class_name, class_id in CLASS_TO_ID.items():
        palette[class_id] = SEMANTIC_COLOR_PALETTE[class_name]
    colored = palette[array]
    Image.fromarray(colored, mode="RGB").save(output_path.with_name("semantic_map_colored.png"))
    return output_path


# ---------------------------------------------------------------------------
# Building metadata
# ---------------------------------------------------------------------------

def build_building_features(
    buildings: List[dict],
    projector,
    radius_m: float,
) -> List[dict]:
    _, _, Polygon, mapping = get_shapely_geometry()
    features = []
    extent = Polygon(
        [
            (-radius_m, -radius_m),
            (radius_m, -radius_m),
            (radius_m, radius_m),
            (-radius_m, radius_m),
        ]
    )
    for element in buildings:
        geom = element_to_geometry(element)
        if geom is None or not isinstance(geom, Polygon):
            continue
        projected = projector(geom)
        if projected.is_empty:
            continue
        intersection = projected.intersection(extent)
        if intersection.is_empty:
            continue

        tags = element.get("tags", {})
        primary, secondary, tertiary, sources = classify_building(tags)
        properties = {
            "primary_label": primary,
            "secondary_label": secondary,
            "tertiary_label": tertiary,
            "source_tags": sources,
        }
        for key in ["building:levels", "height", "levels", "name", "addr:housenumber", "addr:street"]:
            if key in tags:
                properties[key.replace(":", "_")] = tags[key]

        properties["area_m2"] = float(intersection.area)
        centroid_geom = intersection.centroid if not intersection.is_empty else geom.centroid
        centroid = centroid_geom
        properties["centroid_lat"] = centroid.y
        properties["centroid_lon"] = centroid.x

        feature = {
            "type": "Feature",
            "id": str(element.get("id")),
            "properties": properties,
            "geometry": mapping(geom),
        }
        features.append(feature)
    return features


# ---------------------------------------------------------------------------
# Main execution flow
# ---------------------------------------------------------------------------

def run_generation(
    lat: float,
    lon: float,
    radius_m: float,
    resolution: float,
    output_dir: Path,
    overpass_url: str,
    requests_module,
) -> None:
    logging.info("Computing bounding box and projection")
    bbox = compute_bbox(lat, lon, radius_m)
    projector = latlon_to_local_projector(lat, lon)

    logging.info("Downloading OSM data from %s", overpass_url)
    elements = download_osm(bbox, overpass_url, requests_module)
    logging.info("Received %d elements", len(elements))

    class_geometries: Dict[str, List[Tuple[Any, Dict]]] = {c: [] for c in CLASS_TO_ID}
    buildings: List[dict] = []

    for element in elements:
        geom = element_to_geometry(element)
        if geom is None:
            continue
        tags = element.get("tags", {})
        projected = projector(geom)
        if projected.is_empty:
            continue

        semantic_class = classify_semantic(tags)
        if semantic_class in {"road", "traffic_road"}:
            width = ROAD_WIDTHS["traffic" if semantic_class == "traffic_road" else "default"]
            class_geometries[semantic_class].append((projected, {"width_m": width}))
        elif semantic_class and semantic_class in class_geometries:
            class_geometries[semantic_class].append((projected, {}))

        if "building" in tags:
            buildings.append(element)

    raster_output = output_dir / "semantic_map.npy"
    logging.info("Rasterizing semantic map to %s", raster_output)
    rasterize_semantics(class_geometries, resolution, radius_m, raster_output)

    logging.info("Building metadata GeoJSON")
    features = build_building_features(buildings, projector, radius_m)
    geojson = {
        "type": "FeatureCollection",
        "features": features,
    }
    buildings_path = output_dir / "buildings.geojson"
    buildings_path.parent.mkdir(parents=True, exist_ok=True)
    with buildings_path.open("w", encoding="utf-8") as f:
        json.dump(geojson, f, ensure_ascii=False, indent=2)

    metadata = {
        "center": {"lat": lat, "lon": lon},
        "radius_m": radius_m,
        "resolution_m": resolution,
        "bbox": {
            "south": bbox[0],
            "west": bbox[1],
            "north": bbox[2],
            "east": bbox[3],
        },
        "class_to_id": CLASS_TO_ID,
        "semantic_map_path": str(raster_output),
        "semantic_map_preview": str(raster_output.with_suffix(".png")),
        "buildings_geojson": str(buildings_path),
        "overpass_url": overpass_url,
        "element_count": len(elements),
        "building_count": len(features),
    }
    metadata_path = output_dir / "metadata.json"
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    logging.info("Generation complete: %s", metadata_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate semantic map and building taxonomy from OSM data.")
    parser.add_argument("lat", type=float, help="Center latitude in decimal degrees")
    parser.add_argument("lon", type=float, help="Center longitude in decimal degrees")
    parser.add_argument(
        "radius",
        type=float,
        help="Half side length of the square region in meters",
    )
    parser.add_argument(
        "--resolution",
        type=float,
        default=1.0,
        help="Raster resolution in meters per pixel (default: 1.0)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("./semantic_dataset"),
        help="Output directory for generated artifacts",
    )
    parser.add_argument(
        "--overpass-url",
        type=str,
        default="https://overpass-api.de/api/interpreter",
        help="Overpass API endpoint",
    )
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level (default: INFO)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))
    requests_module = get_requests()
    try:
        run_generation(
            lat=args.lat,
            lon=args.lon,
            radius_m=args.radius,
            resolution=args.resolution,
            output_dir=args.output,
            overpass_url=args.overpass_url,
            requests_module=requests_module,
        )
    except requests_module.HTTPError as exc:
        logging.error("HTTP error while contacting Overpass API: %s", exc)
        raise
    except requests_module.RequestException as exc:
        logging.error("Network error while contacting Overpass API: %s", exc)
        raise


if __name__ == "__main__":
    main()
