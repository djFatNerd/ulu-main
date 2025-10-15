"""Utilities for working with OpenStreetMap semantic datasets."""
from .generate_semantic_dataset import (  # noqa: F401
    CLASS_TO_ID,
    DEFAULT_OVERPASS_FALLBACKS,
    DRAW_ORDER,
    ROAD_WIDTHS,
    build_building_features,
    classify_building,
    classify_semantic,
    compute_bbox,
    download_osm,
    element_to_geometry,
    explode_geometries,
    get_numpy,
    get_pillow,
    get_requests,
    get_shapely_geometry,
    get_shapely_transform,
    latlon_to_local_projector,
    local_to_latlon_projector,
    meters_per_degree_lat,
    meters_per_degree_lon,
    rasterize_semantics,
)

# Re-export the module itself for callers that expect the historical layout.
from . import generate_semantic_dataset  # noqa: F401  (re-export for compatibility)
