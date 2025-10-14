# Offline Dataset Curation Plan

## Goals
- Generate 1 m/pixel semantic rasters covering arbitrary circular regions provided by latitude, longitude, and radius.
- Enrich every building footprint with multi-level usage labels (e.g., residential → apartment → public housing).
- Support large-scale, offline curation by mirroring or replacing the online dependencies used in the official V-IRL implementation.

## Re-using V-IRL API Clients
The existing `GoogleMapAPI` wrapper already implements the core Google Places & Street View requests, including geocoding, nearby search, place details, photos, and panorama retrieval.【F:virl/platform/google_map_apis.py†L20-L183】【F:virl/platform/platform.py†L8-L87】  Key functions we can reuse:
- `get_geocode_from_address_v2` for precise seeding coordinates, avoiding early drift when tiling a new area.【F:virl/platform/google_map_apis.py†L85-L127】
- `get_nearby_places`/`get_nearby_places_v2` to crawl Place IDs, ratings, categories, review counts, and photo references within a radius, already supporting pagination, distance filters, and polygon clipping via Shapely.【F:virl/platform/google_map_apis.py†L129-L317】
- `get_place_details`-family helpers (`get_place_reviews`, `get_photo_references_from_place_details`) to enrich each Place ID with textual reviews and imagery metadata for downstream labeling models.【F:virl/platform/google_map_apis.py†L626-L717】
- Offline caching hooks (`init_offline`) that read pre-downloaded panoramas and GPS→panorama indexes; we can extend the same pattern to cache JSON responses and orthophotos for repeatable dataset builds.【F:virl/platform/google_map_apis.py†L35-L56】

By instantiating `Platform` in batch mode we can sweep tiles of Street View panoramas for supervision signals (e.g., perspective projections, road context) while respecting the existing mover utilities.【F:virl/platform/platform.py†L8-L87】

## Semantic Raster Generation
1. **Geometry Backbone**: For dense, license-friendly vector data, use OpenStreetMap (OSM) via Overpass API downloads or regional extracts. Query features within the target circle and categorize them into the six map classes (ground, vegetation, water, building, road, traffic road). Suggested tag mapping:
   - `landuse=grass`, `natural=wood`, `leisure=park` → vegetation
   - `natural=water`, `waterway=*` → water
   - `building=*` polygons → building
   - `highway=service/path/footway` → ground
   - `highway=residential/tertiary/...` → road
   - `highway=primary/trunk/motorway` + `junction=roundabout` → traffic road
2. **Rasterization**: Convert clipped OSM geometries to a 1 m grid (e.g., using `rasterio.features.rasterize`). Ensure the raster extent spans the requested radius plus padding so that a user-specified 1 m resolution aligns with EPSG:3857 meters.
3. **Gap Filling**: When OSM coverage is sparse, backfill using government open data (e.g., Microsoft Global ML Building Footprints, USGS 3DEP), or infer ground vs. vegetation using Sentinel-2 landcover classifiers.
4. **Quality Checks**: Compare semantic tiles with Street View samples fetched through `get_streetview_from_geocode` to validate class consistency, leveraging the same heading sampling utilities already defined for agent perception.【F:virl/platform/google_map_apis.py†L318-L360】

## Building Taxonomy Enrichment
1. **Primary/Secondary Labels**: Start from Google Place types returned by nearby search & details; translate the flat type list into a multi-level taxonomy (e.g., `school` → Education → School). Maintain a mapping table (CSV/JSON) for deterministic assignment.
2. **OSM Tags**: Merge building polygons with OSM attributes (`building=*`, `amenity=*`, `shop=*`). These tags often provide direct secondary categories (e.g., `amenity=hospital`).
3. **Auxiliary Signals**:
   - Use Place reviews (`get_place_reviews`) to mine keywords supporting tertiary labels (e.g., “elementary”, “clinic”).【F:virl/platform/google_map_apis.py†L626-L642】
   - Invoke `get_place_photo` on cached references to run scene classification or OCR for signage, assisting in disambiguating mixed-use structures.【F:virl/platform/google_map_apis.py†L678-L717】
   - Aggregate parcel-level datasets (local government open data) where available to append official zoning classes.
4. **Hierarchy Assembly**: Define rules to reconcile conflicts (e.g., OSM says `residential`, Google type says `lodging`). Prioritize authoritative sources and flag uncertain cases for manual QA or LLM-based adjudication.

## Scalable Workflow
1. Tile the area into overlapping circles smaller than Google API’s 50 km cap; use the existing pagination loop to exhaust each cell before advancing.【F:virl/platform/google_map_apis.py†L168-L191】
2. Persist raw API responses, semantic rasters, and derived labels into an offline cache directory keyed by geohash to enable replays via the existing offline hooks.【F:virl/platform/google_map_apis.py†L35-L56】
3. Parallelize the extraction pipeline with rate-limit aware workers; throttle Street View downloads which incur the highest credit usage per the README notes.
4. Produce a final database schema (e.g., Parquet files) with tables for tiles, polygons, building metadata, and imagery references so downstream agents can operate without live API calls.

## Alternative Data Sources
- **OpenStreetMap**: Primary source for geometry and base semantics, accessible without cost limits.
- **Microsoft Building Footprints / Meta Mapillary**: Supplements building geometries in regions with sparse OSM coverage.
- **Open Data Portals**: City/country specific zoning, land-use, and cadastral datasets for higher-quality building classifications.
- **Copernicus/Sentinel-2 Land Cover**: Provides vegetation/water masks for quick raster backfilling when vector data is missing.

Integrating these sources with the V-IRL API modules yields an efficient, repeatable pipeline for large-scale offline map and building datasets without relying exclusively on real-time Google requests.
