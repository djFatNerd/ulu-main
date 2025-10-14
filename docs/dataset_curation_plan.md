# Offline Dataset Curation Plan

## Goals
- Generate 1 m/pixel semantic rasters covering arbitrary square regions provided by latitude, longitude, and half side length.
- Enrich every building footprint with multi-level usage labels (e.g., residential → apartment → public housing).
- Support large-scale, offline curation by mirroring or replacing online data dependencies with reproducible local workflows.

## Replacing Proprietary API Dependencies
Where the original online pipeline depended on proprietary services (e.g., Google Places, Street View), replicate the required capabilities with locally cached responses or third-party open-data equivalents. The replacement module should provide:
- Geocoding utilities for seeding precise coordinates across the area of interest.
- Nearby search functionality that retrieves place identifiers, ratings, categories, review counts, and photo references within a configurable radius, including pagination and geometric clipping.
- Detail endpoints that enrich each place with textual reviews and imagery metadata for downstream labeling models.
- Offline caching hooks that read pre-downloaded panoramas, JSON responses, and orthophotos, enabling repeatable dataset builds without live API calls.

Batch workers can sweep tiles of imagery or panoramic datasets for supervision signals (e.g., perspective projections, road context) while respecting rate limits and local storage constraints.

## Semantic Raster Generation
1. **Geometry Backbone**: For dense, license-friendly vector data, use OpenStreetMap (OSM) via Overpass API downloads or regional extracts. Query features within the target square extent and categorize them into the six map classes (ground, vegetation, water, building, road, traffic road). Suggested tag mapping:
   - `landuse=grass`, `natural=wood`, `leisure=park` → vegetation
   - `natural=water`, `waterway=*` → water
   - `building=*` polygons → building
   - `highway=service/path/footway` → ground
   - `highway=residential/tertiary/...` → road
   - `highway=primary/trunk/motorway` + `junction=roundabout` → traffic road
2. **Rasterization**: Convert clipped OSM geometries to a 1 m grid (e.g., using `rasterio.features.rasterize`). Ensure the raster extent spans the requested half side length plus padding so that a user-specified 1 m resolution aligns with EPSG:3857 meters.
3. **Gap Filling**: When OSM coverage is sparse, backfill using government open data (e.g., Microsoft Global ML Building Footprints, USGS 3DEP), or infer ground vs. vegetation using Sentinel-2 landcover classifiers.
4. **Quality Checks**: Compare semantic tiles with panoramic or ground-truth samples captured in the same region to validate class consistency. Reuse existing heading sampling utilities from the offline cache to ensure viewpoints are comparable.

## Building Taxonomy Enrichment
1. **Primary/Secondary Labels**: Start from Google Place types returned by nearby search & details; translate the flat type list into a multi-level taxonomy (e.g., `school` → Education → School). Maintain a mapping table (CSV/JSON) for deterministic assignment.
2. **OSM Tags**: Merge building polygons with OSM attributes (`building=*`, `amenity=*`, `shop=*`). These tags often provide direct secondary categories (e.g., `amenity=hospital`).
3. **Auxiliary Signals**:
   - Mine review text or local business descriptions to support tertiary labels (e.g., “elementary”, “clinic”).
   - Run scene classification or OCR on cached imagery to disambiguate mixed-use structures.
   - Aggregate parcel-level datasets (local government open data) where available to append official zoning classes.
4. **Hierarchy Assembly**: Define rules to reconcile conflicts (e.g., OSM says `residential`, Google type says `lodging`). Prioritize authoritative sources and flag uncertain cases for manual QA or LLM-based adjudication.

## Scalable Workflow
1. Tile the area into overlapping squares sized to stay within API result limits; exhaust each cell before advancing to maintain deterministic coverage.
2. Persist raw API responses, semantic rasters, and derived labels into an offline cache directory keyed by geohash to enable replays via the same hooks used for the initial download.
3. Parallelize the extraction pipeline with rate-limit aware workers; throttle high-cost downloads (e.g., imagery) according to provider quotas and local bandwidth.
4. Produce a final database schema (e.g., Parquet files) with tables for tiles, polygons, building metadata, and imagery references so downstream agents can operate without live API calls.

## Alternative Data Sources
- **OpenStreetMap**: Primary source for geometry and base semantics, accessible without cost limits.
- **Microsoft Building Footprints / Meta Mapillary**: Supplements building geometries in regions with sparse OSM coverage.
- **Open Data Portals**: City/country specific zoning, land-use, and cadastral datasets for higher-quality building classifications.
- **Copernicus/Sentinel-2 Land Cover**: Provides vegetation/water masks for quick raster backfilling when vector data is missing.

Integrating these sources with the offline API replacements yields an efficient, repeatable pipeline for large-scale offline map and building datasets without relying exclusively on real-time requests.
