"""Color palette for semantic map visualizations."""
from __future__ import annotations

from typing import Dict, Tuple

# Mapping of semantic class names to RGB color tuples.
SEMANTIC_COLOR_PALETTE: Dict[str, Tuple[int, int, int]] = {
    "ground": (85, 107, 47),        # Army green
    "vegetation": (34, 139, 34),    # Forest green
    "water": (173, 216, 230),       # Light blue
    "building": (255, 165, 0),      # Orange
    "road": (128, 128, 128),        # Gray
    "traffic_road": (0, 0, 0),      # Black
    "bridge": (218, 165, 32),       # Goldenrod
}
