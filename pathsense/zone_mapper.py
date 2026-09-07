"""
PathSense Depth-to-Zone Mapping Module
Pools dense per-pixel depth maps into a 3x3 navigation grid using percentile pooling.
"""

from typing import Dict, List, NamedTuple, Tuple
import numpy as np

from .config import ZoneConfig, ZONE_NAMES


class ZoneMetrics(NamedTuple):
    """Holds aggregated metrics for a single 3x3 zone."""
    name: str
    row: int
    col: int
    raw_value: float       # Representative pooled depth value
    normalized_closeness: float # [0.0, 1.0] where 1.0 = closest / maximum proximity
    pixel_count: int


class ZoneGridResult(NamedTuple):
    """Container for full 3x3 grid pooling output."""
    grid_matrix: np.ndarray          # 3x3 matrix of normalized closeness [0, 1]
    zone_dict: Dict[str, ZoneMetrics] # Mapping from zone name to metrics
    raw_depth_map: np.ndarray        # Original input depth map
    min_depth: float                 # Global depth min
    max_depth: float                 # Global depth max


class ZoneMapper:
    """
    Transforms a 2D monocular depth map (H x W) into a 3x3 zone matrix.
    Uses percentile pooling to detect safety-critical narrow obstacles.
    """

    ROW_NAMES = ["top", "mid", "bottom"]
    COL_NAMES = ["left", "center", "right"]

    def __init__(self, config: ZoneConfig = None):
        self.config = config or ZoneConfig()

    def process(self, depth_map: np.ndarray) -> ZoneGridResult:
        """
        Pools dense depth map into 3x3 grid.
        
        Args:
            depth_map: 2D numpy array (H, W). Higher values = closer objects (closeness/disparity format)
                       or normalized depth.
                       
        Returns:
            ZoneGridResult with 3x3 closeness matrix and per-zone detailed metrics.
        """
        if depth_map.ndim != 2:
            raise ValueError(f"Expected 2D depth map (H, W), got shape {depth_map.shape}")

        h, w = depth_map.shape
        min_val = float(np.min(depth_map))
        max_val = float(np.max(depth_map))

        # Normalize depth map to [0, 1] range safely
        denom = (max_val - min_val) if (max_val - min_val) > 1e-6 else 1.0
        norm_depth = (depth_map - min_val) / denom

        row_edges = np.linspace(0, h, self.config.rows + 1, dtype=int)
        col_edges = np.linspace(0, w, self.config.cols + 1, dtype=int)

        grid_matrix = np.zeros((self.config.rows, self.config.cols), dtype=np.float32)
        zone_dict: Dict[str, ZoneMetrics] = {}

        idx = 0
        for r in range(self.config.rows):
            for c in range(self.config.cols):
                r_start, r_end = row_edges[r], row_edges[r + 1]
                c_start, c_end = col_edges[c], col_edges[c + 1]

                patch = norm_depth[r_start:r_end, c_start:c_end]
                raw_patch = depth_map[r_start:r_end, c_start:c_end]

                if patch.size == 0:
                    pooled_norm = 0.0
                    raw_val = 0.0
                else:
                    # High closeness values = closer obstacle.
                    # 90th percentile of closeness (or 100 - depth_percentile) gives conservative close obstacle measure.
                    target_percentile = 100.0 - self.config.depth_percentile
                    pooled_norm = float(np.percentile(patch, target_percentile))
                    raw_val = float(np.percentile(raw_patch, target_percentile))

                grid_matrix[r, c] = pooled_norm
                zone_name = ZONE_NAMES[idx] if idx < len(ZONE_NAMES) else f"{self.ROW_NAMES[r]}_{self.COL_NAMES[c]}"

                zone_dict[zone_name] = ZoneMetrics(
                    name=zone_name,
                    row=r,
                    col=c,
                    raw_value=raw_val,
                    normalized_closeness=pooled_norm,
                    pixel_count=patch.size
                )
                idx += 1

        return ZoneGridResult(
            grid_matrix=grid_matrix,
            zone_dict=zone_dict,
            raw_depth_map=depth_map,
            min_depth=min_val,
            max_depth=max_val
        )
