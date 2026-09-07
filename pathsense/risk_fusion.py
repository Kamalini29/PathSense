"""
PathSense Risk Fusion Module
Implements confidence-weighted urgency fusion combining 3x3 zone depth map,
detected object hazard weights, and bounding box zone intersections.
"""

from dataclasses import dataclass
from typing import Dict, List, NamedTuple, Tuple
import numpy as np

from .config import DEFAULT_HAZARD_WEIGHTS, RiskFusionConfig, ZONE_NAMES
from .zone_mapper import ZoneGridResult


@dataclass
class DetectedObject:
    """Represents an object detected by the YOLO detector."""
    class_id: int
    class_name: str
    confidence: float
    bbox: Tuple[float, float, float, float]  # Normalized coordinates [x1, y1, x2, y2] in [0, 1]
    hazard_weight: float


@dataclass
class ZoneRiskDetail:
    """Detailed risk breakdown for a single zone."""
    zone_name: str
    row: int
    col: int
    depth_risk: float
    object_risk: float
    fused_risk: float
    dominant_object: str = None
    dominant_confidence: float = 0.0


class RiskFusionResult(NamedTuple):
    """Container for the fused 3x3 risk assessment."""
    risk_matrix: np.ndarray             # 3x3 matrix of fused risk scores in [0.0, 1.0]
    zone_risks: Dict[str, ZoneRiskDetail] # Named zone details
    detected_objects: List[DetectedObject] # Active detected objects
    max_risk_zone: str                  # Name of zone with highest risk
    max_risk_score: float               # Highest risk score across grid


class RiskFusionEngine:
    """
    Fuses dense monocular depth zone metrics and detected object hazards
    into a unified 3x3 risk score grid.
    """

    def __init__(self, config: RiskFusionConfig = None, hazard_weights: Dict[str, float] = None):
        self.config = config or RiskFusionConfig()
        self.hazard_weights = hazard_weights or DEFAULT_HAZARD_WEIGHTS

    def get_hazard_weight(self, class_name: str) -> float:
        """Returns the hazard score for a detected object class name."""
        clean_name = class_name.lower().strip()
        return self.hazard_weights.get(clean_name, self.config.default_hazard_weight)

    def process(
        self,
        zone_result: ZoneGridResult,
        detected_objects: List[DetectedObject],
        image_shape: Tuple[int, int] = (480, 640)
    ) -> RiskFusionResult:
        """
        Fuses zone depth matrix with detected object bounding boxes.
        
        Args:
            zone_result: ZoneGridResult output from ZoneMapper.
            detected_objects: List of DetectedObject instances.
            image_shape: Tuple of (height, width) for coordinate scaling.
            
        Returns:
            RiskFusionResult containing 3x3 risk matrix and zone details.
        """
        rows, cols = zone_result.grid_matrix.shape
        risk_matrix = np.zeros((rows, cols), dtype=np.float32)
        zone_risks: Dict[str, ZoneRiskDetail] = {}

        # 3x3 zone coordinate boundaries in normalized [0, 1]
        row_bounds = np.linspace(0.0, 1.0, rows + 1)
        col_bounds = np.linspace(0.0, 1.0, cols + 1)

        idx = 0
        max_score = -1.0
        max_zone = ""

        for r in range(rows):
            r_min, r_max = row_bounds[r], row_bounds[r + 1]
            for c in range(cols):
                c_min, c_max = col_bounds[c], col_bounds[c + 1]

                zone_name = ZONE_NAMES[idx] if idx < len(ZONE_NAMES) else f"r{r}_c{c}"
                depth_risk = float(zone_result.grid_matrix[r, c])

                # Compute maximum object urgency intersecting this zone patch
                obj_risk = 0.0
                dominant_obj_name = None
                dominant_obj_conf = 0.0

                for obj in detected_objects:
                    ox1, oy1, ox2, oy2 = obj.bbox

                    # Calculate Intersection Over Zone Patch Area
                    ix1 = max(c_min, ox1)
                    iy1 = max(r_min, oy1)
                    ix2 = min(c_max, ox2)
                    iy2 = min(r_max, oy2)

                    if ix2 > ix1 and iy2 > iy1:
                        inter_area = (ix2 - ix1) * (iy2 - iy1)
                        zone_area = (c_max - c_min) * (r_max - r_min)
                        obj_area = (ox2 - ox1) * (oy2 - oy1)

                        # Fraction of object covered inside zone
                        overlap_fraction = inter_area / (obj_area + 1e-6)

                        # Confidence-weighted urgency score
                        urgency = obj.hazard_weight * obj.confidence * min(1.0, overlap_fraction * 1.5)

                        if urgency > obj_risk:
                            obj_risk = urgency
                            dominant_obj_name = obj.class_name
                            dominant_obj_conf = obj.confidence

                # Confidence-weighted urgency fusion formula
                fused = (
                    self.config.depth_weight * depth_risk +
                    self.config.object_weight * obj_risk
                )
                fused_clipped = float(np.clip(fused, 0.0, 1.0))

                risk_matrix[r, c] = fused_clipped

                detail = ZoneRiskDetail(
                    zone_name=zone_name,
                    row=r,
                    col=c,
                    depth_risk=depth_risk,
                    object_risk=obj_risk,
                    fused_risk=fused_clipped,
                    dominant_object=dominant_obj_name,
                    dominant_confidence=dominant_obj_conf
                )
                zone_risks[zone_name] = detail

                if fused_clipped > max_score:
                    max_score = fused_clipped
                    max_zone = zone_name

                idx += 1

        return RiskFusionResult(
            risk_matrix=risk_matrix,
            zone_risks=zone_risks,
            detected_objects=detected_objects,
            max_risk_zone=max_zone,
            max_risk_score=max_score
        )
