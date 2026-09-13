"""
PathSense Configuration Module
Defines default parameters for Depth Estimation, Object Hazard Detection,
Depth-to-Zone Mapping, Risk Scoring Fusion, Decision Logic Hysteresis,
and Spatial Audio Panning.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple


# Default COCO Hazard Class Weights (0.0 = safe/ignored, 1.0 = maximum hazard)
DEFAULT_HAZARD_WEIGHTS: Dict[str, float] = {
    # Vehicles & Heavy Hazards
    "car": 1.0,
    "truck": 1.0,
    "bus": 1.0,
    "motorcycle": 0.95,
    "bicycle": 0.85,
    # Pedestrians & Animals
    "person": 0.9,
    "dog": 0.8,
    "cat": 0.6,
    # Interior Obstacles & Low Hazards
    "chair": 0.75,
    "couch": 0.7,
    "table": 0.7,
    "dining table": 0.7,
    "potted plant": 0.6,
    "tv": 0.5,
    "suitcase": 0.65,
    "backpack": 0.5,
    "door": 0.4,
    "stairs": 1.0,
}

# Standard 3x3 Grid Zone Names
ZONE_NAMES: List[str] = [
    "top_left", "top_center", "top_right",
    "mid_left", "mid_center", "mid_right",
    "bottom_left", "bottom_center", "bottom_right"
]


@dataclass
class ZoneConfig:
    """Config for 3x3 depth map division & percentile pooling."""
    rows: int = 3
    cols: int = 3
    # Percentile to pick per zone (e.g. 10th percentile closest distance)
    # Lower percentile = closer object / conservative hazard safety.
    depth_percentile: float = 10.0
    # Minimum relative depth threshold to consider an obstacle active (0.0 to 1.0)
    min_depth_threshold: float = 0.15


@dataclass
class RiskFusionConfig:
    """Config for confidence-weighted risk fusion formula."""
    # Weight given to depth closeness (1 - normalized_depth)
    depth_weight: float = 0.60
    # Weight given to detected object risk (class_hazard_weight * confidence)
    object_weight: float = 0.40
    # Default hazard weight for unlisted detected objects
    default_hazard_weight: float = 0.5
    # Global risk thresholds for action decisioning
    critical_risk_threshold: float = 0.75
    high_risk_threshold: float = 0.55
    medium_risk_threshold: float = 0.35


@dataclass
class DecisionConfig:
    """Config for safe-direction state machine and temporal hysteresis."""
    # Exponential Moving Average (EMA) smoothing factor across frames (0 < alpha <= 1)
    ema_alpha: float = 0.35
    # Number of consecutive frames an action must be dominant to trigger state change
    hysteresis_frames: int = 3
    # Left/Right risk imbalance threshold to recommend directional shift
    side_imbalance_threshold: float = 0.15


@dataclass
class AudioConfig:
    """Config for spatial audio gain control and TTS output."""
    sample_rate: int = 44100
    base_freq_hz: float = 440.0   # A4 pitch for clear zone
    max_freq_hz: float = 880.0    # A5 pitch for high urgency
    max_beep_tempo_hz: float = 10.0  # Beeps per second at max risk
    min_beep_tempo_hz: float = 1.0   # Beeps per second at low risk
    binaural_panning: bool = True
    enable_tts: bool = True


@dataclass
class SystemConfig:
    """Master configuration container for PathSense."""
    device: str = "cpu"  # "cpu", "cuda", "mps"
    depth_model_name: str = "depth-anything/Depth-Anything-V2-Small-hf"
    detector_model_name: str = "yolov8n.pt"
    hazard_weights: Dict[str, float] = field(default_factory=lambda: DEFAULT_HAZARD_WEIGHTS.copy())
    zone: ZoneConfig = field(default_factory=ZoneConfig)
    fusion: RiskFusionConfig = field(default_factory=RiskFusionConfig)
    decision: DecisionConfig = field(default_factory=DecisionConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
