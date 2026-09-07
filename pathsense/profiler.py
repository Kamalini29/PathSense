"""
PathSense Stage-by-Stage Latency Profiler
Measures execution times for capture, depth estimation, hazard detection,
risk fusion, decision logic, and spatial audio synthesis.
"""

from contextlib import contextmanager
import time
from typing import Dict, NamedTuple


class LatencyBreakdown(NamedTuple):
    """Container for stage latency measurements in milliseconds."""
    capture_ms: float
    depth_ms: float
    detect_ms: float
    fusion_ms: float
    decision_ms: float
    audio_ms: float
    total_ms: float
    fps: float


class PipelineProfiler:
    """
    High-precision microsecond profiler for PathSense processing pipeline stages.
    """

    def __init__(self):
        self.stage_times: Dict[str, float] = {
            "capture": 0.0,
            "depth": 0.0,
            "detect": 0.0,
            "fusion": 0.0,
            "decision": 0.0,
            "audio": 0.0,
        }

    def reset(self):
        """Resets recorded timing values."""
        for key in self.stage_times:
            self.stage_times[key] = 0.0

    @contextmanager
    def profile_stage(self, stage_name: str):
        """Context manager to measure latency of a named pipeline stage."""
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            if stage_name in self.stage_times:
                self.stage_times[stage_name] = elapsed_ms

    def get_breakdown(self) -> LatencyBreakdown:
        """Computes summary latency breakdown and total FPS."""
        total_ms = sum(self.stage_times.values())
        fps = (1000.0 / total_ms) if total_ms > 0 else 0.0

        return LatencyBreakdown(
            capture_ms=round(self.stage_times.get("capture", 0.0), 2),
            depth_ms=round(self.stage_times.get("depth", 0.0), 2),
            detect_ms=round(self.stage_times.get("detect", 0.0), 2),
            fusion_ms=round(self.stage_times.get("fusion", 0.0), 2),
            decision_ms=round(self.stage_times.get("decision", 0.0), 2),
            audio_ms=round(self.stage_times.get("audio", 0.0), 2),
            total_ms=round(total_ms, 2),
            fps=round(fps, 1)
        )
