"""
PathSense Unified Pipeline Module
Coordinates Monocular Depth, Hazard Detection, Zone Pooling, Risk Fusion,
Decision Hysteresis, Spatial Audio, Profiling, and Visual Overlay.
"""

from dataclasses import dataclass
from typing import Dict, List, NamedTuple, Tuple
import cv2
import numpy as np

from .config import SystemConfig
from .decision_engine import ActionDecision, DecisionEngine, NavigationAction
from .depth_estimator import DepthEstimator
from .object_detector import ObjectDetector
from .profiler import LatencyBreakdown, PipelineProfiler
from .risk_fusion import DetectedObject, RiskFusionEngine, RiskFusionResult
from .spatial_audio import AudioSignalParams, SpatialAudioEngine
from .zone_mapper import ZoneGridResult, ZoneMapper


class PipelineOutput(NamedTuple):
    """Container for complete frame processing results."""
    annotated_frame: np.ndarray        # Frame with visual overlays
    depth_map: np.ndarray              # 2D normalized depth map
    depth_heatmap: np.ndarray          # Colorized depth heatmap (JET colormap)
    zone_result: ZoneGridResult        # 3x3 zone pooling result
    fusion_result: RiskFusionResult    # 3x3 fused risk matrix result
    decision: ActionDecision           # Navigation decision
    audio_params: AudioSignalParams    # Spatial audio panning & tone parameters
    latency: LatencyBreakdown          # Stage-by-stage latency profile


class PathSensePipeline:
    """
    Main orchestration class for PathSense real-time navigation assistant.
    """

    def __init__(
        self,
        config: SystemConfig = None,
        force_synthetic: bool = False
    ):
        self.config = config or SystemConfig()
        self.force_synthetic = force_synthetic

        # Initialize core engine modules
        self.depth_estimator = DepthEstimator(self.config, force_synthetic=force_synthetic)
        self.object_detector = ObjectDetector(self.config, force_synthetic=force_synthetic)
        self.zone_mapper = ZoneMapper(self.config.zone)
        self.risk_fusion = RiskFusionEngine(self.config.fusion, self.config.hazard_weights)
        self.decision_engine = DecisionEngine(self.config.decision, self.config.fusion)
        self.spatial_audio = SpatialAudioEngine(self.config.audio)
        self.profiler = PipelineProfiler()

    def process_frame(
        self,
        frame: np.ndarray,
        trigger_audio: bool = True
    ) -> PipelineOutput:
        """
        Executes the full end-to-end PathSense navigation pipeline on a video frame.
        
        Args:
            frame: numpy array (H, W, 3) BGR image from webcam or video file.
            trigger_audio: If True, plays spoken TTS alerts on action change.
            
        Returns:
            PipelineOutput containing annotated visualization frame, depth map, 3x3 risk matrix, action, and latency.
        """
        if frame is None or frame.size == 0:
            raise ValueError("Input video frame is empty or None")

        h, w = frame.shape[:2]
        self.profiler.reset()

        # 1. Capture & Prep Stage
        with self.profiler.profile_stage("capture"):
            prep_frame = frame.copy()

        # 2. Monocular Depth Estimation Stage
        with self.profiler.profile_stage("depth"):
            depth_map = self.depth_estimator.estimate_depth(prep_frame)

        # 3. Object Hazard Detection Stage
        with self.profiler.profile_stage("detect"):
            detected_objects = self.object_detector.detect(prep_frame)

        # 4. Depth-to-Zone Percentile Pooling Stage
        with self.profiler.profile_stage("fusion"):
            zone_result = self.zone_mapper.process(depth_map)
            fusion_result = self.risk_fusion.process(zone_result, detected_objects, (h, w))

        # 5. Safe-Direction Decision Engine Stage
        with self.profiler.profile_stage("decision"):
            decision = self.decision_engine.process(fusion_result)

        # 6. Spatial Audio & TTS Stage
        with self.profiler.profile_stage("audio"):
            audio_params = self.spatial_audio.compute_spatial_params(decision)
            if trigger_audio and decision.is_state_changed:
                self.spatial_audio.speak(decision.spoken_phrase, asynchronous=True)

        latency = self.profiler.get_breakdown()

        # 7. Render Visualization & Overlays
        depth_heatmap, annotated_frame = self.render_visualization(
            frame, depth_map, fusion_result, decision, latency, detected_objects
        )

        return PipelineOutput(
            annotated_frame=annotated_frame,
            depth_map=depth_map,
            depth_heatmap=depth_heatmap,
            zone_result=zone_result,
            fusion_result=fusion_result,
            decision=decision,
            audio_params=audio_params,
            latency=latency
        )

    def render_visualization(
        self,
        frame: np.ndarray,
        depth_map: np.ndarray,
        fusion_result: RiskFusionResult,
        decision: ActionDecision,
        latency: LatencyBreakdown,
        detected_objects: List[DetectedObject]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Draws depth heatmaps, bounding boxes, 3x3 risk grid, action banner, and performance HUD.
        """
        h, w = frame.shape[:2]
        canvas = frame.copy()

        # 1. Colorize depth map (JET colormap: Red = Close, Blue = Far)
        depth_uint8 = (depth_map * 255.0).clip(0, 255).astype(np.uint8)
        depth_heatmap = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_JET)

        # 2. Draw 3x3 Risk Grid Lines & Risk Overlay on Canvas
        grid_rows, grid_cols = fusion_result.risk_matrix.shape
        row_step = h / float(grid_rows)
        col_step = w / float(grid_cols)

        overlay = canvas.copy()
        for r in range(grid_rows):
            for c in range(grid_cols):
                risk = fusion_result.risk_matrix[r, c]
                x1, y1 = int(c * col_step), int(r * row_step)
                x2, y2 = int((c + 1) * col_step), int((r + 1) * row_step)

                # Color coding: Green (Low Risk), Yellow (Medium), Red (High Risk)
                if risk >= self.config.fusion.critical_risk_threshold:
                    color = (0, 0, 220)   # Red
                elif risk >= self.config.fusion.medium_risk_threshold:
                    color = (0, 200, 255) # Yellow
                else:
                    color = (0, 180, 0)   # Green

                # Draw semi-transparent zone fill
                cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
                # Draw grid lines
                cv2.rectangle(canvas, (x1, y1), (x2, y2), (220, 220, 220), 1)

                # Zone risk text
                score_str = f"{risk:.2f}"
                cv2.putText(canvas, score_str, (x1 + 10, y1 + 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

        # Blend overlay (20% opacity fill)
        cv2.addWeighted(overlay, 0.22, canvas, 0.78, 0, canvas)

        # 3. Draw Detected Objects Bounding Boxes
        for obj in detected_objects:
            ox1, oy1, ox2, oy2 = obj.bbox
            px1, py1 = int(ox1 * w), int(oy1 * h)
            px2, py2 = int(ox2 * w), int(oy2 * h)

            label = f"{obj.class_name.upper()} {int(obj.confidence * 100)}%"
            cv2.rectangle(canvas, (px1, py1), (px2, py2), (255, 120, 0), 2)
            cv2.rectangle(canvas, (px1, max(0, py1 - 22)), (px1 + len(label) * 9, py1), (255, 120, 0), -1)
            cv2.putText(canvas, label, (px1 + 4, max(14, py1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

        # 4. Top Action Banner HUD
        banner_height = 50
        banner_bg = np.zeros((banner_height, w, 3), dtype=np.uint8)
        
        if decision.action == NavigationAction.STOP_HAZARD:
            banner_color = (0, 0, 230)      # Bright Red
        elif decision.action in (NavigationAction.MOVE_LEFT, NavigationAction.MOVE_RIGHT):
            banner_color = (0, 140, 255)    # Orange
        elif decision.action in (NavigationAction.SLIGHT_LEFT, NavigationAction.SLIGHT_RIGHT):
            banner_color = (0, 210, 255)    # Yellow
        else:
            banner_color = (0, 160, 0)      # Green

        cv2.rectangle(banner_bg, (0, 0), (w, banner_height), banner_color, -1)
        action_text = f"ACTION: {decision.spoken_phrase.upper()}"
        cv2.putText(banner_bg, action_text, (15, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

        # Overlay banner at top of canvas
        canvas[0:banner_height, 0:w] = banner_bg

        # 5. Bottom Latency HUD
        hud_str = f"FPS: {latency.fps:.1f} | Total: {latency.total_ms:.1f}ms (Depth: {latency.depth_ms}ms, Detect: {latency.detect_ms}ms, Fusion: {latency.fusion_ms}ms)"
        cv2.putText(canvas, hud_str, (10, h - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (240, 240, 240), 1, cv2.LINE_AA)

        return depth_heatmap, canvas
