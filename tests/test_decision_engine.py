"""
Unit tests for PathSense Safe-Direction Decision Engine.
"""

import numpy as np
import pytest

from pathsense.config import DecisionConfig, RiskFusionConfig
from pathsense.decision_engine import DecisionEngine, NavigationAction, ActionDecision
from pathsense.risk_fusion import RiskFusionResult, ZoneRiskDetail


def build_mock_fusion_result(risk_matrix: np.ndarray) -> RiskFusionResult:
    """Helper to generate a mock RiskFusionResult for decision testing."""
    zone_risks = {}
    return RiskFusionResult(
        risk_matrix=risk_matrix,
        zone_risks=zone_risks,
        detected_objects=[],
        max_risk_zone="mid_center",
        max_risk_score=float(np.max(risk_matrix))
    )


def test_decision_engine_proceed_clear():
    engine = DecisionEngine(DecisionConfig(hysteresis_frames=1))
    clear_matrix = np.zeros((3, 3), dtype=np.float32)

    decision = engine.process(build_mock_fusion_result(clear_matrix))

    assert decision.action == NavigationAction.PROCEED_CLEAR
    assert "clear" in decision.spoken_phrase.lower()


def test_decision_engine_stop_hazard_immediate():
    engine = DecisionEngine(DecisionConfig(hysteresis_frames=5))
    hazard_matrix = np.zeros((3, 3), dtype=np.float32)
    hazard_matrix[2, 1] = 0.90  # Critical danger in bottom-center zone

    # STOP_HAZARD should bypass hysteresis counter immediately
    decision = engine.process(build_mock_fusion_result(hazard_matrix))

    assert decision.action == NavigationAction.STOP_HAZARD
    assert decision.is_state_changed is True


def test_decision_engine_directional_steering():
    engine = DecisionEngine(DecisionConfig(hysteresis_frames=1, side_imbalance_threshold=0.1))

    # Center and right blocked -> move left
    blocked_right_matrix = np.array([
        [0.1, 0.6, 0.8],
        [0.1, 0.7, 0.9],
        [0.1, 0.7, 0.9]
    ], dtype=np.float32)

    decision = engine.process(build_mock_fusion_result(blocked_right_matrix))

    assert decision.action in (NavigationAction.MOVE_LEFT, NavigationAction.SLIGHT_LEFT)
    assert decision.left_risk < decision.right_risk


def test_decision_engine_ema_smoothing():
    engine = DecisionEngine(DecisionConfig(ema_alpha=0.5, hysteresis_frames=1))

    m1 = np.zeros((3, 3), dtype=np.float32)
    m2 = np.ones((3, 3), dtype=np.float32)

    engine.process(build_mock_fusion_result(m1))
    engine.process(build_mock_fusion_result(m2))

    # After EMA with alpha 0.5, smoothed matrix should be 0.5
    np.testing.assert_allclose(engine.smoothed_risk_matrix, 0.5, atol=1e-5)
