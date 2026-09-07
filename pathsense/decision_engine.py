"""
PathSense Safe-Direction Decision Engine & Hysteresis State Machine
Converts 3x3 fused zone risk matrix into temporal-smoothed navigation actions.
"""

from enum import Enum
from typing import Dict, List, NamedTuple, Optional, Tuple
import numpy as np

from .config import DecisionConfig, RiskFusionConfig
from .risk_fusion import RiskFusionResult


class NavigationAction(str, Enum):
    """Possible movement instructions output by PathSense."""
    PROCEED_CLEAR = "PROCEED_CLEAR"
    SLIGHT_LEFT   = "SLIGHT_LEFT"
    MOVE_LEFT     = "MOVE_LEFT"
    SLIGHT_RIGHT  = "SLIGHT_RIGHT"
    MOVE_RIGHT    = "MOVE_RIGHT"
    STOP_HAZARD   = "STOP_HAZARD"


class ActionDecision(NamedTuple):
    """Container for the current frame's navigation recommendation."""
    action: NavigationAction
    raw_action: NavigationAction
    confidence: float
    spoken_phrase: str
    left_risk: float
    center_risk: float
    right_risk: float
    is_state_changed: bool


class DecisionEngine:
    """
    Evaluates 3x3 risk score grid to produce safe movement actions.
    Applies Exponential Moving Average (EMA) and state transition hysteresis
    to prevent frame-to-frame action flicker.
    """

    # Human-readable audio messages mapped to actions
    ACTION_PHRASES: Dict[NavigationAction, str] = {
        NavigationAction.PROCEED_CLEAR: "Path clear, proceed forward",
        NavigationAction.SLIGHT_LEFT:   "Obstacle ahead right, steer slightly left",
        NavigationAction.MOVE_LEFT:     "Obstacle center ahead, move left",
        NavigationAction.SLIGHT_RIGHT:  "Obstacle ahead left, steer slightly right",
        NavigationAction.MOVE_RIGHT:    "Obstacle center ahead, move right",
        NavigationAction.STOP_HAZARD:   "Warning! Obstacle immediately ahead, stop",
    }

    def __init__(self, decision_config: DecisionConfig = None, fusion_config: RiskFusionConfig = None):
        self.config = decision_config or DecisionConfig()
        self.fusion_config = fusion_config or RiskFusionConfig()

        # State memory for Exponential Moving Average (3x3 grid)
        self.smoothed_risk_matrix: Optional[np.ndarray] = None

        # State memory for Hysteresis
        self.current_action: NavigationAction = NavigationAction.PROCEED_CLEAR
        self.candidate_action: NavigationAction = NavigationAction.PROCEED_CLEAR
        self.candidate_count: int = 0

    def reset_state(self):
        """Resets temporal memory across video stream boundaries."""
        self.smoothed_risk_matrix = None
        self.current_action = NavigationAction.PROCEED_CLEAR
        self.candidate_action = NavigationAction.PROCEED_CLEAR
        self.candidate_count = 0

    def process(self, fusion_result: RiskFusionResult) -> ActionDecision:
        """
        Evaluates risk fusion output and returns temporal-smoothed action recommendation.
        
        Args:
            fusion_result: RiskFusionResult from RiskFusionEngine.
            
        Returns:
            ActionDecision containing current action, spoken phrase, and left/center/right risk scores.
        """
        raw_matrix = fusion_result.risk_matrix

        # 1. Apply Exponential Moving Average (EMA) smoothing
        if self.smoothed_risk_matrix is None:
            self.smoothed_risk_matrix = raw_matrix.copy()
        else:
            alpha = self.config.ema_alpha
            self.smoothed_risk_matrix = alpha * raw_matrix + (1.0 - alpha) * self.smoothed_risk_matrix

        matrix = self.smoothed_risk_matrix

        # Column-wise aggregate risks: Left (col 0), Center (col 1), Right (col 2)
        # Give higher weight to Mid and Bottom rows (immediate path of travel)
        row_weights = np.array([0.2, 0.4, 0.4])
        
        left_risk = float(np.average(matrix[:, 0], weights=row_weights))
        center_risk = float(np.average(matrix[:, 1], weights=row_weights))
        right_risk = float(np.average(matrix[:, 2], weights=row_weights))

        # Bottom-center immediate danger check (highest urgency)
        immediate_danger = matrix[2, 1] >= self.fusion_config.critical_risk_threshold or \
                           center_risk >= self.fusion_config.critical_risk_threshold

        # 2. Rule-Based Instantaneous Raw Action Determination
        if immediate_danger:
            raw_action = NavigationAction.STOP_HAZARD
        elif center_risk >= self.fusion_config.high_risk_threshold:
            # Center blocked: determine safer side (Left vs Right)
            if left_risk < right_risk:
                raw_action = NavigationAction.MOVE_LEFT
            else:
                raw_action = NavigationAction.MOVE_RIGHT
        elif center_risk >= self.fusion_config.medium_risk_threshold:
            # Medium center hazard or side imbalance
            if left_risk + self.config.side_imbalance_threshold < right_risk:
                raw_action = NavigationAction.SLIGHT_LEFT
            elif right_risk + self.config.side_imbalance_threshold < left_risk:
                raw_action = NavigationAction.SLIGHT_RIGHT
            elif left_risk < right_risk:
                raw_action = NavigationAction.MOVE_LEFT
            else:
                raw_action = NavigationAction.MOVE_RIGHT
        elif right_risk >= self.fusion_config.high_risk_threshold and left_risk < self.fusion_config.medium_risk_threshold:
            raw_action = NavigationAction.MOVE_LEFT
        elif left_risk >= self.fusion_config.high_risk_threshold and right_risk < self.fusion_config.medium_risk_threshold:
            raw_action = NavigationAction.MOVE_RIGHT
        elif right_risk >= self.fusion_config.medium_risk_threshold:
            raw_action = NavigationAction.SLIGHT_LEFT
        elif left_risk >= self.fusion_config.medium_risk_threshold:
            raw_action = NavigationAction.SLIGHT_RIGHT
        else:
            raw_action = NavigationAction.PROCEED_CLEAR

        # STOP_HAZARD bypasses hysteresis delay for immediate safety
        state_changed = False
        if raw_action == NavigationAction.STOP_HAZARD:
            if self.current_action != NavigationAction.STOP_HAZARD:
                self.current_action = NavigationAction.STOP_HAZARD
                state_changed = True
            self.candidate_action = NavigationAction.STOP_HAZARD
            self.candidate_count = 0
        else:
            # 3. Apply State Transition Hysteresis Counter
            if raw_action == self.candidate_action:
                self.candidate_count += 1
            else:
                self.candidate_action = raw_action
                self.candidate_count = 1

            if self.candidate_count >= self.config.hysteresis_frames:
                if self.current_action != self.candidate_action:
                    self.current_action = self.candidate_action
                    state_changed = True

        spoken = self.ACTION_PHRASES.get(self.current_action, "Proceed with caution")

        # Action confidence based on score margin
        confidence = float(np.clip(1.0 - (center_risk if self.current_action == NavigationAction.PROCEED_CLEAR else (1.0 - max(left_risk, right_risk))), 0.5, 1.0))

        return ActionDecision(
            action=self.current_action,
            raw_action=raw_action,
            confidence=confidence,
            spoken_phrase=spoken,
            left_risk=left_risk,
            center_risk=center_risk,
            right_risk=right_risk,
            is_state_changed=state_changed
        )
