"""
PathSense Binaural Spatial Audio Engine & Voice Synthesizer
Implements equal-power stereo panning (g_L, g_R), pitch/tempo risk audio,
and pyttsx3 Text-to-Speech output.
"""

from dataclasses import dataclass
import math
import threading
from typing import NamedTuple, Optional, Tuple
import numpy as np

from .config import AudioConfig
from .decision_engine import ActionDecision, NavigationAction


class BinauralGain(NamedTuple):
    """Equal-power stereo gain pair."""
    left_gain: float   # [0.0, 1.0]
    right_gain: float  # [0.0, 1.0]
    azimuth_deg: float # [-90.0, +90.0]


class AudioSignalParams(NamedTuple):
    """Parameters for risk-modulated spatial audio tone generation."""
    frequency_hz: float
    pulse_tempo_hz: float
    left_gain: float
    right_gain: float
    is_warning: bool


class SpatialAudioEngine:
    """
    Computes equal-power binaural gain panning and synthesizes
    spatial audio alert tones & spoken navigation instructions.
    """

    def __init__(self, config: AudioConfig = None):
        self.config = config or AudioConfig()
        self._tts_engine = None
        self._tts_lock = threading.Lock()
        self._last_spoken_phrase: str = ""

        # Attempt lazy initialization of pyttsx3 TTS
        if self.config.enable_tts:
            self._init_tts()

    def _init_tts(self):
        """Initializes pyttsx3 in a thread-safe manner if available."""
        try:
            import pyttsx3
            self._tts_engine = pyttsx3.init()
            self._tts_engine.setProperty("rate", 175) # Moderate reading speed
            self._tts_engine.setProperty("volume", 0.9)
        except Exception as e:
            # Fallback gracefully if pyttsx3 driver is unavailable on environment
            self._tts_engine = None

    @staticmethod
    def calculate_binaural_gains(azimuth_deg: float) -> BinauralGain:
        """
        Calculates constant-power equal loudness stereo panning gains.
        
        Args:
            azimuth_deg: Angle in degrees (-90.0 = Far Left, 0.0 = Center, +90.0 = Far Right).
            
        Returns:
            BinauralGain containing left_gain and right_gain where left^2 + right^2 == 1.
        """
        # Clamp azimuth to [-90, +90]
        az = max(-90.0, min(90.0, float(azimuth_deg)))
        
        # Map [-90, +90] -> [0, pi/2]
        rad = ((az + 90.0) / 180.0) * (math.pi / 2.0)
        
        # Equal-power sine/cosine panning
        left_gain = math.cos(rad)
        right_gain = math.sin(rad)
        
        return BinauralGain(
            left_gain=left_gain,
            right_gain=right_gain,
            azimuth_deg=az
        )

    def compute_spatial_params(self, decision: ActionDecision) -> AudioSignalParams:
        """
        Maps current action decision & risk imbalance to spatial audio parameters.
        
        Args:
            decision: ActionDecision from DecisionEngine.
            
        Returns:
            AudioSignalParams for tone generation and frontend playback.
        """
        # Determine hazard azimuth direction from left/right risk imbalance
        risk_diff = decision.right_risk - decision.left_risk # positive = right riskier
        
        if decision.action in (NavigationAction.MOVE_LEFT, NavigationAction.SLIGHT_LEFT):
            # Obstacle on right -> hazard azimuth on right (+45 to +75 deg)
            azimuth = 45.0 + 30.0 * min(1.0, max(0.0, risk_diff))
        elif decision.action in (NavigationAction.MOVE_RIGHT, NavigationAction.SLIGHT_RIGHT):
            # Obstacle on left -> hazard azimuth on left (-45 to -75 deg)
            azimuth = -45.0 + 30.0 * max(-1.0, min(0.0, risk_diff))
        elif decision.action == NavigationAction.STOP_HAZARD:
            azimuth = 0.0 # Center danger
        else:
            azimuth = 0.0 # Clear

        gains = self.calculate_binaural_gains(azimuth)

        # Max risk score across channels
        max_risk = max(decision.left_risk, decision.center_risk, decision.right_risk)

        # Risk-to-Frequency & Tempo mapping
        freq = self.config.base_freq_hz + max_risk * (self.config.max_freq_hz - self.config.base_freq_hz)
        tempo = self.config.min_beep_tempo_hz + max_risk * (self.config.max_beep_tempo_hz - self.config.min_beep_tempo_hz)

        is_warning = decision.action != NavigationAction.PROCEED_CLEAR

        return AudioSignalParams(
            frequency_hz=freq,
            pulse_tempo_hz=tempo,
            left_gain=gains.left_gain,
            right_gain=gains.right_gain,
            is_warning=is_warning
        )

    def generate_stereo_pcm(
        self,
        params: AudioSignalParams,
        duration_sec: float = 0.5
    ) -> np.ndarray:
        """
        Generates 2-channel stereo PCM waveform audio samples (float32 [-1, 1]).
        
        Args:
            params: AudioSignalParams specifying frequency, gains, and tempo.
            duration_sec: Duration in seconds.
            
        Returns:
            numpy array of shape (samples, 2) with left and right channel audio.
        """
        sr = self.config.sample_rate
        t = np.linspace(0, duration_sec, int(sr * duration_sec), endpoint=False)
        
        # Continuous sine tone
        carrier = np.sin(2.0 * np.pi * params.frequency_hz * t)

        # Square wave pulse modulation based on tempo
        if params.is_warning:
            pulse = (np.sin(2.0 * np.pi * params.pulse_tempo_hz * t) > 0.0).astype(np.float32)
        else:
            pulse = np.ones_like(t, dtype=np.float32) * 0.1 # Soft ambient pulse for clear path

        mono = carrier * pulse * 0.3 # Master amplitude scale

        # Apply binaural gains
        stereo = np.column_stack([
            mono * params.left_gain,
            mono * params.right_gain
        ]).astype(np.float32)

        return stereo

    def speak(self, text: str, asynchronous: bool = True):
        """
        Speaks text using TTS engine.
        
        Args:
            text: Spoken navigation message.
            asynchronous: If True, runs speech in a non-blocking daemon thread.
        """
        if not text or text == self._last_spoken_phrase:
            return
        self._last_spoken_phrase = text

        if self._tts_engine is None:
            return

        def _do_speak():
            with self._tts_lock:
                try:
                    self._tts_engine.say(text)
                    self._tts_engine.runAndWait()
                except Exception:
                    pass

        if asynchronous:
            t = threading.Thread(target=_do_speak, daemon=True)
            t.start()
        else:
            _do_speak()
