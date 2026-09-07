"""
Unit tests for PathSense Spatial Audio Engine.
"""

import math
import numpy as np
import pytest

from pathsense.config import AudioConfig
from pathsense.decision_engine import ActionDecision, NavigationAction
from pathsense.spatial_audio import SpatialAudioEngine, BinauralGain, AudioSignalParams


def test_binaural_panning_equal_power():
    # Test equal power property: left^2 + right^2 == 1.0
    for az in [-90.0, -45.0, 0.0, 45.0, 90.0]:
        gains = SpatialAudioEngine.calculate_binaural_gains(az)
        power = gains.left_gain**2 + gains.right_gain**2
        assert pytest.approx(power, abs=1e-5) == 1.0


def test_binaural_panning_directions():
    far_left = SpatialAudioEngine.calculate_binaural_gains(-90.0)
    assert pytest.approx(far_left.left_gain, abs=1e-4) == 1.0
    assert pytest.approx(far_left.right_gain, abs=1e-4) == 0.0

    far_right = SpatialAudioEngine.calculate_binaural_gains(90.0)
    assert pytest.approx(far_right.left_gain, abs=1e-4) == 0.0
    assert pytest.approx(far_right.right_gain, abs=1e-4) == 1.0

    center = SpatialAudioEngine.calculate_binaural_gains(0.0)
    assert pytest.approx(center.left_gain, abs=1e-4) == pytest.approx(center.right_gain, abs=1e-4)


def test_stereo_pcm_waveform_generation():
    engine = SpatialAudioEngine(AudioConfig(sample_rate=44100, enable_tts=False))

    params = AudioSignalParams(
        frequency_hz=440.0,
        pulse_tempo_hz=2.0,
        left_gain=0.7071,
        right_gain=0.7071,
        is_warning=True
    )

    pcm = engine.generate_stereo_pcm(params, duration_sec=0.1)

    assert isinstance(pcm, np.ndarray)
    assert pcm.shape == (int(44100 * 0.1), 2)
    assert pcm.dtype == np.float32
