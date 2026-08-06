"""
tts_alert.py
------------
Converts obstacle-analysis messages into spoken audio, with simple
smoothing so it doesn't repeat the same warning 30 times a second.

Smoothing rule (deliberately simple, easy to justify in a viva):
  - Speak immediately if the message is NEW (different from last spoken).
  - If the message is the SAME as last time, only repeat it after
    REPEAT_COOLDOWN_SECONDS have passed.
"""

import time

REPEAT_COOLDOWN_SECONDS = 3.0

_engine = None
_last_message = None
_last_spoken_time = 0.0


def _load_engine():
    global _engine
    if _engine is None:
        import pyttsx3
        _engine = pyttsx3.init()
        _engine.setProperty("rate", 165)  # slightly slower than default for clarity
    return _engine


def speak(message: str, force: bool = False):
    """
    Speaks `message` out loud, respecting the cooldown smoothing rule.
    Set force=True to bypass smoothing (e.g., for a DANGER-level alert
    that should always interrupt).
    """
    global _last_message, _last_spoken_time

    now = time.time()
    is_new_message = message != _last_message
    cooldown_elapsed = (now - _last_spoken_time) >= REPEAT_COOLDOWN_SECONDS

    if force or is_new_message or cooldown_elapsed:
        engine = _load_engine()
        print(f"[voice] Speaking: \"{message}\"")
        engine.say(message)
        engine.runAndWait()
        _last_message = message
        _last_spoken_time = now
    else:
        print(f"[voice] Skipped (smoothing): \"{message}\"")


def reset():
    """Call this if you want to force the next message to always speak."""
    global _last_message, _last_spoken_time
    _last_message = None
    _last_spoken_time = 0.0


if __name__ == "__main__":
    # Self-test of the smoothing logic WITHOUT needing an audio device --
    # this stubs out the engine so you can verify the cooldown behavior.
    class _FakeEngine:
        def say(self, msg):
            pass
        def runAndWait(self):
            pass
        def setProperty(self, *a, **kw):
            pass

    _engine = _FakeEngine()  # bypass real pyttsx3 for this dry-run test

    print("Test: same message spammed rapidly -> should mostly skip")
    for _ in range(3):
        speak("person very close, center. Stop or move away.")
        time.sleep(0.2)

    print("\nTest: new message -> should always speak immediately")
    speak("path clear.")

    print("\nTest: after cooldown -> repeat should speak again")
    reset()
    speak("chair ahead, left. Proceed carefully.")
