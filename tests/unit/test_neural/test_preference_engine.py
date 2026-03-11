from homie_core.neural.preference_engine import PreferenceEngine


def test_record_preference():
    pe = PreferenceEngine()
    pe.record("tool", "vscode", 1.0)
    prefs = pe.get_preferences("tool")
    assert "vscode" in prefs
    assert prefs["vscode"] > 0


def test_preference_strength_increases():
    pe = PreferenceEngine()
    pe.record("tool", "vscode", 1.0)
    s1 = pe.get_preferences("tool")["vscode"]
    pe.record("tool", "vscode", 1.0)
    s2 = pe.get_preferences("tool")["vscode"]
    assert s2 > s1


def test_ema_decay():
    pe = PreferenceEngine(ema_alpha=0.3)
    pe.record("tool", "vim", 1.0)
    pe.record("tool", "vim", 0.0)
    pe.record("tool", "vim", 0.0)
    pref = pe.get_preferences("tool")["vim"]
    assert pref < 0.5


def test_cusum_detects_shift():
    pe = PreferenceEngine(cusum_threshold=2.0)
    for _ in range(20):
        pe.record("activity", "coding", 1.0)
    for _ in range(10):
        pe.record("activity", "writing", 1.0)
        pe.record("activity", "coding", 0.0)
    shifts = pe.get_detected_shifts()
    assert len(shifts) > 0
    assert any(s["domain"] == "activity" for s in shifts)


def test_no_false_shift_on_stable():
    pe = PreferenceEngine(cusum_threshold=3.0)
    for _ in range(50):
        pe.record("activity", "coding", 1.0)
    shifts = pe.get_detected_shifts()
    coding_shifts = [s for s in shifts if s["key"] == "coding"]
    assert len(coding_shifts) == 0


def test_get_all_preferences():
    pe = PreferenceEngine()
    pe.record("tool", "vscode", 1.0)
    pe.record("time", "morning", 0.8)
    all_prefs = pe.get_all_preferences()
    assert "tool" in all_prefs
    assert "time" in all_prefs


def test_get_dominant_preference():
    pe = PreferenceEngine()
    pe.record("tool", "vscode", 1.0)
    pe.record("tool", "vscode", 1.0)
    pe.record("tool", "vim", 0.3)
    dominant = pe.get_dominant("tool")
    assert dominant == "vscode"


def test_serialize_deserialize():
    pe = PreferenceEngine()
    pe.record("tool", "vscode", 1.0)
    data = pe.serialize()
    pe2 = PreferenceEngine.deserialize(data)
    prefs = pe2.get_preferences("tool")
    assert "vscode" in prefs
