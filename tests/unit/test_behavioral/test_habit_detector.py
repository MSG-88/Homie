from homie_core.behavioral.habit_detector import HabitDetector


def test_detect_trigger_response_habit():
    hd = HabitDetector(min_occurrences=3, min_confidence=0.6)
    for _ in range(5):
        hd.record_trigger_response("open_ide", "start_music")
    habits = hd.detect_habits()
    trigger_habits = [h for h in habits if h["type"] == "trigger_response"]
    assert len(trigger_habits) >= 1
    assert trigger_habits[0]["trigger"] == "open_ide"


def test_suggest_automations():
    hd = HabitDetector(min_occurrences=3)
    for _ in range(10):
        hd.record_trigger_response("open_ide", "start_music")
    suggestions = hd.suggest_automations()
    assert len(suggestions) >= 1


def test_no_habit_below_threshold():
    hd = HabitDetector(min_occurrences=5)
    hd.record_trigger_response("open_browser", "check_email")
    hd.record_trigger_response("open_browser", "check_email")
    habits = hd.detect_habits()
    trigger_habits = [h for h in habits if h["type"] == "trigger_response"]
    assert len(trigger_habits) == 0
