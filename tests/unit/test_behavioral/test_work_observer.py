from homie_core.behavioral.work_observer import WorkObserver


def test_observe_coding():
    wo = WorkObserver()
    wo.observe("code.exe", "main.py - MyProject - Visual Studio Code")
    profile = wo.get_profile_updates()
    assert "Python" in profile["languages"]


def test_detect_project():
    wo = WorkObserver()
    wo.observe("code.exe", "app.ts - Homie - Visual Studio Code")
    assert wo._current_project is not None


def test_multiple_languages():
    wo = WorkObserver()
    wo.observe("code.exe", "main.py - Project")
    wo.observe("code.exe", "main.py - Project")
    wo.observe("code.exe", "index.ts - Project")
    profile = wo.get_profile_updates()
    assert len(profile["languages"]) >= 1
