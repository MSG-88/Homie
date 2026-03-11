import time
from homie_app.loading import (
    CLILoadingSpinner,
    OverlayLoadingAnimation,
    get_random_thinking_message,
    _THINKING_MESSAGES,
    _SPINNER_FRAMES,
    _BRAINWAVE_FRAMES,
    _DANCE_FRAMES,
)


def test_thinking_messages_not_empty():
    assert len(_THINKING_MESSAGES) >= 20


def test_spinner_frames_not_empty():
    assert len(_SPINNER_FRAMES) > 0
    assert len(_BRAINWAVE_FRAMES) > 0
    assert len(_DANCE_FRAMES) > 0


def test_get_random_thinking_message():
    msg = get_random_thinking_message()
    assert msg in _THINKING_MESSAGES


def test_cli_spinner_start_stop():
    spinner = CLILoadingSpinner(style="brainwave")
    spinner.start()
    assert spinner._running
    time.sleep(0.3)
    spinner.stop()
    assert not spinner._running


def test_cli_spinner_double_start():
    spinner = CLILoadingSpinner()
    spinner.start()
    spinner.start()  # should not crash
    spinner.stop()


def test_cli_spinner_stop_without_start():
    spinner = CLILoadingSpinner()
    spinner.stop()  # should not crash


def test_overlay_animation_creates():
    anim = OverlayLoadingAnimation()
    assert not anim._running


def test_overlay_animation_stop():
    anim = OverlayLoadingAnimation()
    anim._running = True
    anim.stop()
    assert not anim._running


def test_all_styles_valid():
    for style in ["random", "brainwave", "dance"]:
        spinner = CLILoadingSpinner(style=style)
        frames = spinner._pick_frames()
        assert len(frames) > 0
