"""Tests for rich_console, boot, and status_bar modules."""
from __future__ import annotations

import io

import pytest

try:
    from rich.console import Console as RichConsole
    from rich.theme import Theme

    from homie_app.console.rich_console import HOMIE_THEME, rc
    from homie_app.console.boot import (
        show_boot_screen,
        show_system_check,
        get_greeting,
    )
    from homie_app.console.status_bar import print_status_bar

    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

pytestmark = pytest.mark.skipif(not RICH_AVAILABLE, reason="rich not installed")


# ---------------------------------------------------------------------------
# rich_console module
# ---------------------------------------------------------------------------

EXPECTED_THEME_KEYS = {
    "homie.system",
    "homie.user",
    "homie.assistant",
    "homie.tool",
    "homie.tool.ok",
    "homie.tool.err",
    "homie.memory",
    "homie.dim",
    "homie.error",
    "homie.warn",
    "homie.stage",
    "homie.brand",
}


def test_homie_theme_has_expected_keys() -> None:
    assert EXPECTED_THEME_KEYS.issubset(set(HOMIE_THEME.styles.keys()))


def test_rc_is_rich_console_instance() -> None:
    assert isinstance(rc, RichConsole)


def test_rc_uses_homie_theme() -> None:
    # Verify homie custom styles are resolvable on the console's theme stack.
    # _theme_stack._entries is the merged dict used by Rich internally.
    merged = rc._theme_stack._entries[0]  # type: ignore[attr-defined]
    assert "homie.brand" in merged


# ---------------------------------------------------------------------------
# boot module
# ---------------------------------------------------------------------------

def _make_test_console():
    """Return a RichConsole that writes to a StringIO buffer (no TTY noise)."""
    from rich.console import Console as RC
    from homie_app.console.rich_console import HOMIE_THEME

    buf = io.StringIO()
    return RC(file=buf, theme=HOMIE_THEME, highlight=False, soft_wrap=True)


def test_show_boot_screen_does_not_crash() -> None:
    console = _make_test_console()
    show_boot_screen(console)
    show_boot_screen(console, version="1.2.3")


def test_show_boot_screen_output_contains_version() -> None:
    buf = io.StringIO()
    from rich.console import Console as RC
    from homie_app.console.rich_console import HOMIE_THEME

    console = RC(file=buf, theme=HOMIE_THEME, highlight=False, soft_wrap=True)
    show_boot_screen(console, version="9.9.9")
    output = buf.getvalue()
    assert "9.9.9" in output


def test_show_system_check_all_pass() -> None:
    console = _make_test_console()
    checks = [
        ("model", True, "llama3.gguf loaded"),
        ("memory", True, "42 facts"),
        ("network", True, "online"),
    ]
    show_system_check(console, checks)


def test_show_system_check_mixed_results() -> None:
    console = _make_test_console()
    checks = [
        ("model", True, "ok"),
        ("voice", False, "no mic found"),
        ("tray", False, "pystray missing"),
    ]
    show_system_check(console, checks)  # must not raise


def test_show_system_check_empty() -> None:
    console = _make_test_console()
    show_system_check(console, [])


# ---------------------------------------------------------------------------
# get_greeting
# ---------------------------------------------------------------------------

def test_get_greeting_includes_user_name() -> None:
    result = get_greeting("Alice")
    assert "Alice" in result


def test_get_greeting_returns_string() -> None:
    assert isinstance(get_greeting("Bob"), str)


def test_get_greeting_omits_name_for_default_user() -> None:
    result = get_greeting("User")
    assert "User" not in result


def test_get_greeting_empty_name() -> None:
    result = get_greeting("")
    assert isinstance(result, str)
    assert len(result) > 0


def test_get_greeting_memory_hint_shown_when_facts_gte_3() -> None:
    result = get_greeting("Dave", fact_count=5)
    assert "remember" in result.lower()


def test_get_greeting_no_memory_hint_when_few_facts() -> None:
    result = get_greeting("Dave", fact_count=2)
    assert "remember" not in result.lower()


def test_get_greeting_ends_with_help_hint() -> None:
    result = get_greeting("Eve")
    assert "/help" in result


# ---------------------------------------------------------------------------
# status_bar
# ---------------------------------------------------------------------------

def test_print_status_bar_does_not_crash() -> None:
    console = _make_test_console()
    print_status_bar(console)


def test_print_status_bar_with_all_fields() -> None:
    console = _make_test_console()
    print_status_bar(console, model_name="llama3", memory_count=10, project="Homie")


def test_print_status_bar_output_contains_model_name() -> None:
    buf = io.StringIO()
    from rich.console import Console as RC
    from homie_app.console.rich_console import HOMIE_THEME

    console = RC(file=buf, theme=HOMIE_THEME, highlight=False, soft_wrap=True)
    print_status_bar(console, model_name="mistral-7b")
    assert "mistral-7b" in buf.getvalue()


def test_print_status_bar_output_contains_memory_count() -> None:
    buf = io.StringIO()
    from rich.console import Console as RC
    from homie_app.console.rich_console import HOMIE_THEME

    console = RC(file=buf, theme=HOMIE_THEME, highlight=False, soft_wrap=True)
    print_status_bar(console, memory_count=7)
    assert "7 facts" in buf.getvalue()


def test_print_status_bar_output_contains_project() -> None:
    buf = io.StringIO()
    from rich.console import Console as RC
    from homie_app.console.rich_console import HOMIE_THEME

    console = RC(file=buf, theme=HOMIE_THEME, highlight=False, soft_wrap=True)
    print_status_bar(console, project="HomeServer")
    assert "HomeServer" in buf.getvalue()
