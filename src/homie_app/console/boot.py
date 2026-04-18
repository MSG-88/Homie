from __future__ import annotations

from rich.align import Align
from rich.text import Text
from rich.rule import Rule

HOMIE_ASCII = r"""
  ██╗  ██╗ ██████╗ ███╗   ███╗██╗███████╗
  ██║  ██║██╔═══██╗████╗ ████║██║██╔════╝
  ███████║██║   ██║██╔████╔██║██║█████╗
  ██╔══██║██║   ██║██║╚██╔╝██║██║██╔══╝
  ██║  ██║╚██████╔╝██║ ╚═╝ ██║██║███████╗
  ╚═╝  ╚═╝ ╚═════╝ ╚═╝     ╚═╝╚═╝╚══════╝
"""


def show_boot_screen(rc, version: str = "0.3.0") -> None:
    try:
        rc.print()
        rc.print(Align.center(Text(HOMIE_ASCII, style="homie.brand")))
        rc.print(Align.center(Text(f"v{version}  ·  Local AI Assistant  ·  100% Private", style="homie.dim")))
        rc.print()
    except (UnicodeEncodeError, UnicodeDecodeError):
        # Fallback for terminals that can't handle box-drawing characters
        rc.print()
        rc.print(Align.center(Text("H O M I E", style="bold")))
        rc.print(Align.center(Text(f"v{version} - Local AI Assistant - 100% Private", style="dim")))
        rc.print()


def show_system_check(rc, checks: list[tuple[str, bool, str]]) -> None:
    """checks: [(label, passed, detail), ...]"""
    rc.print(Rule("[homie.system]System Check[/]", style="homie.dim"))
    for label, passed, detail in checks:
        icon = "[homie.tool.ok]✓[/]" if passed else "[homie.warn]✗[/]"
        rc.print(f"  {icon} [homie.dim]{label:<16}[/] {detail}")
    rc.print()


def get_greeting(user_name: str, fact_count: int = 0) -> str:
    from datetime import datetime

    hour = datetime.now().hour
    if hour < 6:
        prefix = "Still up?"
    elif hour < 12:
        prefix = "Good morning,"
    elif hour < 17:
        prefix = "Hey,"
    elif hour < 21:
        prefix = "Good evening,"
    else:
        prefix = "Evening,"

    name_part = f" {user_name}" if user_name and user_name != "User" else ""
    memory_hint = " I remember where we left off." if fact_count >= 3 else ""
    return f"{prefix}{name_part}.{memory_hint} Type /help or just chat."
