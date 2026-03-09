from __future__ import annotations

from datetime import datetime, timezone
from typing import Any


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def privacy_tag(data: dict[str, Any], tags: list[str]) -> dict[str, Any]:
    return {**data, "_privacy_tags": tags}


def truncate_text(text: str, max_len: int) -> str:
    if len(text) <= max_len:
        return text
    return text[:max_len] + "..."
