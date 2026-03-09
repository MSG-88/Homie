from datetime import datetime, timezone

from homie_core.utils import utc_now, privacy_tag, truncate_text


def test_utc_now_returns_utc():
    now = utc_now()
    assert now.tzinfo == timezone.utc


def test_privacy_tag():
    tagged = privacy_tag({"data": "hello"}, tags=["personal", "ephemeral"])
    assert tagged["_privacy_tags"] == ["personal", "ephemeral"]
    assert tagged["data"] == "hello"


def test_truncate_text():
    assert truncate_text("hello world", 5) == "hello..."
    assert truncate_text("hi", 10) == "hi"
