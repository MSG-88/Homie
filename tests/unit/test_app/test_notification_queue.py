"""Tests for NotificationQueue and CliNotification."""
from __future__ import annotations

import pytest

try:
    from homie_app.console.notification_queue import CliNotification, NotificationQueue

    NQ_AVAILABLE = True
except ImportError:
    NQ_AVAILABLE = False

pytestmark = pytest.mark.skipif(not NQ_AVAILABLE, reason="notification_queue not importable")


# ---------------------------------------------------------------------------
# CliNotification dataclass
# ---------------------------------------------------------------------------

def test_cli_notification_defaults() -> None:
    n = CliNotification(text="hello", category="system")
    assert n.text == "hello"
    assert n.category == "system"
    assert n.priority == "normal"


def test_cli_notification_urgent() -> None:
    n = CliNotification(text="alert!", category="security", priority="urgent")
    assert n.priority == "urgent"


def test_cli_notification_low_priority() -> None:
    n = CliNotification(text="fyi", category="info", priority="low")
    assert n.priority == "low"


# ---------------------------------------------------------------------------
# NotificationQueue — push and drain
# ---------------------------------------------------------------------------

def test_push_and_drain_single() -> None:
    q = NotificationQueue()
    n = CliNotification(text="msg", category="test")
    q.push(n)
    items = q.drain()
    assert len(items) == 1
    assert items[0].text == "msg"


def test_drain_returns_items_in_order() -> None:
    q = NotificationQueue()
    for i in range(3):
        q.push(CliNotification(text=f"msg{i}", category="test"))
    items = q.drain()
    assert [item.text for item in items] == ["msg0", "msg1", "msg2"]


def test_drain_clears_queue() -> None:
    q = NotificationQueue()
    q.push(CliNotification(text="x", category="test"))
    q.drain()
    assert len(q) == 0


def test_drain_on_empty_queue_returns_empty_list() -> None:
    q = NotificationQueue()
    assert q.drain() == []


def test_second_drain_returns_empty() -> None:
    q = NotificationQueue()
    q.push(CliNotification(text="once", category="test"))
    q.drain()
    assert q.drain() == []


# ---------------------------------------------------------------------------
# NotificationQueue — max_size (bounded deque)
# ---------------------------------------------------------------------------

def test_max_size_respected() -> None:
    q = NotificationQueue(max_size=3)
    for i in range(5):
        q.push(CliNotification(text=f"msg{i}", category="test"))
    assert len(q) == 3


def test_max_size_keeps_latest_items() -> None:
    q = NotificationQueue(max_size=3)
    for i in range(5):
        q.push(CliNotification(text=f"msg{i}", category="test"))
    items = q.drain()
    assert [item.text for item in items] == ["msg2", "msg3", "msg4"]


def test_default_max_size_is_ten() -> None:
    q = NotificationQueue()
    for i in range(15):
        q.push(CliNotification(text=f"m{i}", category="test"))
    assert len(q) == 10


# ---------------------------------------------------------------------------
# NotificationQueue — has_urgent
# ---------------------------------------------------------------------------

def test_has_urgent_false_when_empty() -> None:
    q = NotificationQueue()
    assert q.has_urgent() is False


def test_has_urgent_false_when_only_normal() -> None:
    q = NotificationQueue()
    q.push(CliNotification(text="a", category="test", priority="normal"))
    q.push(CliNotification(text="b", category="test", priority="low"))
    assert q.has_urgent() is False


def test_has_urgent_true_when_one_urgent() -> None:
    q = NotificationQueue()
    q.push(CliNotification(text="a", category="test", priority="normal"))
    q.push(CliNotification(text="ALERT", category="security", priority="urgent"))
    assert q.has_urgent() is True


def test_has_urgent_false_after_drain() -> None:
    q = NotificationQueue()
    q.push(CliNotification(text="ALERT", category="security", priority="urgent"))
    q.drain()
    assert q.has_urgent() is False


# ---------------------------------------------------------------------------
# NotificationQueue — __len__
# ---------------------------------------------------------------------------

def test_len_empty() -> None:
    q = NotificationQueue()
    assert len(q) == 0


def test_len_after_pushes() -> None:
    q = NotificationQueue()
    q.push(CliNotification(text="a", category="test"))
    q.push(CliNotification(text="b", category="test"))
    assert len(q) == 2


def test_len_after_drain() -> None:
    q = NotificationQueue()
    q.push(CliNotification(text="x", category="test"))
    q.drain()
    assert len(q) == 0
