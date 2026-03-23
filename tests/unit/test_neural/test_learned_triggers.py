"""Tests for the LearnedTriggerManager module."""

import pytest

from homie_core.neural.proactive.learned_triggers import (
    LearnedTriggerManager,
    _LEARNING_THRESHOLD,
)


class DictStorage:
    """Simple dict-based storage for testing."""

    def __init__(self):
        self._data = {}

    def get(self, key, default=None):
        return self._data.get(key, default)

    def set(self, key, value):
        self._data[key] = value


@pytest.fixture
def storage():
    return DictStorage()


@pytest.fixture
def manager(storage):
    return LearnedTriggerManager(storage)


# ---------------------------------------------------------------------------
# learn_from_request
# ---------------------------------------------------------------------------

class TestLearnFromRequest:
    def test_single_request_does_not_create_task(self, manager):
        manager.learn_from_request("show me the expense report", "finance")
        assert len(manager.get_learned_tasks()) == 0

    def test_repeated_request_creates_task(self, manager):
        for _ in range(_LEARNING_THRESHOLD):
            manager.learn_from_request("show me the expense report", "finance")
        tasks = manager.get_learned_tasks()
        assert len(tasks) == 1
        assert tasks[0].domain == "finance"

    def test_different_requests_tracked_separately(self, manager):
        for _ in range(_LEARNING_THRESHOLD):
            manager.learn_from_request("expense report", "finance")
            manager.learn_from_request("contract review", "legal")
        tasks = manager.get_learned_tasks()
        assert len(tasks) == 2

    def test_empty_request_ignored(self, manager):
        manager.learn_from_request("", "finance")
        assert manager.get_request_counts() == {}

    def test_whitespace_request_ignored(self, manager):
        manager.learn_from_request("   ", "finance")
        assert manager.get_request_counts() == {}

    def test_strips_common_prefixes(self, manager):
        # "can you expense report" and "please expense report" should normalise similarly
        for _ in range(_LEARNING_THRESHOLD):
            manager.learn_from_request("can you show the expense report", "finance")
        tasks = manager.get_learned_tasks()
        assert len(tasks) == 1
        assert "expense" in tasks[0].action


# ---------------------------------------------------------------------------
# get_learned_tasks
# ---------------------------------------------------------------------------

class TestGetLearnedTasks:
    def test_returns_proactive_task_objects(self, manager):
        for _ in range(_LEARNING_THRESHOLD):
            manager.learn_from_request("budget analysis", "finance")
        tasks = manager.get_learned_tasks()
        assert len(tasks) == 1
        task = tasks[0]
        assert task.trigger_type == "schedule"
        assert task.enabled is True

    def test_no_tasks_initially(self, manager):
        assert manager.get_learned_tasks() == []


# ---------------------------------------------------------------------------
# remove_learned_task
# ---------------------------------------------------------------------------

class TestRemoveLearnedTask:
    def test_remove_existing(self, manager):
        for _ in range(_LEARNING_THRESHOLD):
            manager.learn_from_request("weekly summary", "general")
        counts = manager.get_request_counts()
        pattern = list(counts.keys())[0]
        assert manager.remove_learned_task(pattern) is True
        assert len(manager.get_learned_tasks()) == 0

    def test_remove_nonexistent(self, manager):
        assert manager.remove_learned_task("no_such_pattern") is False


# ---------------------------------------------------------------------------
# persistence
# ---------------------------------------------------------------------------

class TestPersistence:
    def test_data_persisted_to_storage(self, storage):
        mgr = LearnedTriggerManager(storage)
        for _ in range(_LEARNING_THRESHOLD):
            mgr.learn_from_request("tax summary", "tax")
        # Create a new manager from the same storage
        mgr2 = LearnedTriggerManager(storage)
        tasks = mgr2.get_learned_tasks()
        assert len(tasks) == 1
        assert tasks[0].domain == "tax"

    def test_request_counts_persisted(self, storage):
        mgr = LearnedTriggerManager(storage)
        mgr.learn_from_request("balance sheet", "accounting")
        mgr2 = LearnedTriggerManager(storage)
        counts = mgr2.get_request_counts()
        assert sum(counts.values()) >= 1
