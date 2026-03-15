from __future__ import annotations

import pytest
from homie_core.middleware.hooks import HookRegistry, PipelineStage, RetrievalBundle


def test_has_five_stages():
    assert len(PipelineStage) == 5


def test_stage_values():
    assert PipelineStage.PERCEIVED == "on_perceived"
    assert PipelineStage.CLASSIFIED == "on_classified"
    assert PipelineStage.RETRIEVED == "on_retrieved"
    assert PipelineStage.PROMPT_BUILT == "on_prompt_built"
    assert PipelineStage.REFLECTED == "on_reflected"


def test_retrieval_bundle_creation():
    bundle = RetrievalBundle(facts=["f1"], episodes=["e1"], documents=["d1"])
    assert bundle.facts == ["f1"]
    assert bundle.episodes == ["e1"]
    assert bundle.documents == ["d1"]


def test_emit_no_hooks_returns_data_unchanged():
    registry = HookRegistry()
    data = {"key": "value"}
    result = registry.emit(PipelineStage.PERCEIVED, data)
    assert result is data


def test_register_and_emit_single_hook():
    registry = HookRegistry()
    received = []

    def hook(stage, data):
        received.append((stage, data))
        return data

    registry.register(PipelineStage.PERCEIVED, hook)
    result = registry.emit(PipelineStage.PERCEIVED, "hello")
    assert received == [(PipelineStage.PERCEIVED, "hello")]
    assert result == "hello"


def test_hook_can_modify_data():
    registry = HookRegistry()

    def hook(stage, data):
        return data + " modified"

    registry.register(PipelineStage.CLASSIFIED, hook)
    result = registry.emit(PipelineStage.CLASSIFIED, "input")
    assert result == "input modified"


def test_hook_returning_none_does_not_modify():
    registry = HookRegistry()

    def hook(stage, data):
        return None  # observe only

    registry.register(PipelineStage.RETRIEVED, hook)
    result = registry.emit(PipelineStage.RETRIEVED, "original")
    assert result == "original"


def test_multiple_hooks_chain():
    registry = HookRegistry()

    def hook1(stage, data):
        return data + " A"

    def hook2(stage, data):
        return data + " B"

    registry.register(PipelineStage.PROMPT_BUILT, hook1)
    registry.register(PipelineStage.PROMPT_BUILT, hook2)
    result = registry.emit(PipelineStage.PROMPT_BUILT, "start")
    assert result == "start A B"


def test_hooks_on_different_stages_are_independent():
    registry = HookRegistry()
    calls = []

    def hook_a(stage, data):
        calls.append("a")
        return data

    def hook_b(stage, data):
        calls.append("b")
        return data

    registry.register(PipelineStage.PERCEIVED, hook_a)
    registry.register(PipelineStage.REFLECTED, hook_b)

    registry.emit(PipelineStage.PERCEIVED, "x")
    assert calls == ["a"]

    registry.emit(PipelineStage.REFLECTED, "y")
    assert calls == ["a", "b"]
