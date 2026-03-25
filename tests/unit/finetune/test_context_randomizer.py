"""Tests for the context randomizer module."""

from __future__ import annotations

from homie_core.finetune.synthetic.context_randomizer import (
    ContextRandomizer,
    FictionalContext,
)


class TestContextRandomizer:
    """Tests for ContextRandomizer and FictionalContext."""

    def test_generates_context(self) -> None:
        randomizer = ContextRandomizer(seed=42)
        ctx = randomizer.generate()
        assert isinstance(ctx, FictionalContext)

    def test_has_required_fields(self) -> None:
        randomizer = ContextRandomizer(seed=42)
        ctx = randomizer.generate()
        assert isinstance(ctx.user_name, str) and ctx.user_name
        assert isinstance(ctx.role, str) and ctx.role
        assert isinstance(ctx.projects, list) and len(ctx.projects) > 0
        assert isinstance(ctx.preferences, dict) and len(ctx.preferences) > 0
        assert isinstance(ctx.connected_services, list) and len(ctx.connected_services) > 0
        assert ctx.time_of_day in ("morning", "afternoon", "evening")
        assert isinstance(ctx.day_of_week, str) and ctx.day_of_week
        assert isinstance(ctx.recent_activity, list)
        assert isinstance(ctx.pending_tasks, list)
        # Check project structure
        proj = ctx.projects[0]
        assert "name" in proj and "language" in proj and "status" in proj

    def test_deterministic_with_seed(self) -> None:
        ctx1 = ContextRandomizer(seed=99).generate()
        ctx2 = ContextRandomizer(seed=99).generate()
        assert ctx1 == ctx2

    def test_different_seeds_differ(self) -> None:
        ctx1 = ContextRandomizer(seed=1).generate()
        ctx2 = ContextRandomizer(seed=2).generate()
        assert ctx1 != ctx2

    def test_to_system_prompt(self) -> None:
        ctx = ContextRandomizer(seed=42).generate()
        prompt = ctx.to_system_prompt()
        assert isinstance(prompt, str)
        assert ctx.user_name in prompt
        assert ctx.role in prompt
        assert ctx.time_of_day in prompt

    def test_to_detail_string(self) -> None:
        ctx = ContextRandomizer(seed=42).generate()
        detail = ctx.to_detail()
        assert isinstance(detail, str) and len(detail) > 0
        # Should be the first project name
        assert detail == ctx.projects[0]["name"]
