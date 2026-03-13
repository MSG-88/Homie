from __future__ import annotations

from typing import Any

try:
    from fastapi import FastAPI
    from fastapi.responses import JSONResponse
    _HAS_FASTAPI = True
except ImportError:
    FastAPI = None
    JSONResponse = None
    _HAS_FASTAPI = False


def create_dashboard_app(
    config=None,
    memory_semantic=None,
    belief_system=None,
    plugin_manager=None,
    suggestion_engine=None,
):
    if not _HAS_FASTAPI:
        raise ImportError(
            "fastapi is required for the dashboard. "
            "Install with: pip install homie-ai[app]"
        )
    app = FastAPI(title="Homie AI Dashboard", version="0.1.0")

    @app.get("/api/health")
    def health():
        return {"status": "ok", "version": "0.1.0"}

    @app.get("/api/profile")
    def get_profile():
        if memory_semantic:
            return memory_semantic.get_all_profiles()
        return {}

    @app.get("/api/beliefs")
    def get_beliefs():
        if belief_system:
            return belief_system.get_beliefs()
        return []

    @app.get("/api/plugins")
    def get_plugins():
        if plugin_manager:
            return plugin_manager.list_plugins()
        return []

    @app.get("/api/suggestions/rates")
    def get_suggestion_rates():
        if suggestion_engine:
            return suggestion_engine.get_acceptance_rates()
        return {}

    @app.get("/api/memory/facts")
    def get_facts():
        if memory_semantic:
            return memory_semantic.get_facts(min_confidence=0.3)
        return []

    @app.post("/api/memory/forget")
    def forget_topic(body: dict):
        topic = body.get("topic", "")
        if memory_semantic and topic:
            memory_semantic.forget_topic(topic)
            return {"status": "ok", "forgotten": topic}
        return JSONResponse(status_code=400, content={"error": "Missing topic"})

    return app
