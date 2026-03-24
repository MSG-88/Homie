from __future__ import annotations

from typing import Any

try:
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.responses import HTMLResponse, JSONResponse
    _HAS_FASTAPI = True
except ImportError:
    FastAPI = None
    HTTPException = None
    Request = None
    JSONResponse = None
    _HAS_FASTAPI = False


def create_dashboard_app(
    config=None,
    memory_semantic=None,
    belief_system=None,
    plugin_manager=None,
    suggestion_engine=None,
    email_service=None,
    session_token: str | None = None,
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

    # ---- session auth helper ----

    def _check_auth(request: Request):
        if not session_token:
            return
        token = request.cookies.get("homie_session")
        if token != session_token:
            raise HTTPException(status_code=401, detail="Unauthorized")

    # ---- email routes ----

    @app.get("/api/email/summary")
    def email_summary(request: Request):
        _check_auth(request)
        if not email_service:
            return {"total": 0, "unread": 0, "high_priority": []}
        return email_service.get_summary(days=1)

    @app.get("/api/email/unread")
    def email_unread(request: Request):
        _check_auth(request)
        if not email_service:
            return {"high": [], "medium": [], "low": []}
        return email_service.get_unread()

    @app.post("/api/email/triage")
    def email_triage(request: Request):
        _check_auth(request)
        if not email_service:
            return {"status": "Email not configured", "emails": []}
        return email_service.triage()

    @app.get("/api/email/digest")
    def email_digest(request: Request):
        _check_auth(request)
        if not email_service:
            return {"digest": "Email not configured."}
        result = email_service.get_intelligent_digest(days=1)
        if isinstance(result, str):
            return {"digest": result}
        return {"digest": result}

    @app.get("/briefing", response_class=HTMLResponse)
    def briefing_page(request: Request):
        _check_auth(request)
        from homie_app.tray.briefing_page import render_briefing_page

        user_name = "User"
        if config:
            user_name = getattr(config, "user_name", "User") or "User"

        summary = {"total": 0, "unread": 0, "high_priority": []}
        unread_data = {"high": [], "medium": [], "low": []}
        digest = "Email not configured."

        if email_service:
            summary = email_service.get_summary(days=1)
            unread_data = email_service.get_unread()
            raw_digest = email_service.get_intelligent_digest(days=1)
            digest = raw_digest if isinstance(raw_digest, str) else str(raw_digest)

        port = 8721
        return render_briefing_page(
            user_name=user_name,
            summary=summary,
            unread=unread_data,
            digest=digest,
            session_token=session_token or "",
            api_port=port,
        )

    @app.post("/api/email/mark-read/{message_id}")
    def mark_read(message_id: str, request: Request):
        _check_auth(request)
        if email_service:
            email_service.mark_read(message_id)
        return {"status": "ok"}

    return app
