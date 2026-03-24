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
    inference_router=None,
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

    # ---- chat ----
    chat_history: list[dict] = []

    @app.post("/api/chat")
    async def chat(request: Request):
        _check_auth(request)
        body = await request.json()
        user_message = body.get("message", "").strip()
        if not user_message:
            return {"error": "Empty message"}

        # Build email context
        email_context = ""
        if email_service:
            try:
                summary = email_service.get_summary(days=1)
                email_context = f"\n\nEmail context: {summary.get('unread', 0)} unread emails, {len(summary.get('high_priority', []))} high priority."
                # Add recent high priority subjects
                for hp in summary.get("high_priority", [])[:3]:
                    email_context += f"\n- {hp.get('subject', '')} from {hp.get('sender', '')}"
            except Exception:
                pass

        # Build conversation for the model
        system = f"You are Homie, a privacy-first local AI assistant. All data stays on this device. Be helpful, concise, and direct.{email_context}"

        # Build prompt with recent history (last 10 turns)
        recent = chat_history[-10:]
        prompt_parts = [f"System: {system}"]
        for turn in recent:
            role = "User" if turn["role"] == "user" else "Homie"
            prompt_parts.append(f"{role}: {turn['content']}")
        prompt_parts.append(f"User: {user_message}")
        prompt_parts.append("Homie:")
        prompt = "\n\n".join(prompt_parts)

        # Generate response
        response_text = ""
        if inference_router:
            try:
                response_text = inference_router.generate(
                    prompt, max_tokens=1024, temperature=0.7,
                )
            except Exception as e:
                response_text = f"Inference unavailable: {e}"
        else:
            response_text = "No inference engine configured. Connect a local model via Ollama or configure cloud fallback."

        # Store in history
        chat_history.append({"role": "user", "content": user_message})
        chat_history.append({"role": "assistant", "content": response_text})

        return {
            "response": response_text,
            "source": getattr(inference_router, "active_source", "none") if inference_router else "none",
        }

    @app.get("/api/chat/history")
    def get_chat_history(request: Request):
        _check_auth(request)
        return {"messages": chat_history[-50:]}

    @app.delete("/api/chat/history")
    def clear_chat_history(request: Request):
        _check_auth(request)
        chat_history.clear()
        return {"status": "ok"}

    # ---- email action routes ----

    @app.post("/api/email/search")
    async def email_search(request: Request):
        _check_auth(request)
        if not email_service:
            return {"results": []}
        body = await request.json()
        query = body.get("query", "")
        results = email_service.search(query, max_results=10)
        return {"results": [{"id": m.id, "subject": m.subject, "sender": m.sender, "snippet": m.snippet, "date": m.date} for m in results]}

    @app.get("/api/email/read/{message_id}")
    def email_read(message_id: str, request: Request):
        _check_auth(request)
        if not email_service:
            return {"error": "Email not configured"}
        return email_service.read_message(message_id)

    return app
