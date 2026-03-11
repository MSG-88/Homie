from __future__ import annotations

import signal
import sys
from pathlib import Path
from typing import Iterator, Optional

from homie_core.config import HomieConfig, load_config
from homie_core.enterprise import load_enterprise_policy, apply_policy
from homie_core.intelligence.task_graph import TaskGraph
from homie_core.intelligence.proactive_retrieval import ProactiveRetrieval
from homie_core.intelligence.interruption_model import InterruptionModel
from homie_core.intelligence.session_tracker import SessionTracker
from homie_core.intelligence.briefing import BriefingGenerator
from homie_core.intelligence.observer_loop import ObserverLoop
from homie_core.intelligence.audit_log import AuditLogger
from homie_core.memory.working import WorkingMemory
from homie_app.hotkey import HotkeyListener
from homie_app.overlay import OverlayPopup


class HomieDaemon:
    """Always-active background daemon.

    Three threads:
    1. Main — hotkey listener + overlay UI
    2. Observer — watches OS events, feeds task graph
    3. Scheduler — morning briefing, end-of-day, memory consolidation
    """

    def __init__(self, config_path: Optional[str] = None):
        self._config = load_config(config_path)
        self._running = False

        # Apply enterprise policy if present
        storage = Path(self._config.storage.path)
        policy = load_enterprise_policy(storage)
        if policy:
            self._config = apply_policy(self._config, policy)
            self._audit = AuditLogger(
                log_dir=storage / self._config.storage.log_dir,
                enabled=policy.privacy.audit_log,
            )
        else:
            self._audit = AuditLogger(
                log_dir=storage / self._config.storage.log_dir,
                enabled=False,
            )

        # Core components
        self._working_memory = WorkingMemory()
        self._task_graph = TaskGraph()
        self._interruption_model = InterruptionModel()

        # Session tracker
        self._session_tracker = SessionTracker(storage_dir=storage / "sessions")

        # Proactive retrieval (memories injected later when model loads)
        self._retrieval = ProactiveRetrieval()

        # Briefing generator
        self._briefing = BriefingGenerator(
            session_tracker=self._session_tracker,
            user_name=self._config.user_name,
        )

        # Observer loop
        self._observer = ObserverLoop(
            working_memory=self._working_memory,
            task_graph=self._task_graph,
            on_context_change=self._retrieval.on_context_change,
        )

        # UI components (created but not started)
        self._overlay = OverlayPopup(
            on_submit=self._on_user_query,
            on_submit_stream=self._on_user_query_stream,
        )
        self._hotkey = HotkeyListener(hotkey="alt+8", callback=self._on_hotkey)

        # Model engine (lazy loaded)
        self._engine = None

    def _on_hotkey(self) -> None:
        self._overlay.toggle()

    def _build_daemon_prompt(self, text: str) -> str:
        """Build a compact prompt with staged context within a tight budget."""
        system = "You are Homie, a helpful AI assistant. Be concise."
        budget = 3000 - len(system) - len(text) - 50
        parts = [system]

        # Active window context (cheap, high value)
        active = self._working_memory.get("active_window", "")
        if active and budget > 50:
            ctx = f"\nContext: User is in {active}"
            parts.append(ctx)
            budget -= len(ctx)

        # Staged retrieval context (facts + episodes, truncated)
        staged = self._retrieval.consume_staged_context()
        if staged.get("facts") and budget > 100:
            facts = [f.get("fact", str(f))[:100] for f in staged["facts"][:3]]
            block = "Facts:\n- " + "\n- ".join(facts)
            if len(block) <= budget:
                parts.append(block)
                budget -= len(block)

        if staged.get("episodes") and budget > 100:
            eps = [e.get("summary", str(e))[:150] for e in staged["episodes"][:1]]
            block = "Related: " + eps[0]
            if len(block) <= budget:
                parts.append(block)
                budget -= len(block)

        parts.append(f"\nUser: {text}\nAssistant:")
        return "\n".join(parts)

    def _on_user_query(self, text: str) -> str:
        if not self._engine:
            self._load_engine()
        if not self._engine:
            return "Model not available. Run 'homie init' to set up."

        prompt = self._build_daemon_prompt(text)
        try:
            response = self._engine.generate(prompt, max_tokens=512, timeout=120)
            self._audit.log_query(prompt=text, response=response,
                                  model=self._config.llm.model_path)
            return response
        except TimeoutError:
            return "Timed out — the model may be overloaded."
        except Exception as e:
            return f"Error: {e}"

    def _on_user_query_stream(self, text: str):
        """Yield tokens as they're generated for real-time overlay updates."""
        if not self._engine:
            self._load_engine()
        if not self._engine:
            yield "Model not available. Run 'homie init' to set up."
            return

        prompt = self._build_daemon_prompt(text)
        chunks = []
        try:
            for token in self._engine.stream(prompt, max_tokens=512, temperature=0.7):
                chunks.append(token)
                yield token
            # Audit the full response
            full = "".join(chunks)
            self._audit.log_query(prompt=text, response=full,
                                  model=self._config.llm.model_path)
        except Exception as e:
            yield f"\nError: {e}"

    def _load_engine(self) -> None:
        try:
            from homie_core.model.engine import ModelEngine
            from homie_core.model.registry import ModelRegistry, ModelEntry

            engine = ModelEngine()
            registry = ModelRegistry(
                Path(self._config.storage.path) / self._config.storage.models_dir
            )
            registry.initialize()
            entry = registry.get_active()

            if not entry and self._config.llm.model_path:
                entry = ModelEntry(
                    name=self._config.llm.model_path,
                    path=self._config.llm.model_path,
                    format=self._config.llm.backend,
                    params="cloud" if self._config.llm.backend == "cloud" else "unknown",
                )

            if entry:
                kwargs = {}
                if entry.format == "cloud":
                    kwargs["api_key"] = self._config.llm.api_key
                    kwargs["base_url"] = self._config.llm.api_base_url or "https://api.openai.com/v1"
                else:
                    kwargs["n_ctx"] = self._config.llm.context_length
                    kwargs["n_gpu_layers"] = self._config.llm.gpu_layers
                engine.load(entry, **kwargs)
                self._engine = engine
        except Exception:
            self._engine = None

    def start(self) -> None:
        self._running = True
        print("Homie daemon starting...")

        # Show morning briefing if there's a previous session
        briefing = self._briefing.morning_briefing()
        print(f"\n{briefing}\n")

        # Start observer thread
        self._observer.start()
        print("  Observer: running")

        # Start hotkey listener
        self._hotkey.start()
        print("  Hotkey (Alt+8): active")

        print("\nHomie is running in the background. Press Alt+8 or say 'hey homie' to activate.")
        print("Press Ctrl+C to stop.\n")

        # Main thread waits
        try:
            signal.signal(signal.SIGINT, lambda *_: self.stop())
            while self._running:
                import time
                time.sleep(1)
        except KeyboardInterrupt:
            self.stop()

    def stop(self) -> None:
        print("\nHomie daemon stopping...")
        self._running = False

        # Save session for tomorrow
        apps = self._observer.get_app_tracker().get_usage()
        switch_count = self._observer.get_app_tracker().get_switch_count(minutes=1440)

        digest = self._briefing.end_of_day_digest(
            self._task_graph, apps_used=apps, switch_count=switch_count,
        )
        print(f"\n{digest}")

        self._observer.stop()
        self._hotkey.stop()
        self._overlay.hide()

        if self._engine:
            self._engine.unload()

        print("Goodbye!")
        sys.exit(0)
