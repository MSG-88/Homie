from __future__ import annotations

import os
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
from homie_core.intelligence.flow_detector import FlowDetector
from homie_core.intelligence.workflow_predictor import WorkflowPredictor
from homie_core.memory.working import WorkingMemory
from homie_core.brain.orchestrator import BrainOrchestrator
from homie_app.hotkey import HotkeyListener
from homie_app.overlay import OverlayPopup
from homie_app.prompts.system import SYSTEM_PROMPT


class HomieDaemon:
    """Always-active background daemon with full neural intelligence.

    Threads:
    1. Main — hotkey listener + overlay UI
    2. Observer — watches OS events, feeds neural components + task graph
    3. Scheduler — morning briefing, end-of-day, memory consolidation

    When HF_KEY is present, activates:
    - HF Inference API for chat (streaming)
    - HF Embeddings for neural perception (activity classification,
      semantic context tracking, sentiment analysis, memory consolidation)
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

        # Proactive retrieval
        self._retrieval = ProactiveRetrieval()

        # Briefing generator
        self._briefing = BriefingGenerator(
            session_tracker=self._session_tracker,
            user_name=self._config.user_name,
        )

        # Neural components (initialized if HF_KEY available)
        self._embeddings = None
        self._context_engine = None
        self._activity_classifier = None
        self._sentiment_analyzer = None
        self._neural_consolidator = None
        self._flow_detector = FlowDetector()
        self._workflow_predictor = WorkflowPredictor(order=2, smoothing_k=1.0)
        self._rhythm_model = None
        self._behavioral_profile = None
        self._preference_engine = None

        # Try to initialize neural components with HF embeddings
        self._init_neural_components()

        # Observer loop — with neural components if available
        self._observer = ObserverLoop(
            working_memory=self._working_memory,
            task_graph=self._task_graph,
            on_context_change=self._retrieval.on_context_change,
            context_engine=self._context_engine,
            activity_classifier=self._activity_classifier,
            rhythm_model=self._rhythm_model,
            behavioral_profile=self._behavioral_profile,
            preference_engine=self._preference_engine,
            workflow_predictor=self._workflow_predictor,
            flow_detector=self._flow_detector,
        )

        # UI components
        self._overlay = OverlayPopup(
            on_submit=self._on_user_query,
            on_submit_stream=self._on_user_query_stream,
        )
        self._hotkey = HotkeyListener(hotkey="alt+8", callback=self._on_hotkey)

        # Model engine + brain (lazy loaded)
        self._engine = None
        self._brain: Optional[BrainOrchestrator] = None

    def _init_neural_components(self) -> None:
        """Initialize neural perception if HF_KEY is available.

        This activates:
        - SemanticContextEngine (tracks what user is doing via embeddings)
        - ActivityClassifier (classifies activity into 9 categories)
        - SentimentAnalyzer (detects user mood/stress)
        - NeuralConsolidator (clusters episodic memories)
        - CircadianRhythmModel (Fourier-based productivity curves)
        - BehavioralProfile (PCA fingerprint of user patterns)
        - PreferenceEngine (EMA + CUSUM preference tracking)
        """
        hf_key = os.environ.get("HF_KEY", "") or self._config.llm.api_key
        if not hf_key:
            print("  Neural: offline (no HF_KEY — set it for full intelligence)")
            return

        try:
            from homie_core.model.hf_backend import HFEmbeddings
            self._embeddings = HFEmbeddings(api_key=hf_key)
            self._embeddings.connect()
            embed_fn = self._embeddings.embed
            embed_dim = self._embeddings.dimension
            print(f"  Neural: HF embeddings connected (dim={embed_dim})")
        except Exception as e:
            print(f"  Neural: HF embeddings failed ({e}) — running without")
            return

        # Now wire up all neural components with the real embed_fn
        try:
            from homie_core.neural.context_engine import SemanticContextEngine
            self._context_engine = SemanticContextEngine(
                embed_fn=embed_fn, embed_dim=embed_dim,
            )
            print("  Neural: SemanticContextEngine active")
        except Exception as e:
            print(f"  Neural: context engine failed ({e})")

        try:
            from homie_core.neural.activity_classifier import ActivityClassifier
            self._activity_classifier = ActivityClassifier(
                embed_fn=embed_fn, embed_dim=embed_dim,
            )
            # Pre-compute prototype embeddings
            self._activity_classifier._init_prototypes()
            print("  Neural: ActivityClassifier active (9 categories)")
        except Exception as e:
            print(f"  Neural: activity classifier failed ({e})")

        try:
            from homie_core.neural.sentiment import SentimentAnalyzer
            self._sentiment_analyzer = SentimentAnalyzer(embed_fn=embed_fn)
            print("  Neural: SentimentAnalyzer active")
        except Exception as e:
            print(f"  Neural: sentiment analyzer failed ({e})")

        try:
            from homie_core.neural.consolidator import NeuralConsolidator
            self._neural_consolidator = NeuralConsolidator(
                embed_fn=embed_fn, similarity_threshold=0.7,
            )
            print("  Neural: NeuralConsolidator active")
        except Exception as e:
            print(f"  Neural: consolidator failed ({e})")

        try:
            from homie_core.neural.rhythm_model import CircadianRhythmModel
            self._rhythm_model = CircadianRhythmModel()
            print("  Neural: CircadianRhythmModel active")
        except Exception as e:
            print(f"  Neural: rhythm model failed ({e})")

        try:
            from homie_core.neural.behavioral_profile import BehavioralProfile
            self._behavioral_profile = BehavioralProfile(embed_dim=embed_dim)
            print("  Neural: BehavioralProfile active")
        except Exception as e:
            print(f"  Neural: behavioral profile failed ({e})")

        try:
            from homie_core.neural.preference_engine import PreferenceEngine
            self._preference_engine = PreferenceEngine()
            print("  Neural: PreferenceEngine active")
        except Exception as e:
            print(f"  Neural: preference engine failed ({e})")

    def _on_hotkey(self) -> None:
        self._overlay.toggle()

    def _inject_proactive_context(self) -> None:
        """Feed proactive retrieval + sentiment into working memory."""
        staged = self._retrieval.consume_staged_context()
        if staged.get("facts"):
            self._working_memory.update("staged_facts", staged["facts"][:5])
        if staged.get("episodes"):
            self._working_memory.update("staged_episodes", staged["episodes"][:3])

    def _analyze_sentiment(self, text: str) -> None:
        """Run sentiment analysis on user input and update working memory."""
        if self._sentiment_analyzer:
            try:
                result = self._sentiment_analyzer.analyze(text)
                self._working_memory.update("sentiment", result.sentiment)
                self._working_memory.update("arousal", result.arousal)
            except Exception:
                pass

    def _ensure_brain(self) -> bool:
        """Ensure model + brain are loaded with tools. Returns True if ready."""
        if not self._engine:
            self._load_engine()
        if not self._engine:
            return False
        if not self._brain:
            # Build tool registry with built-in tools
            from homie_core.brain.tool_registry import ToolRegistry
            from homie_core.brain.builtin_tools import register_builtin_tools

            tool_registry = ToolRegistry()
            register_builtin_tools(
                registry=tool_registry,
                working_memory=self._working_memory,
            )

            self._brain = BrainOrchestrator(
                model_engine=self._engine,
                working_memory=self._working_memory,
                tool_registry=tool_registry,
            )
            self._brain.set_system_prompt(SYSTEM_PROMPT)
        return True

    def _on_user_query(self, text: str) -> str:
        if not self._ensure_brain():
            return "Model not available. Run 'homie init' to set up."

        self._inject_proactive_context()
        self._analyze_sentiment(text)
        try:
            response = self._brain.process(text)
            self._audit.log_query(prompt=text, response=response,
                                  model=self._config.llm.model_path)
            return response
        except TimeoutError:
            return "Timed out — the model may be overloaded."
        except Exception as e:
            return f"Error: {e}"

    def _on_user_query_stream(self, text: str):
        """Yield tokens via the full cognitive pipeline — streaming."""
        if not self._ensure_brain():
            yield "Model not available. Run 'homie init' to set up."
            return

        self._inject_proactive_context()
        self._analyze_sentiment(text)
        chunks = []
        try:
            for token in self._brain.process_stream(text):
                chunks.append(token)
                yield token
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
                    params="hf" if self._config.llm.backend == "hf" else (
                        "cloud" if self._config.llm.backend == "cloud" else "unknown"
                    ),
                )

            if entry:
                kwargs = {}
                if entry.format == "hf":
                    kwargs["api_key"] = self._config.llm.api_key or os.environ.get("HF_KEY", "")
                elif entry.format == "cloud":
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

        # Start observer thread (with neural components if available)
        self._observer.start()
        neural_status = "with neural perception" if self._context_engine else "basic"
        print(f"  Observer: running ({neural_status})")

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

        # Consolidate session into episodic memory
        if self._brain:
            summary = self._brain.consolidate_session()
            if summary:
                print(f"  Session saved: {summary}")

        self._observer.stop()
        self._hotkey.stop()
        self._overlay.hide()

        if self._engine:
            self._engine.unload()

        print("Goodbye!")
        sys.exit(0)
