from __future__ import annotations

import logging
import os
import signal
import sys
from pathlib import Path
from typing import Iterator, Optional

logger = logging.getLogger(__name__)

from homie_core.config import HomieConfig, load_config
from homie_core.enterprise import load_enterprise_policy, apply_policy
from homie_core.voice.voice_manager import VoiceManager
from homie_core.voice.voice_pipeline import PipelineState
from homie_core.intelligence.task_graph import TaskGraph
from homie_core.intelligence.proactive_retrieval import ProactiveRetrieval
from homie_core.intelligence.interruption_model import InterruptionModel
from homie_core.intelligence.session_tracker import SessionTracker
from homie_core.intelligence.briefing import BriefingGenerator
from homie_core.intelligence.observer_loop import ObserverLoop
from homie_core.intelligence.audit_log import AuditLogger
from homie_core.intelligence.flow_detector import FlowDetector
from homie_core.intelligence.workflow_predictor import WorkflowPredictor
from homie_core.intelligence.proactive_engine import ProactiveEngine
from homie_core.memory.working import WorkingMemory
from homie_core.brain.orchestrator import BrainOrchestrator
from homie_core.rag.pipeline import RagPipeline
from homie_core.scheduler.cron import Scheduler, JobStore, Job
from homie_core.skills.loader import SkillLoader
from homie_app.hotkey import HotkeyListener
from homie_app.overlay import OverlayPopup
from homie_app.prompts.system import build_system_prompt
from homie_core.vault.secure_vault import SecureVault
from homie_core.vault.sync_manager import SyncManager as VaultSyncManager


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

        # RAG pipeline for document search
        self._rag = RagPipeline()
        self._rhythm_model = None
        self._behavioral_profile = None
        self._preference_engine = None

        # Scheduler for recurring tasks
        scheduler_dir = storage / "scheduler"
        self._job_store = JobStore(path=scheduler_dir / "jobs.json")
        self._scheduler = Scheduler(
            job_store=self._job_store,
            on_job_due=self._execute_scheduled_job,
        )

        # Skill loader for user-defined skill files
        self._skill_loader = SkillLoader()

        # Secure vault for credentials and encrypted data
        vault_dir = storage / "vault"
        self._vault = SecureVault(storage_dir=vault_dir)
        try:
            self._vault.unlock()
            print("  Vault: unlocked")
        except Exception as e:
            print(f"  Vault: failed to unlock ({e})")

        # Auto-provision internal cloud AI key on first run
        try:
            from homie_core.cloud_ai_provision import provision as _provision_cloud
            if _provision_cloud(self._vault):
                print("  Cloud AI: provisioned")
        except Exception:
            pass

        # Vault sync manager (callbacks registered by sub-projects)
        self._vault_sync = VaultSyncManager(vault=self._vault)

        # Initialize email service if Gmail is connected
        self._email_service = None
        try:
            gmail_creds = self._vault.list_credentials("gmail")
            active_gmail = [c for c in gmail_creds if c.active]
            if active_gmail:
                from homie_core.email import EmailService
                from homie_core.vault.schema import create_cache_db
                cache_path = storage / "cache.db"
                cache_conn = create_cache_db(cache_path)
                self._email_service = EmailService(
                    vault=self._vault, cache_conn=cache_conn,
                    working_memory=self._working_memory,
                )
                accounts = self._email_service.initialize()
                if accounts:
                    self._vault_sync.register_callback("gmail", self._email_service.sync_tick)
                    print(f"  Email: {len(accounts)} Gmail account(s) connected")
        except Exception as e:
            print(f"  Email: not available ({e})")

        # Initialize financial service
        self._financial_service = None
        try:
            from homie_core.financial import FinancialService
            self._financial_service = FinancialService(
                vault=self._vault, working_memory=self._working_memory,
            )
            self._vault_sync.register_callback("financial", self._financial_service.reminder_tick)
            print("  Financial: active")
        except Exception as e:
            print(f"  Financial: not available ({e})")

        # Initialize folder service
        self._folder_service = None
        try:
            from homie_core.folders import FolderService
            from homie_core.vault.schema import create_cache_db
            cache_path = storage / "cache.db"
            folder_conn = create_cache_db(cache_path)
            self._folder_service = FolderService(cache_conn=folder_conn)
            watches = self._folder_service.list_watches()
            if watches:
                self._vault_sync.register_callback("folders", self._folder_service.scan_tick)
                print(f"  Folders: {len(watches)} folder(s) watched")
        except Exception as e:
            print(f"  Folders: not available ({e})")

        # Initialize social service if Slack is connected
        self._social_service = None
        try:
            slack_creds = self._vault.list_credentials("slack")
            active_slack = [c for c in slack_creds if c.active and c.account_id != "oauth_client"]
            if active_slack:
                from homie_core.social import SocialService
                self._social_service = SocialService(
                    vault=self._vault, working_memory=self._working_memory,
                )
                workspaces = self._social_service.initialize()
                if workspaces:
                    self._vault_sync.register_callback("slack", self._social_service.sync_tick)
                    print(f"  Social: {len(workspaces)} Slack workspace(s) connected")
        except Exception as e:
            print(f"  Social: not available ({e})")

        # Initialize web analyzer (shared utility)
        self._web_analyzer = None
        try:
            from homie_core.web.analyzer import WebAnalyzer
            self._web_analyzer = WebAnalyzer()
            print("  Web Analyzer: active")
        except Exception as e:
            print(f"  Web Analyzer: not available ({e})")

        # Initialize social media service
        self._social_media_service = None
        try:
            from homie_core.social_media import SocialMediaService
            self._social_media_service = SocialMediaService(
                vault=self._vault, working_memory=self._working_memory,
            )
            platforms = self._social_media_service.initialize()
            if platforms:
                self._vault_sync.register_callback(
                    "social_media", self._social_media_service.sync_tick,
                )
                print(f"  Social Media: {', '.join(platforms)}")
            else:
                print("  Social Media: no platforms connected")
        except Exception as e:
            print(f"  Social Media: not available ({e})")

        # Initialize browser history service
        self._browser_service = None
        try:
            from homie_core.browser import BrowserHistoryService
            self._browser_service = BrowserHistoryService(
                vault=self._vault,
                working_memory=self._working_memory,
                web_analyzer=self._web_analyzer,
            )
            status = self._browser_service.initialize()
            if status.get("enabled"):
                self._vault_sync.register_callback(
                    "browser", self._browser_service.sync_tick,
                )
                print(f"  Browser History: {', '.join(status.get('browsers', []))}")
            else:
                print("  Browser History: disabled (run 'homie browser enable')")
        except Exception as e:
            print(f"  Browser History: not available ({e})")

        # Voice
        self._voice_manager: Optional[VoiceManager] = None
        if self._config.voice.enabled:
            try:
                self._voice_manager = VoiceManager(
                    config=self._config.voice,
                    on_query=self._on_user_query_stream,
                    on_state_change=self._on_voice_state,
                )
            except Exception:
                logger.warning("Voice initialization failed, continuing without voice")

        # Try to initialize neural components with HF embeddings
        self._init_neural_components()

        # Screen reader
        self._screen_scheduler = None
        self._init_screen_reader()

        # Notifications
        self._notification_router = None
        self._toast_notifier = None
        self._init_notifications()

        # Proactive suggestion engine
        self._proactive_engine = ProactiveEngine(
            working_memory=self._working_memory,
            interruption_model=self._interruption_model,
        )

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
            proactive_engine=self._proactive_engine,
        )

        # UI components
        self._overlay = OverlayPopup(
            on_submit=self._on_user_query,
            on_submit_stream=self._on_user_query_stream,
        )
        hotkey_str = self._config.voice.hotkey if self._config.voice.enabled else "alt+8"
        self._hotkey = HotkeyListener(hotkey=hotkey_str, callback=self._on_hotkey)

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

    def _init_screen_reader(self) -> None:
        if not self._config.screen_reader.enabled:
            return
        try:
            from homie_core.screen_reader.capture_scheduler import CaptureScheduler
            from homie_core.screen_reader.window_tracker import WindowTracker
            from homie_core.screen_reader.pii_filter import PIIFilter
            from homie_core.screen_reader.ocr_reader import OCRReader
            from homie_core.screen_reader.visual_analyzer import VisualAnalyzer

            pii = PIIFilter()
            tracker = WindowTracker(blocklist=self._config.screen_reader.blocklist)
            ocr = OCRReader(pii_filter=pii) if self._config.screen_reader.level >= 2 else None
            visual = None
            if self._config.screen_reader.level >= 3:
                visual = VisualAnalyzer(
                    engine=self._config.screen_reader.analysis_engine,
                    api_base_url=self._config.llm.api_base_url,
                    api_key=self._config.llm.api_key,
                )
            self._screen_scheduler = CaptureScheduler(
                config=self._config.screen_reader,
                window_tracker=tracker,
                ocr_reader=ocr,
                visual_analyzer=visual,
            )
        except Exception:
            logger.warning("Screen reader initialization failed", exc_info=True)

    def _init_notifications(self) -> None:
        if not self._config.notifications.enabled:
            return
        try:
            from homie_core.notifications.router import NotificationRouter
            from homie_core.notifications.toast import ToastNotifier
            self._notification_router = NotificationRouter(config=self._config.notifications)
            self._toast_notifier = ToastNotifier()
        except Exception:
            logger.warning("Notification system initialization failed", exc_info=True)

    def _execute_scheduled_job(self, job: Job) -> str:
        """Callback for the scheduler — processes a due job's prompt."""
        if not self._ensure_brain():
            return "Model not available for scheduled task."
        try:
            return self._brain.process(job.prompt)
        except Exception as e:
            return f"Scheduled task error: {e}"

    def _on_hotkey(self) -> None:
        if self._voice_manager:
            self._voice_manager.on_hotkey()
        elif self._overlay:
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
                rag_pipeline=self._rag,
            )

            # Register email tools if service is available
            if self._email_service:
                from homie_core.email.tools import register_email_tools
                register_email_tools(tool_registry, self._email_service)

            # Register financial tools
            if self._financial_service:
                from homie_core.financial.tools import register_financial_tools
                register_financial_tools(tool_registry, self._financial_service)

            # Register folder tools
            if self._folder_service:
                from homie_core.folders.tools import register_folder_tools
                register_folder_tools(tool_registry, self._folder_service)

            # Register social tools
            if self._social_service:
                from homie_core.social.tools import register_social_tools
                register_social_tools(tool_registry, self._social_service)

            if self._web_analyzer:
                from homie_core.web.tools import register_web_tools
                register_web_tools(tool_registry, self._web_analyzer)
            if self._social_media_service:
                from homie_core.social_media.tools import register_social_media_tools
                register_social_media_tools(tool_registry, self._social_media_service)
            if self._browser_service:
                from homie_core.browser.tools import register_browser_tools
                register_browser_tools(tool_registry, self._browser_service)

            self._brain = BrainOrchestrator(
                model_engine=self._engine,
                working_memory=self._working_memory,
                tool_registry=tool_registry,
                rag_pipeline=self._rag,
            )
            # Dynamic system prompt with user name and time awareness
            system_prompt = build_system_prompt(
                user_name=self._config.user_name or "Master",
            )
            self._brain.set_system_prompt(system_prompt)
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

    def _on_voice_state(self, state: PipelineState) -> None:
        logger.debug("Voice state: %s", state.value)

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

            # Internal cloud AI fallback — if no model is configured,
            # try the internal cloud backend stored in the vault.
            if not entry:
                entry = self._resolve_internal_cloud(entry)

            if entry:
                kwargs = {}
                if entry.format == "hf":
                    kwargs["api_key"] = self._config.llm.api_key or os.environ.get("HF_KEY", "")
                elif entry.format == "cloud":
                    kwargs["api_key"] = self._config.llm.api_key or entry.repo_id
                    kwargs["base_url"] = self._config.llm.api_base_url or getattr(entry, "_base_url", "https://api.openai.com/v1")
                else:
                    kwargs["n_ctx"] = self._config.llm.context_length
                    kwargs["n_gpu_layers"] = self._config.llm.gpu_layers
                engine.load(entry, **kwargs)
                self._engine = engine
        except Exception:
            self._engine = None

    def _resolve_internal_cloud(self, current_entry) -> Optional["ModelEntry"]:
        """Try to resolve an internal cloud AI backend from the vault."""
        try:
            from homie_core.cloud_ai import get_cloud_config
            from homie_core.model.registry import ModelEntry

            cloud_cfg = get_cloud_config(self._vault)
            if cloud_cfg:
                entry = ModelEntry(
                    name="internal-cloud",
                    path=cloud_cfg["model"],
                    format="cloud",
                    params="cloud",
                    repo_id=cloud_cfg["api_key"],
                )
                # Stash base_url so _load_engine can pick it up
                entry._base_url = cloud_cfg["base_url"]
                # Also set on config so downstream code sees it
                self._config.llm.api_key = cloud_cfg["api_key"]
                self._config.llm.api_base_url = cloud_cfg["base_url"]
                self._config.llm.backend = "cloud"
                return entry
        except Exception:
            pass
        return current_entry

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

        if self._voice_manager:
            self._voice_manager.start()
            logger.info("Voice pipeline started")

        # Start hotkey listener
        self._hotkey.start()
        print("  Hotkey (Alt+8): active")

        # Load skills index
        skills = self._skill_loader.scan()
        if skills:
            print(f"  Skills: {len(skills)} loaded from ~/.homie/skills/")

        # Show pending scheduled jobs
        pending = self._job_store.list_jobs()
        if pending:
            print(f"  Scheduler: {len(pending)} jobs registered")

        print("\nHomie is running in the background. Press Alt+8 or say 'hey homie' to activate.")
        print("Press Ctrl+C to stop.\n")

        # Main thread waits — ticks scheduler every 60s
        try:
            signal.signal(signal.SIGINT, lambda *_: self.stop())
            tick_counter = 0
            while self._running:
                import time
                time.sleep(1)
                tick_counter += 1
                if tick_counter >= 60:
                    tick_counter = 0
                    try:
                        results = self._scheduler.tick()
                        for job, output in results:
                            print(f"  [Scheduler] {job.name}: {output[:100]}")
                    except Exception:
                        pass
                    try:
                        if self._vault_sync:
                            sync_results = self._vault_sync.tick()
                            for provider, output in sync_results:
                                print(f"  [Sync] {provider}: {output[:100]}")
                    except Exception:
                        pass
        except KeyboardInterrupt:
            self.stop()

    def stop(self) -> None:
        print("\nHomie daemon stopping...")
        self._running = False
        if self._voice_manager:
            self._voice_manager.stop()

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

        # Close services
        if self._email_service:
            self._email_service = None
        if self._financial_service:
            self._financial_service = None
        if self._folder_service:
            self._folder_service = None
        if self._social_service:
            self._social_service = None

        # Lock vault and stop sync
        self._vault_sync = None
        self._vault.lock()

        self._observer.stop()
        self._hotkey.stop()
        self._overlay.hide()

        if self._engine:
            self._engine.unload()

        print("Goodbye!")
        sys.exit(0)
