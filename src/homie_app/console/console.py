"""Main console loop — routes input to slash commands or brain chat."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

from homie_app.console.router import SlashCommandRouter
from homie_app.console.rich_console import rc
from homie_app.console.boot import show_boot_screen, show_system_check, get_greeting
from homie_app.console.status_bar import print_status_bar
from homie_app.console.notification_queue import NotificationQueue, CliNotification


class Console:
    """Unified Homie console: wizard, slash commands, and chat in one loop."""

    def __init__(
        self,
        config,
        config_path: Optional[str] = None,
        no_voice: bool = False,
        no_tray: bool = False,
        skip_init: bool = False,
    ):
        self._config = config
        self._config_path = config_path
        self._no_voice = no_voice
        self._no_tray = no_tray
        self._router = SlashCommandRouter()
        self._brain = None
        self._wm = None
        self._sm = None
        self._em = None
        self._engine = None
        self._vault = None
        self._user_name = getattr(config, "user_name", None) or "User"

        if not skip_init:
            self._bootstrap()

    def _needs_wizard(self) -> bool:
        """Check if first-run wizard is needed."""
        cfg = self._config
        if not getattr(cfg.llm, "model_path", ""):
            return True
        if not getattr(cfg, "user_name", ""):
            return True
        return False

    def _run_wizard(self):
        """Run the init wizard inline in the console."""
        self._print("\nWelcome to Homie AI! Let's get you set up.\n")
        try:
            from homie_app.init import run_init
            run_init(auto=False, config_path=self._config_path)
            from homie_core.config import load_config
            self._config = load_config(self._config_path)
            self._user_name = self._config.user_name or "User"
        except Exception as e:
            self._print(f"Wizard error: {e}. You can configure manually with /settings.")

    def _bootstrap(self):
        """Load model + intelligence stack, register commands."""
        show_boot_screen(rc)

        # Check if wizard needed
        if self._needs_wizard():
            self._run_wizard()
            if self._needs_wizard():
                self._print("Setup incomplete. Use /settings to finish configuration.\n")
                self._register_commands()
                return

        # Initialize vault once for the session
        vault_ok = True
        vault_detail = "ok"
        try:
            from homie_core.vault.secure_vault import SecureVault
            self._vault = SecureVault()
            self._vault.unlock()
        except Exception as e:
            vault_ok = False
            vault_detail = str(e)

        self._print("\n[Loading model...]")
        from homie_app.cli import _load_model_engine
        self._engine, entry = _load_model_engine(self._config)
        model_ok = self._engine is not None
        model_detail = getattr(entry, "name", str(entry)) if entry else "none"

        self._print("\n[Initializing intelligence...]")
        from homie_app.cli import _init_intelligence_stack
        self._wm, self._em, self._sm, tool_registry, rag, plugin_mgr = _init_intelligence_stack(self._config)

        checks = [
            ("vault",   vault_ok,  vault_detail),
            ("model",   model_ok,  model_detail if model_ok else "not found"),
            ("memory",  self._wm is not None, "working memory loaded" if self._wm else "unavailable"),
            ("rag",     rag is not None, "pipeline ready" if rag else "disabled"),
        ]
        show_system_check(rc, checks)

        if not self._engine:
            self._print("  No model found. The setup wizard will help you configure one.")
            return

        # Register weather/news as Brain tools for conversational access
        try:
            from homie_core.brain.builtin_tools import register_intelligence_tools
            register_intelligence_tools(tool_registry, config=self._config, vault=self._vault)
        except Exception:
            pass

        from homie_core.brain.orchestrator import BrainOrchestrator
        from homie_app.prompts.system import build_system_prompt
        from homie_core.backend.local_filesystem import LocalFilesystemBackend
        from homie_app.middleware_factory import build_middleware_stack

        backend = LocalFilesystemBackend(root_dir=self._config.storage.path)
        tool_registry.set_context({"backend": backend})

        middleware_stack = build_middleware_stack(
            config=self._config,
            working_memory=self._wm,
            backend=backend,
        )

        self._brain = BrainOrchestrator(
            model_engine=self._engine,
            working_memory=self._wm,
            episodic_memory=self._em,
            semantic_memory=self._sm,
            tool_registry=tool_registry if tool_registry.list_tools() else None,
            rag_pipeline=rag,
            middleware_stack=middleware_stack,
        )

        known_facts = []
        if self._sm:
            try:
                facts = self._sm.get_facts(min_confidence=0.5)
                known_facts = [f["fact"] for f in facts[:10]]
            except Exception:
                pass

        system_prompt = build_system_prompt(
            user_name=self._user_name,
            known_facts=known_facts if known_facts else None,
        )
        self._brain.set_system_prompt(system_prompt)

        # Register all slash commands
        self._register_commands(plugin_mgr=plugin_mgr)

        greeting = get_greeting(self._user_name, len(known_facts))
        self._print(f"\n{greeting}\n")

    def _register_commands(self, **services):
        """Register all slash commands from console/commands/."""
        from homie_app.console.commands import register_all_commands
        register_all_commands(self._router, self._build_ctx(**services))

    def _build_ctx(self, **services) -> dict:
        """Build the context dict passed to all command handlers."""
        return {
            "config": self._config,
            "config_path": self._config_path,
            "brain": self._brain,
            "wm": self._wm,
            "sm": self._sm,
            "em": self._em,
            "engine": self._engine,
            "vault": self._vault,
            "_router": self._router,
            "console": self,
            **services,
        }

    def run(self):
        """Main input loop with autocomplete."""
        try:
            from prompt_toolkit import PromptSession
            from homie_app.console.autocomplete import HomieCompleter
            session = PromptSession(
                completer=HomieCompleter(self._router),
                complete_while_typing=False,
            )
            use_prompt_toolkit = True
        except ImportError:
            session = None
            use_prompt_toolkit = False

        while True:
            try:
                # Status bar before each prompt
                model_name = getattr(getattr(self._engine, "entry", None), "name", "") if self._engine else ""
                memory_count = 0
                if self._sm:
                    try:
                        memory_count = len(self._sm.get_facts(min_confidence=0.5))
                    except Exception:
                        pass
                project = getattr(self._config, "project", "") or ""
                print_status_bar(rc, model_name=model_name, memory_count=memory_count, project=project)

                if use_prompt_toolkit:
                    user_input = session.prompt(f"{self._user_name}> ").strip()
                else:
                    user_input = input(f"{self._user_name}> ").strip()
            except (EOFError, KeyboardInterrupt):
                self._print("\n")
                break

            if user_input.lower() in ("quit", "exit", ":q", "/quit"):
                break
            if not user_input:
                continue

            if user_input.startswith("/"):
                ctx = self._build_ctx()
                result = self._router.dispatch(user_input, **ctx)
                if result:
                    self._print(f"Homie> {result}\n")
                    # Record slash command interactions in working memory
                    # so sessions with only commands are still saved
                    if self._wm:
                        self._wm.add_message("user", user_input)
                        # Truncate long command output for memory storage
                        self._wm.add_message("assistant", result[:500])
                continue

            if not self._brain:
                self._print("Homie> No model loaded. Use /settings to configure one.\n")
                continue

            self._chat(user_input)

        self._shutdown()

    def _chat(self, user_input: str):
        """Send input to brain, stream response."""
        from rich.live import Live
        from rich.markdown import Markdown
        from rich.panel import Panel

        try:
            from homie_app.loading import CLILoadingSpinner

            spinner = CLILoadingSpinner(style="random")
            spinner.start()
            first_token = True
            buffer = ""
            try:
                with Live(console=rc, refresh_per_second=12, transient=False) as live:
                    for token in self._brain.process_stream(user_input):
                        if first_token:
                            spinner.stop()
                            first_token = False
                        buffer += token
                        live.update(Panel(Markdown(buffer), title="[homie.assistant]Homie[/]", border_style="homie.dim"))
            except Exception:
                if not first_token:
                    raise
                spinner.stop()
                response = self._brain.process(user_input)
                rc.print(Panel(Markdown(response), title="[homie.assistant]Homie[/]", border_style="homie.dim"))
            finally:
                if first_token:
                    spinner.stop()
            rc.print()
        except ConnectionError as e:
            self._print(f"Homie> Connection failed: {e}\n")
        except Exception as e:
            self._print(f"Homie> [Error: {e}]\n")

    def _shutdown(self):
        """Save session and clean up."""
        self._print("[Saving session...]")
        if self._brain:
            try:
                summary = self._brain.consolidate_session()
                if summary:
                    self._print(f"  Session saved: {summary}")
                else:
                    self._print("  Nothing to save.")
            except Exception:
                self._print("  Could not save session.")
        if self._engine:
            self._engine.unload()

    def _print(self, text: str):
        """Print to console. Abstracted for testability."""
        rc.print(text)
