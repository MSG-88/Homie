"""Main console loop — routes input to slash commands or brain chat."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

from homie_app.console.router import SlashCommandRouter


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

    def _bootstrap(self):
        """Load model + intelligence stack, register commands."""
        self._print("=" * 50)
        self._print("  Homie AI v0.1.0 — Interactive Console")
        self._print("=" * 50)

        # Initialize vault once for the session
        try:
            from homie_core.vault.secure_vault import SecureVault
            self._vault = SecureVault()
            self._vault.unlock()
        except Exception as e:
            self._print(f"  [-] Vault unavailable: {e}")

        self._print("\n[Loading model...]")
        from homie_app.cli import _load_model_engine
        self._engine, entry = _load_model_engine(self._config)
        if not self._engine:
            self._print("  No model found. The setup wizard will help you configure one.")
            return

        self._print("\n[Initializing intelligence...]")
        from homie_app.cli import _init_intelligence_stack
        self._wm, self._em, self._sm, tool_registry, rag, plugin_mgr = _init_intelligence_stack(self._config)

        from homie_core.brain.orchestrator import BrainOrchestrator
        from homie_app.prompts.system import build_system_prompt

        self._brain = BrainOrchestrator(
            model_engine=self._engine,
            working_memory=self._wm,
            episodic_memory=self._em,
            semantic_memory=self._sm,
            tool_registry=tool_registry if tool_registry.list_tools() else None,
            rag_pipeline=rag,
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

        self._print(
            f"\nHey{' ' + self._user_name if self._user_name != 'User' else ''}! "
            f"Type /help for commands or just start chatting.\n"
        )

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
        """Main input loop."""
        while True:
            try:
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
                continue

            # Chat with brain
            if not self._brain:
                self._print("Homie> No model loaded. Use /settings to configure one.\n")
                continue

            self._chat(user_input)

        self._shutdown()

    def _chat(self, user_input: str):
        """Send input to brain, stream response."""
        try:
            from homie_app.loading import CLILoadingSpinner

            spinner = CLILoadingSpinner(style="random")
            spinner.start()
            first_token = True
            try:
                for token in self._brain.process_stream(user_input):
                    if first_token:
                        spinner.stop()
                        sys.stdout.write("Homie> ")
                        sys.stdout.flush()
                        first_token = False
                    sys.stdout.write(token)
                    sys.stdout.flush()
            except Exception:
                if not first_token:
                    raise
                spinner.stop()
                sys.stdout.write(f"Homie> {self._brain.process(user_input)}")
                sys.stdout.flush()
            finally:
                if first_token:
                    spinner.stop()
            print("\n")
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
        print(text)
