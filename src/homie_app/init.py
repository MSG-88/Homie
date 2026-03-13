from __future__ import annotations

from pathlib import Path

from homie_core.config import HomieConfig, load_config
from homie_core.hardware.detector import detect_hardware
from homie_core.hardware.profiles import recommend_model, discover_local_model
from homie_core.model.registry import ModelRegistry

import yaml


# Known cloud providers with their API endpoints
CLOUD_PROVIDERS = [
    {"name": "OpenAI", "base_url": "https://api.openai.com/v1"},
    {"name": "Groq", "base_url": "https://api.groq.com/openai/v1"},
    {"name": "Together", "base_url": "https://api.together.xyz/v1"},
    {"name": "DeepSeek", "base_url": "https://api.deepseek.com/v1"},
    {"name": "OpenRouter", "base_url": "https://openrouter.ai/api/v1"},
]

# Fallback model presets when /v1/models discovery fails
PRESET_MODELS = [
    "gpt-4o",
    "gpt-4o-mini",
    "gpt-4.1",
    "claude-sonnet-4",
    "deepseek-chat",
    "llama-3.1-70b",
    "mixtral-8x7b",
]


def _discover_cloud_models(api_key: str, base_url: str) -> list[str]:
    """Try to discover available models from the cloud API."""
    from homie_core.model.cloud_backend import CloudBackend
    backend = CloudBackend()
    return backend.discover_models(api_key=api_key, base_url=base_url)


def _ask_choice(prompt: str, options: list[str]) -> int:
    """Ask the user to pick from a numbered list. Returns 0-based index."""
    print(prompt)
    for i, opt in enumerate(options, 1):
        print(f"  {i}. {opt}")
    while True:
        try:
            raw = input("  > ").strip()
            idx = int(raw) - 1
            if 0 <= idx < len(options):
                return idx
        except (ValueError, EOFError, KeyboardInterrupt):
            pass
        print(f"  Please enter a number between 1 and {len(options)}.")


def _setup_cloud(cfg: HomieConfig) -> None:
    """Interactive cloud API configuration."""
    # Pick provider
    provider_names = [p["name"] for p in CLOUD_PROVIDERS] + ["Custom endpoint"]
    idx = _ask_choice("\n  Select cloud provider:", provider_names)

    if idx < len(CLOUD_PROVIDERS):
        base_url = CLOUD_PROVIDERS[idx]["base_url"]
        print(f"  Endpoint: {base_url}")
    else:
        base_url = input("  Enter API endpoint URL (e.g. https://api.example.com/v1): ").strip()

    cfg.llm.api_base_url = base_url

    # Get API key
    api_key = input("  Enter your API key: ").strip()
    cfg.llm.api_key = api_key

    # Discover or pick model
    print("  Discovering available models...")
    models = _discover_cloud_models(api_key, base_url)

    if models:
        # Filter out embedding models
        chat_models = [m for m in models if "embed" not in m.lower()]
        if not chat_models:
            chat_models = models
        # Show up to 15 models
        display_models = chat_models[:15]
        if len(chat_models) > 15:
            display_models.append(f"... and {len(chat_models) - 15} more")
        display_models.append("Enter model name manually")
        idx = _ask_choice("  Select a model:", display_models)

        if idx < len(chat_models) and idx < 15:
            cfg.llm.model_path = chat_models[idx]
        else:
            cfg.llm.model_path = input("  Enter model name: ").strip()
    else:
        print("  Could not auto-discover models. Showing presets.")
        preset_options = PRESET_MODELS + ["Enter model name manually"]
        idx = _ask_choice("  Select a model:", preset_options)

        if idx < len(PRESET_MODELS):
            cfg.llm.model_path = PRESET_MODELS[idx]
        else:
            cfg.llm.model_path = input("  Enter model name: ").strip()

    cfg.llm.backend = "cloud"
    print(f"  Selected: {cfg.llm.model_path}")


def _step_notifications(cfg: HomieConfig) -> None:
    """Configure notification preferences (settings-only, not in init wizard)."""
    print("\n  Notification Settings")
    print(f"  Currently: {'enabled' if cfg.notifications.enabled else 'disabled'}")
    for cat, enabled in cfg.notifications.categories.items():
        status = "on" if enabled else "off"
        label = cat.replace("_", " ").title()
        toggle = input(f"  {label} [{status}]: ").strip().lower()
        if toggle in ("on", "off"):
            cfg.notifications.categories[cat] = toggle == "on"

    dnd = input(f"  DND schedule [{cfg.notifications.dnd_schedule_start}-{cfg.notifications.dnd_schedule_end}] (y to change/n): ").strip().lower()
    if dnd == "y":
        cfg.notifications.dnd_schedule_enabled = True
        cfg.notifications.dnd_schedule_start = input("  DND start time (HH:MM): ").strip() or "22:00"
        cfg.notifications.dnd_schedule_end = input("  DND end time (HH:MM): ").strip() or "07:00"


def _save_config(cfg: HomieConfig, path: str) -> None:
    """Save config to YAML file."""
    import yaml
    from pathlib import Path
    data = cfg.model_dump()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)


def _step_user_profile(cfg: HomieConfig) -> None:
    """Step 5: User profile."""
    cfg.user.name = input("  What should I call you? [Master]: ").strip() or "Master"


def _step_screen_reader(cfg: HomieConfig) -> None:
    """Step 6: Screen reader consent."""
    print("\n  Screen awareness helps me understand what you're working on.")
    choice = _ask_choice("  Choose your comfort level:", [
        "Window titles only — I see app names, nothing more",
        "Window titles + text reading — I can read on-screen text (PII filtered)",
        "Full visual awareness — I can see and understand your screen (PII filtered)",
        "Off — I only know what you tell me",
    ])
    if choice < 3:
        cfg.screen_reader.enabled = True
        cfg.screen_reader.level = choice + 1
        cfg.privacy.screen_reader_consent = True
    else:
        cfg.screen_reader.enabled = False


def _step_email(cfg: HomieConfig) -> None:
    """Step 7: Email connection (Gmail OAuth)."""
    connect = input("  Connect Gmail? (y/n) [n]: ").strip().lower()
    if connect == "y":
        print("  Gmail OAuth setup will be handled through homie settings.")
        cfg.connections.gmail.connected = True
    else:
        cfg.connections.gmail.connected = False


def _step_social_connections(cfg: HomieConfig) -> None:
    """Step 8: Social connections (all platforms)."""
    print("\n  Social connections can be set up later in homie settings.")


def _step_privacy(cfg: HomieConfig) -> None:
    """Step 9: Privacy preferences."""
    print("\n  Privacy preferences use secure defaults.")


def _step_plugins(cfg: HomieConfig) -> None:
    """Step 10: Plugin selection."""
    print("\n  Plugins can be configured in homie settings.")


def _step_summary(cfg: HomieConfig) -> None:
    """Step 11: Summary of configuration."""
    print("\n  Configuration Summary:")
    print(f"    Name: {cfg.user.name}")
    print(f"    Voice: {'enabled' if cfg.voice.enabled else 'disabled'}")
    print(f"    Screen Reader: {'level ' + str(cfg.screen_reader.level) if cfg.screen_reader.enabled else 'off'}")
    print(f"    LLM Backend: {cfg.llm.backend}")


def _step_service_mode(cfg: HomieConfig) -> None:
    """Step 12: Service mode."""
    choice = _ask_choice("  How should Homie run?", [
        "On-demand — I start when you run 'homie start'",
        "Windows Service — I start on login and run in the background",
    ])
    cfg.service.mode = "windows_service" if choice == 1 else "on_demand"


def _detect_existing_config(path: str) -> tuple[bool, dict]:
    """Check for existing config file."""
    import yaml
    from pathlib import Path
    p = Path(path)
    if p.exists():
        with open(p) as f:
            return True, yaml.safe_load(f) or {}
    return False, {}


def run_init(auto: bool = False, config_path: str | None = None) -> HomieConfig:
    print("=" * 50)
    print("  Welcome to Homie AI Setup")
    print("=" * 50)

    # Step 1: Detect hardware
    print("\n[1/7] Detecting hardware...")
    hw = detect_hardware()
    print(f"  OS: {hw.os_name} {hw.os_version}")
    print(f"  CPU: {hw.cpu_cores} cores")
    print(f"  RAM: {hw.ram_gb} GB")
    if hw.gpus:
        for gpu in hw.gpus:
            print(f"  GPU: {gpu.name} ({gpu.vram_mb} MB VRAM)")
    else:
        print("  GPU: None detected")
    print(f"  Microphone: {'Yes' if hw.has_microphone else 'No'}")

    cfg = HomieConfig()

    # Step 2: Local or Cloud?
    if not auto:
        idx = _ask_choice(
            "\n[2/7] How would you like to run your AI model?",
            ["Local model (GGUF on this machine)", "Cloud API (OpenAI, Groq, Together, etc.)"],
        )
        use_cloud = idx == 1
    else:
        use_cloud = False

    if use_cloud:
        # Step 3: Cloud setup
        print("\n[3/7] Cloud API configuration...")
        _setup_cloud(cfg)

        # Step 4: Skip local model discovery
        print("\n[4/7] Skipping local model search (using cloud).")
        model_path = None
        rec = None
    else:
        # Step 3: Recommend model
        print("\n[3/7] Selecting optimal model...")
        rec = recommend_model(hw.best_gpu_vram_gb)
        print(f"  Recommended: {rec['model']} ({rec['quant']}, {rec['format']})")
        print(f"  Backend: {rec['backend']}")
        print(f"  Context length: {rec['context_length']:,} tokens")

        # Step 4: Discover existing local model files
        print("\n[4/7] Searching for existing model files...")
        model_path = discover_local_model(rec["model"])
        if model_path:
            print(f"  Found: {model_path}")
        else:
            print(f"  No local copy of {rec['model']} found.")
            print(f"  You can download it later with: homie model download {rec.get('repo_id', '')}")

        cfg.llm.backend = rec["format"]
        cfg.llm.context_length = rec["context_length"]
        cfg.llm.repo_id = rec.get("repo_id", "")
        if model_path:
            cfg.llm.model_path = model_path
        print(f"  Backend: {rec['format']} (direct loading via llama-server)")

    # Step 5: Voice setup
    cfg.voice.enabled = hw.has_microphone
    if hw.has_microphone:
        cfg.voice.mode = "push_to_talk"
    print(f"\n[5/7] Voice: {'push-to-talk' if hw.has_microphone else 'disabled (no microphone)'}")

    # Step 6: Get user name
    if not auto:
        try:
            name = input("\n[6/7] What should I call you? ").strip()
            if name:
                cfg.user_name = name
        except (EOFError, KeyboardInterrupt):
            pass
    else:
        print("\n[6/7] Skipping (auto mode)")

    # Step 7: Save config & register model
    print("\n[7/7] Saving configuration...")
    storage_path = Path(cfg.storage.path)
    storage_path.mkdir(parents=True, exist_ok=True)

    config_file = Path("homie.config.yaml")
    config_data = cfg.model_dump()
    config_file.write_text(yaml.dump(config_data, default_flow_style=False))
    print(f"  Config saved to {config_file}")
    print(f"  Data directory: {storage_path}")

    # Initialize model registry and register model
    registry = ModelRegistry(storage_path / cfg.storage.models_dir)
    registry.initialize()

    if use_cloud:
        entry = registry.register(
            name=cfg.llm.model_path,
            path=cfg.llm.model_path,
            format="cloud",
            params="cloud",
        )
        registry.set_active(cfg.llm.model_path)
        print(f"  Cloud model registered: {cfg.llm.model_path} (active)")
    elif model_path and rec:
        entry = registry.register(
            name=rec["model"],
            path=model_path,
            format=rec["format"],
            params="35B-A3B",
            repo_id=rec.get("repo_id", ""),
            quant=rec["quant"],
        )
        registry.set_active(rec["model"])
        print(f"  Model registered: {rec['model']} (active)")

    print("\n" + "=" * 50)
    print("  Setup complete! Run 'homie start' to begin.")
    print("=" * 50)

    return cfg
