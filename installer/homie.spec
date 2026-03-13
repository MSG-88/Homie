# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for Homie AI — two executables, one bundle."""

from pathlib import Path

import os
block_cipher = None
# SPECPATH is set by PyInstaller to the directory containing this .spec file
ROOT = Path(SPECPATH).parent
if ROOT.name == "installer":
    ROOT = ROOT.parent
# Fallback: if still wrong, use cwd
if not (ROOT / "src" / "homie_app" / "cli.py").exists():
    ROOT = Path(os.getcwd())

# ── Analysis for CLI entry point ─────────────────────────────────────────
a_cli = Analysis(
    [str(ROOT / "src" / "homie_app" / "cli.py")],
    pathex=[str(ROOT / "src")],
    binaries=[],
    datas=[
        (str(ROOT / "homie.config.yaml"), "."),
        (str(ROOT / "installer" / "icon.ico"), "."),
    ],
    hiddenimports=[
        # Core deps
        "pydantic", "pydantic_core", "pydantic.deprecated",
        "pydantic.deprecated.decorator",
        "cryptography", "cryptography.hazmat", "cryptography.hazmat.primitives",
        "cffi", "_cffi_backend",
        "keyring", "keyring.backends", "keyring.backends.Windows",
        "feedparser",
        "prompt_toolkit", "prompt_toolkit.shortcuts", "prompt_toolkit.history",
        "psutil", "requests", "yaml",
        # App modules
        "homie_app", "homie_app.cli", "homie_app.daemon",
        "homie_app.console", "homie_app.console.console",
        "homie_app.console.router",
        "homie_app.console.commands",
        "homie_app.tray", "homie_app.tray.app",
        "homie_app.wizard",
        "homie_app.service", "homie_app.service.scheduler_task",
        # Core modules
        "homie_core", "homie_core.brain",
        "homie_core.brain.engine", "homie_core.brain.tool_registry",
        "homie_core.config",
        "homie_core.memory", "homie_core.memory.working",
        "homie_core.memory.semantic", "homie_core.memory.episodic",
        "homie_core.email", "homie_core.email.gmail_provider",
        "homie_core.email.classifier", "homie_core.email.sync_engine",
        "homie_core.email.organizer", "homie_core.email.tools",
        "homie_core.context", "homie_core.context.file_indexer",
        "homie_core.hardware", "homie_core.hardware.detector",
        "homie_core.model", "homie_core.model.downloader",
        "homie_core.neural", "homie_core.neural.model_manager",
        "homie_core.notifications", "homie_core.notifications.toast",
        "homie_core.voice",
        "homie_core.storage", "homie_core.storage.database",
        "homie_core.storage.vectors",
        "homie_core.rag",
        "homie_core.screen_reader",
        "homie_core.plugins",
        "homie_core.social", "homie_core.messaging",
        # Optional deps (guarded imports, but include if available)
        "chromadb", "chromadb.config",
        "uvicorn", "uvicorn.lifespan", "uvicorn.lifespan.on",
        "fastapi", "fastapi.responses",
        "pystray", "PIL", "PIL.Image",
        "apscheduler", "apscheduler.schedulers.background",
        "apscheduler.triggers.cron",
        "numpy",
        "google.auth", "google.auth.transport.requests",
        "google.oauth2.credentials",
        "googleapiclient", "googleapiclient.discovery",
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "matplotlib", "scipy", "notebook", "jupyterlab",
        "pytest", "sphinx", "setuptools", "pip", "wheel",
        "tkinter", "_tkinter",
        "IPython", "ipykernel",
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

# ── Analysis for daemon entry point ──────────────────────────────────────
a_daemon = Analysis(
    [str(ROOT / "src" / "homie_app" / "daemon.py")],
    pathex=[str(ROOT / "src")],
    binaries=[],
    datas=[],
    hiddenimports=a_cli.hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=a_cli.excludes,
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

# ── PYZ archives ─────────────────────────────────────────────────────────
pyz_cli = PYZ(a_cli.pure, a_cli.zipped_data, cipher=block_cipher)
pyz_daemon = PYZ(a_daemon.pure, a_daemon.zipped_data, cipher=block_cipher)

# ── Executables ──────────────────────────────────────────────────────────
exe_cli = EXE(
    pyz_cli,
    a_cli.scripts,
    [],
    exclude_binaries=True,
    name="homie",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    icon=str(ROOT / "installer" / "icon.ico"),
)

exe_daemon = EXE(
    pyz_daemon,
    a_daemon.scripts,
    [],
    exclude_binaries=True,
    name="homie-daemon",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    icon=str(ROOT / "installer" / "icon.ico"),
)

# ── Single COLLECT — shared _internal/ ───────────────────────────────────
coll = COLLECT(
    exe_cli,
    a_cli.binaries,
    a_cli.zipfiles,
    a_cli.datas,
    exe_daemon,
    a_daemon.binaries,
    a_daemon.zipfiles,
    a_daemon.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="homie",
)
