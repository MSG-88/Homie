# Homie AI — Windows MSI Installer Design

**Date:** 2026-03-14
**Status:** Approved
**Author:** MSG + Claude

## Overview

Create a standalone Windows MSI installer for Homie AI that bundles Python, all dependencies (including optional extras like voice, email, neural), and both executables into a single `HomieSetup-0.2.0.msi`. Users double-click to install — no Python, pip, or terminal knowledge required.

## Approach

**PyInstaller** freezes the Python application into standalone executables. **WiX Toolset** wraps those into a proper Windows Installer (MSI) package.

## Build Pipeline

Three stages, orchestrated by a single `installer/build_msi.py` script:

```
Stage 1: PyInstaller
  pyinstaller installer/homie.spec
  → dist/homie/   (standalone folder: .exe files, _internal/ with runtime)

Stage 2: WiX v3.14
  heat.exe dir dist/homie/ → components.wxs   (auto-harvest file manifest)
  candle.exe product.wxs components.wxs       (compile to .wixobj)
  light.exe *.wixobj → HomieSetup-0.2.0.msi  (link into MSI)

Stage 3: Output
  dist/HomieSetup-0.2.0.msi
```

Version is extracted from `pyproject.toml` by `build_msi.py` using `tomllib` (stdlib in 3.11+) and passed to both PyInstaller (via `--name`) and WiX (via `-d Version=X.Y.Z` preprocessor variable).

## Entry Points

Two executables frozen from one spec file:

| Executable | Source | Console | Purpose |
|---|---|---|---|
| `homie.exe` | `homie_app.cli:main` | Yes | Interactive CLI console |
| `homie-daemon.exe` | `homie_app.daemon:run_daemon` | No (windowless) | Background daemon service |

Both share the same `_internal/` runtime directory via a single `COLLECT()` target — PyInstaller's `MERGE()` is deprecated, so instead we use two `Analysis()` + two `EXE()` targets collected into one output folder.

## Required Code Change: Daemon PID Detection

The daemon's `_acquire_lock()` in `daemon.py` currently checks `"python" in proc.name().lower()` to detect stale lock files. After PyInstaller freezing, the process name becomes `homie-daemon.exe`, not `python.exe`. This check must be updated to also match `homie-daemon` or use PID-only checking (just verify the PID is alive, regardless of process name).

## Install Layout

```
C:\Program Files\HomieAI\
├── homie.exe
├── homie-daemon.exe
├── _internal/             (PyInstaller runtime, DLLs, packages)
├── homie.config.yaml      (default config, copied to ~/.homie/ on first run)
└── icon.ico
```

## MSI Install Actions

1. Copy files to `C:\Program Files\HomieAI\`
2. Add install directory to **user PATH** (`HKCU\Environment\Path`) — does not require elevation, scoped to current user. Broadcasts `WM_SETTINGCHANGE` so running shells pick up the change. Idempotent: checks if entry already exists before appending.
3. Create Start Menu shortcuts:
   - "Homie AI Console" → `homie.exe`
   - "Homie AI (Background)" → `homie-daemon.exe --headless`
4. Register daemon auto-start via registry Run key (`HKCU\Software\Microsoft\Windows\CurrentVersion\Run`, value `HomieAI` → `"C:\Program Files\HomieAI\homie-daemon.exe" --headless`). This is simpler and more reliable than Task Scheduler custom actions and does not require elevation.
5. Optional desktop shortcut (checkbox in installer UI)

## MSI Uninstall Actions

1. Stop running daemon (kill process via PID file at `~/.homie/daemon.pid`)
2. Remove registry Run key (`HKCU\...\Run\HomieAI`)
3. Remove PATH entry from `HKCU\Environment\Path`, broadcast `WM_SETTINGCHANGE`
4. Delete `C:\Program Files\HomieAI\`
5. Leave `~/.homie/` intact (user data, config, memories, models)

## MSI Upgrade Behavior

- Fixed Upgrade GUID across versions
- New MSI automatically uninstalls previous version before installing
- Users just double-click the new MSI to upgrade

## PyInstaller Spec File

`installer/homie.spec`:

- Single spec file with two `Analysis()` + two `EXE()` targets, one `COLLECT()`
- `homie.exe`: `console=True` (interactive terminal)
- `homie-daemon.exe`: `console=False` (windowless background)
- `hiddenimports` for dynamically loaded modules:
  - `pydantic`, `pydantic_core` (compiled validators)
  - `cryptography`, `cffi` (C extensions)
  - `prompt_toolkit` (data files)
  - `uvicorn` (dynamic event loop imports)
- `--collect-all` for packages with data files:
  - `pystray`, `feedparser`, `prompt_toolkit`
  - `pydantic` (compiled validators and schema files)
  - `chromadb` (ONNX models, migration scripts)
  - `transformers` (model configs, tokenizer data)
  - `faster_whisper` (CTranslate2 shared libraries)
- Excludes: torch test data, transformers test modules, unused scipy/matplotlib

Note: The hidden imports and collect-all lists will require iterative testing — some dynamic imports only surface at runtime. The spec file should be treated as a living document during implementation.

## WiX MSI Definition

`installer/product.wxs`:

- Product: "Homie AI", version injected via preprocessor variable, manufacturer "MSG"
- UI: WiX minimal dialog set (welcome → license → install dir → progress → finish)
- License: MPL-2.0 in RTF format
- Components auto-harvested by `heat.exe` — never manually listed
- Custom actions for PATH modification and registry Run key
- WiX Toolset **v3.14** (final v3 release) — v4 has a different CLI interface

## Code Signing

Windows SmartScreen will show a warning for unsigned MSI files. Code signing is **deferred** — the initial release will be unsigned. A code signing certificate (e.g., from SignPath or Certum) can be added later. The `build_msi.py` script will accept an optional `--sign` flag with certificate path for when signing is available.

## File Structure

```
installer/
├── homie.spec          (PyInstaller spec file)
├── product.wxs         (WiX MSI definition)
├── build_msi.py        (orchestrator: deps → PyInstaller → WiX → MSI)
├── icon.ico            (application icon)
└── LICENSE.rtf         (MPL-2.0 for installer license dialog)
```

## Build Requirements

- Python 3.12 with all `[all]` extras installed
- PyInstaller (`pip install pyinstaller`)
- WiX Toolset v3.14 on PATH (`heat.exe`, `candle.exe`, `light.exe`)
- Windows 10/11 build machine

## Estimated Bundle Size

- ~150MB without voice/torch
- ~400-600MB with full `[all]` bundle (torch, whisper, etc.)

Exact size to be determined after first test freeze.

## Website Integration

The download page (`website/src/pages/download.astro`) will be updated to offer the MSI as the primary Windows download, with the pip install method as an alternative for developers.

## Future Considerations

- **Code signing** for SmartScreen trust
- **MSIX** as a modern alternative (supports auto-updates via App Installer, no Store requirement)
- **GitHub Actions CI** for automated MSI builds on Windows runners
