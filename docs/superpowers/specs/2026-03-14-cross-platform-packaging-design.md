# Cross-Platform Packaging & Android App Design

**Date**: 2026-03-14
**Status**: Draft
**Scope**: Linux/macOS desktop packaging, Android native app, LAN sync, Qubrid inference fallback

---

## 1. Overview

Extend Homie AI distribution beyond Windows MSI to Linux (.deb, .rpm, AppImage), macOS (.dmg), and a native Android app (Kotlin/Jetpack Compose). Introduce a unified inference routing layer with Qubrid cloud fallback and LAN-based desktop↔phone sync.

**Distribution strategy**:
- **Developers**: `pip install homie-ai` on all desktop platforms
- **End users**: Native packages per platform
- **Android**: Google Play / sideload APK

---

## 2. Desktop Packaging (Linux & macOS)

### 2.1 Build System

Extend `installer/` with a unified `build.py` accepting `--target {deb,rpm,appimage,dmg,msi}`.

All targets use PyInstaller to freeze `homie` + `homie-daemon` binaries, then wrap them in platform-specific packages.

### 2.2 Linux — .deb (Ubuntu/Debian)

- PyInstaller freeze → `homie` + `homie-daemon` binaries
- Package with `dpkg-deb`:
  - Binaries: `/usr/local/bin/`
  - Config: `~/.homie/`
  - Desktop entry: `/usr/share/applications/homie.desktop`
- Systemd user service: `~/.config/systemd/user/homie.service` for `homie-daemon`
- Post-install script: creates `~/.homie/` dir, runs `homie init` on first launch

### 2.3 Linux — .rpm (Fedora)

- Same frozen binaries, packaged via `rpmbuild` spec file
- Same systemd service and paths as .deb

### 2.4 Linux — AppImage

- Bundle frozen output into AppImage using `appimagetool`
- Single portable file, no install required — distro-agnostic

### 2.5 macOS — .dmg

- PyInstaller freeze for macOS → `homie` + `homie-daemon` binaries
- Bundle into `.app` using PyInstaller `--windowed` + `BUNDLE` spec
- Wrap `.app` in `.dmg` via `create-dmg` (drag-to-Applications layout)
- LaunchAgent plist: `~/Library/LaunchAgents/com.heyhomie.daemon.plist`
- Keychain integration instead of Windows keyring

### 2.6 Platform-Specific Dependencies

In `pyproject.toml`:
- `pywin32` / `windows-toasts` → Windows only (already gated)
- `keyring.backends.macOS` → macOS (Keychain)
- `keyring.backends.SecretService` → Linux (GNOME Keyring / KWallet)
- `screen-reader` extras: `mss` + `Pillow` on all, drop `pywin32` on non-Windows

### 2.7 CI/CD

GitHub Actions matrix build:
- Matrix: `[ubuntu-latest, macos-latest, windows-latest]`
- Steps: build → test → package → upload as release artifacts
- Triggered on version tags (`v*`)

---

## 3. Android App — Architecture & Core

### 3.1 Project Structure

```
android/
├── app/
│   ├── src/main/
│   │   ├── kotlin/com/heyhomie/app/
│   │   │   ├── HomieApp.kt              # Application class
│   │   │   ├── ui/                       # Jetpack Compose screens
│   │   │   ├── core/                     # Brain, memory, config
│   │   │   ├── inference/                # llama.cpp JNI + Qubrid fallback
│   │   │   ├── phone/                    # Device analysis, accessors
│   │   │   ├── network/                  # LAN discovery & sync
│   │   │   ├── voice/                    # STT/TTS on Android
│   │   │   ├── email/                    # Gmail API (native)
│   │   │   └── notifications/            # Notification listener + engine
│   │   ├── cpp/                          # llama.cpp NDK build
│   │   └── res/                          # Retro pixel assets
│   └── build.gradle.kts
├── gradle/
└── settings.gradle.kts
```

### 3.2 On-Device Inference

- Bundle `llama.cpp` via NDK (CMake) — JNI bridge in `inference/LlamaBridge.kt`
- Ship a default small model (~1.5B Q4, ~800MB) downloadable on first launch
- User can download larger models (7B) if device has 8GB+ RAM
- `InferenceRouter.kt` — checks for local model → if none, routes to Qubrid API
- Silent fallback message: "No local model found! Using Homie's intelligence until local model is setup!"

### 3.3 Core Modules (Kotlin Reimplementation)

- **Brain** — prompt orchestration, tool registry, conversation management
- **Memory** — working (in-memory), episodic (Room DB), semantic (on-device embeddings via ONNX)
- **Config** — YAML config mirroring desktop, stored in app internal storage
- **Vault** — Android Keystore for credentials (replaces desktop AES vault + keyring)

### 3.4 Data Layer

- Room database for conversations, memory, email cache, phone analysis data
- DataStore (Preferences) for settings
- On-device vector store for semantic memory (lightweight, ONNX embeddings)

---

## 4. Android — Retro Pixel Theme & UI

### 4.1 Visual Identity

- Dark base (#0D0D0D) with pixel-art elements and 8-bit color palette
- Primary accent: electric green (#39FF14)
- Secondary: amber (#FFB000)
- Tertiary: cyan (#00E5FF)
- Pixel font for headers: Press Start 2P (Google Fonts)
- Clean monospace for body text
- Scanline overlay effect on backgrounds (subtle, toggleable)
- CRT screen glow on chat bubbles

### 4.2 UI Components

- **Chat screen**: Terminal-style with blinking cursor, typewriter animation for messages, pixel robot avatar for Homie
- **Navigation**: Bottom bar styled as retro game HUD — pixel art icons (chat, phone stats, network, settings)
- **Model manager**: Progress bars styled as retro loading bars with percentage in pixel font
- **Phone analysis dashboard**: Stats as retro game stat screens — HP bar for battery, XP bar for storage, pixel charts for usage
- **Notifications**: Toast-style popups with 8-bit sound effects (optional)
- **Connection screen**: Desktop pairing shown as "Player 2 has joined" animation

### 4.3 Animations

- Screen transitions: pixel dissolve / wipe effects
- Loading states: bouncing pixel dots or spinning pixel gear
- Haptic feedback on key interactions
- Boot screen: retro startup sequence with ASCII art Homie logo

### 4.4 Compose Theme

- Custom `HomieRetroTheme` wrapping Material3 with overridden colors, typography, shapes (sharp corners, no rounding)
- `PixelBorder` modifier for cards and containers (stepped/staircase borders instead of smooth)
- All icons from a custom pixel sprite sheet

---

## 5. Android — Phone Analysis & System Access

### 5.1 Hardware Profiling (`phone/DeviceProfiler.kt`)

- CPU: architecture, cores, clock speed, thermal state
- GPU: renderer, OpenGL/Vulkan support (determines inference acceleration)
- RAM: total, available, app usage
- Storage: total, free, app cache size, model storage
- Battery: level, health, temperature, charging state, drain rate
- Screen: resolution, density, refresh rate
- Sensors: accelerometer, gyroscope, proximity, ambient light
- Network: WiFi/cellular type, signal strength, bandwidth estimate
- Generates a "device capability score" to recommend appropriate model size

### 5.2 Email Integration (`email/`)

- Gmail API via Google Play Services (native Android OAuth, no redirect server needed)
- Sync, classify, organize — same logic as desktop, reimplemented in Kotlin
- Background sync via WorkManager (respects Doze mode, battery optimization)

### 5.3 Notification Intelligence (`notifications/`)

- `NotificationListenerService` — captures all incoming notifications from all apps
- Classifies by priority, category, sender
- Homie can summarize: "You got 12 notifications in the last hour — 3 important, 9 noise"
- User can ask: "What did I miss?" → Homie summarizes unread notifications

### 5.4 Message Access (`phone/MessageReader.kt`)

- SMS/MMS via Android content provider (`content://sms`, `content://mms`)
- User can ask Homie to search, summarize, or find messages
- Read-only access — Homie never sends messages without explicit user action

### 5.5 Usage Intelligence (`phone/UsageAnalyzer.kt`)

- `UsageStatsManager` — app usage time, launch counts, last used
- Screen time patterns, most used apps, daily/weekly trends
- Battery drain per app analysis
- Storage breakdown by category (apps, media, models, cache)
- Network data usage per app

### 5.6 Permissions

Progressive permission requests — ask only when features are first used:

| Permission | Purpose |
|---|---|
| `NOTIFICATION_LISTENER` | Notification access |
| `READ_SMS` | Message reading |
| `PACKAGE_USAGE_STATS` | App usage data |
| `READ_CONTACTS` | Contact-aware message/email context |
| `RECORD_AUDIO` | Voice input |
| `POST_NOTIFICATIONS` | Homie's own notifications |
| `ACCESS_NETWORK_STATE` | LAN discovery |
| `ACCESS_WIFI_STATE` | LAN discovery |

---

## 6. LAN Connection — Desktop ↔ Android Sync

### 6.1 Discovery

- **Desktop**: mDNS/DNS-SD service advertisement — broadcasts `_homie._tcp.local` on the LAN
- **Android** (`network/LanDiscovery.kt`): discovers nearby Homie instances automatically
- **Pairing**: one-time 6-digit code displayed on desktop, entered on phone (like Bluetooth pairing)
- After pairing, devices exchange Ed25519 public keys for future authenticated connections

### 6.2 Transport

- WebSocket connection over LAN (no internet required)
- All traffic encrypted with TLS (self-signed cert pinned during pairing)
- Auto-reconnect when devices rejoin the same network
- Heartbeat ping every 30s to detect disconnection

### 6.3 HomieSync Protocol

JSON-based messages over WebSocket:

| Message Type | Direction | Purpose |
|---|---|---|
| `inference_request` / `inference_response` | Phone → Desktop | Offload prompts to desktop GPU |
| `memory_sync` | Bidirectional | Merge episodic/semantic memory |
| `conversation_sync` | Bidirectional | Share chat history |
| `command` / `command_result` | Phone → Desktop | Remote control |
| `status` | Bidirectional | Device status (battery, model, daemon) |
| `file_transfer` | Bidirectional | Model files, config sync |

### 6.4 Inference Offloading

- Phone detects if desktop Homie is available on LAN
- If desktop has a larger/better model loaded, phone routes inference there
- Priority order: desktop LAN model → local on-device model → Qubrid cloud
- Latency-aware: if LAN round-trip > 500ms, falls back to local/Qubrid

### 6.5 Memory Sync

- Conflict resolution: last-write-wins with device ID + timestamp
- Sync on connect, then incremental changes via WebSocket stream
- Each device maintains a sync log (`sync_version` counter)
- User can choose sync scope in settings: all memory, conversations only, or manual

### 6.6 Remote Control from Phone

- View desktop Homie status (model loaded, daemon running, active tasks)
- Send voice/text commands to desktop Homie
- Trigger desktop actions: "read my latest emails", "summarize today's screen time"
- View desktop responses on phone

### 6.7 Desktop — New Module (`src/homie_core/network/`)

- `discovery.py` — mDNS advertisement via `zeroconf` library
- `server.py` — WebSocket server (uses existing FastAPI/uvicorn stack)
- `sync.py` — memory merge engine, conflict resolution
- `protocol.py` — message types, serialization
- New optional dependency: `zeroconf>=0.131`

---

## 7. Qubrid Fallback & Inference Routing

### 7.1 Unified Inference Router

**Desktop** (`src/homie_core/inference/router.py`):
- `check_local_model()` → llama.cpp GGUF loaded?
- `check_lan_devices()` → any paired device with better model?
- `check_qubrid()` → API key configured?
- `route(prompt)` → response from best available source

**Android** (`inference/InferenceRouter.kt`):
- Same priority chain, Kotlin implementation

### 7.2 Priority Order

1. **LAN desktop model** — if paired desktop is online with a larger model
2. **Local on-device model** — llama.cpp (desktop) or llama.cpp NDK (Android)
3. **Qubrid cloud** — OpenAI-compatible API at `platform.qubrid.com/v1`

### 7.3 Qubrid Integration

- OpenAI-compatible client (desktop: `openai` Python package, Android: OkHttp)
- Default model: `Qwen/Qwen3.5-Flash` (configurable in settings)
- API key stored in vault (desktop) / Android Keystore (phone)
- Silent fallback notification: "No local model found! Using Homie's intelligence until local model is setup!"
- Notification persists as a subtle banner in chat, not a blocking dialog
- Settings page shows current inference source: "Local", "Desktop (LAN)", or "Homie Intelligence (Cloud)"

### 7.4 First-Launch Flow

1. No local model exists → show banner, start using Qubrid immediately
2. Background prompt: "Download a local model for offline use?" with recommended size based on device profiling
3. Model downloads in background, Qubrid continues serving until ready
4. Once local model loads → banner disappears, switches seamlessly

### 7.5 Config Additions (`homie.config.yaml`)

```yaml
inference:
  priority: [lan, local, qubrid]
  qubrid:
    enabled: true
    model: "Qwen/Qwen3.5-Flash"
    # api_key stored in vault, not config
  lan:
    prefer_desktop: true
    max_latency_ms: 500
```

---

## 8. New Files & Modifications Summary

### 8.1 Desktop — New Files

| Path | Purpose |
|---|---|
| `installer/build.py` | Unified build script with `--target` flag |
| `installer/linux/homie.desktop` | Linux desktop entry |
| `installer/linux/homie.service` | Systemd user service |
| `installer/linux/build_deb.py` | .deb packaging |
| `installer/linux/build_rpm.py` | .rpm packaging |
| `installer/linux/build_appimage.py` | AppImage packaging |
| `installer/macos/build_dmg.py` | .dmg packaging |
| `installer/macos/com.heyhomie.daemon.plist` | LaunchAgent for daemon |
| `installer/macos/homie.spec` | macOS PyInstaller spec |
| `src/homie_core/inference/router.py` | Unified inference routing |
| `src/homie_core/inference/qubrid.py` | Qubrid API client |
| `src/homie_core/network/discovery.py` | mDNS advertisement |
| `src/homie_core/network/server.py` | WebSocket sync server |
| `src/homie_core/network/sync.py` | Memory merge engine |
| `src/homie_core/network/protocol.py` | Sync protocol messages |
| `.github/workflows/release.yml` | Matrix CI/CD build |

### 8.2 Desktop — Modified Files

| Path | Change |
|---|---|
| `pyproject.toml` | Add `network` optional deps (`zeroconf`, `openai`), platform markers |
| `homie.config.yaml` | Add `inference` and `network` sections |
| `src/homie_core/brain/engine.py` | Use InferenceRouter instead of direct llama.cpp |
| `src/homie_app/daemon.py` | Initialize network discovery + sync server |

### 8.3 Android — New Project

Full Kotlin project at `android/` as described in Section 3.1.

---

## 9. Priority Order

1. **Linux .deb** packaging
2. **Linux .rpm** packaging
3. **Linux AppImage** packaging
4. **macOS .dmg** packaging
5. **Desktop inference router + Qubrid fallback**
6. **Desktop LAN network module**
7. **GitHub Actions CI/CD matrix**
8. **Android app — core + inference**
9. **Android app — retro pixel theme**
10. **Android app — phone analysis**
11. **Android app — LAN sync**
12. **Android app — email, voice, notifications (full parity)**
