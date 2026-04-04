# Homie Desktop Companion

A Tauri v2 desktop GUI for Homie, your local AI assistant.

## Prerequisites

- **Rust** (stable): https://rustup.rs/
- **Tauri CLI v2**: `cargo install tauri-cli --version "^2"`
- **Homie daemon** running on `localhost:3141`
- **System dependencies** (Windows): WebView2 (ships with Windows 10/11)

## Project Structure

```
desktop/
├── src-tauri/          # Rust backend — Tauri v2
│   ├── Cargo.toml
│   ├── tauri.conf.json
│   └── src/main.rs     # System tray, window management, IPC
├── src/                # Frontend — vanilla HTML/CSS/JS
│   ├── index.html      # Chat interface
│   ├── styles.css      # Dark glassmorphism theme
│   ├── app.js          # Chat logic, WebSocket to daemon
│   ├── settings.js     # Settings panel logic
└── package.json
```

## Development

```bash
cd desktop

# Install Tauri CLI (one time)
cargo install tauri-cli --version "^2"

# Start Homie daemon first (in another terminal)
# python -m homie.daemon

# Run in dev mode (hot-reload frontend, compiles Rust)
cargo tauri dev
```

## Production Build

```bash
cd desktop
cargo tauri build
```

The installer will be in `src-tauri/target/release/bundle/`.

## Features

- **System tray** with show/hide, settings, and quit
- **Chat interface** with streaming responses via WebSocket
- **Markdown rendering** with code block highlighting
- **Voice toggle** for voice responses
- **Settings panel** — model selection, theme, plugins
- **Frameless window** with custom title bar
- **Always-on-top** toggle
- **Minimize to tray** on close

## Backend Connection

The UI connects to Homie's local daemon:
- HTTP: `http://localhost:3141/api/chat`
- WebSocket: `ws://localhost:3141/ws`

Make sure the Homie daemon is running before launching the desktop app.
