# Installing Homie AI

## Quick Install (Python, all platforms)

```bash
pip install homie-ai
homie init
homie start
```

## Platform-Specific Packages

### Windows
Download `HomieSetup-{version}.msi` from [Releases](https://github.com/MSG-88/Homie/releases).
Double-click to install. Homie starts automatically on login.

### Linux (Debian/Ubuntu)
```bash
sudo dpkg -i homie-ai_{version}_amd64.deb
homie init
```

### Linux (Fedora/RHEL)
```bash
sudo rpm -i homie-ai-{version}-1.x86_64.rpm
homie init
```

### Linux (AppImage — Universal)
```bash
chmod +x HomieAI-{version}-x86_64.AppImage
./HomieAI-{version}-x86_64.AppImage init
```

### macOS
Download `HomieAI-{version}.dmg` from Releases.
Drag to Applications. Right-click > Open (first launch only).

### Android
Download from Google Play or install the APK from Releases.
The Android app connects to your desktop Homie as a Spoke.

### Docker (Headless/Server)
```bash
# Hub (with model support)
docker build --target full -t homie-hub .
docker run -d -p 8721:8721 -v homie-data:/data homie-hub

# Spoke (lightweight, remote inference)
docker build --target spoke -t homie-spoke .
docker run -d -p 8722:8721 -v spoke-data:/data homie-spoke

# Or use docker-compose for Hub + Spoke:
docker-compose up -d
```

## Mesh Setup

After installing on multiple machines:

1. On the most powerful machine: `homie start` (auto-becomes Hub)
2. On other machines: `homie start`, then `/mesh join --code <CODE>`
3. Get the code from Hub: `/mesh pair`
4. Check status: `/mesh health`

## Feature Groups

Install only what you need:

```bash
pip install homie-ai[voice]     # Voice: STT, TTS, wake word
pip install homie-ai[model]     # Local model: transformers, HF
pip install homie-ai[mesh]      # Distributed mesh
pip install homie-ai[app]       # Desktop: tray, hotkeys, API
pip install homie-ai[email]     # Gmail integration
pip install homie-ai[all]       # Everything
```

## System Requirements

| Mode | CPU | RAM | GPU | Disk |
|------|-----|-----|-----|------|
| Spoke (remote inference) | 2+ cores | 2 GB | None | 500 MB |
| Hub (small model, 3.8B) | 4+ cores | 8 GB | 4 GB VRAM | 5 GB |
| Hub (medium model, 9B) | 8+ cores | 16 GB | 8 GB VRAM | 10 GB |
| Hub (large model, 35B) | 16+ cores | 32 GB | 16 GB VRAM | 25 GB |
