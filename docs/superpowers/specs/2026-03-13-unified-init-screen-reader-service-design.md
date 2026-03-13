# Unified Init, Screen Reader & Windows Service Design

**Date:** 2026-03-13
**Status:** Approved
**Scope:** Init wizard redesign, screen reader system, Windows service mode, microphone permission gate, command consolidation, OAuth config updates, messaging platform integrations

---

## 1. Problem Statement

Three core issues degrade the non-technical user experience:

1. **Social/email auth is developer-hostile** — Every provider requires users to create an OAuth app in a developer console, obtain a Client ID + Secret, and paste them in. Non-technical users cannot do this without hand-holding.
2. **Voice defaults to disabled** — The init wizard only enables voice if a microphone is detected at setup time, with no attempt to request OS-level permission.
3. **Setup wizard misses key connections** — The 7-step init handles hardware + LLM + voice toggle + username, but email/social connections are separate `homie connect <provider>` commands users must discover independently.

Additional goals:
- Add screen awareness (window tracking, OCR, visual analysis) for contextual help
- Support running Homie as a Windows background service with toast notifications
- Add Telegram, WhatsApp (experimental), and Phone Link SMS/RCS integrations
- Consolidate all CLI commands into the init wizard and an in-app `homie settings` menu

---

## 2. Init Wizard — 12-Step Guided Flow

The current 7-step wizard becomes a 12-step guided flow. All configuration happens within `homie init`. No separate CLI commands exist for individual features.

| Step | What Happens | Can Skip? |
|------|-------------|-----------|
| 1. **Hardware Detection** | Auto-detect OS, CPU, RAM, GPU/VRAM, microphone, speakers | No |
| 2. **Microphone Permission** | Try Windows permission prompt → if fails, open Settings with guidance → retry detection | Yes |
| 3. **LLM Setup** | Local vs Qubrid cloud, model selection, path/API config | No |
| 4. **Voice Configuration** | Enable/disable, mode (push-to-talk/hybrid/conversational), hotkey, wake word | Yes (defaults to enabled if mic found) |
| 5. **User Profile** | Name, preferred language, timezone, work schedule/hours | Name required, rest skippable |
| 6. **Screen Reader Consent** | Explain 3 tiers, get permission level, set capture frequency | Yes (defaults to off) |
| 7. **Email Connection** | Walk through Gmail OAuth (client ID/secret → browser auth) | Yes |
| 8. **Social Connections** | Offer each platform one by one with guided walkthrough | Yes per platform |
| 9. **Privacy Preferences** | Which observers enabled, data retention days, max storage | Yes (sensible defaults) |
| 10. **Plugin Selection** | Checklist of available plugins: browser, IDE, git, terminal, music, health, notes, shortcuts | Yes (defaults to core set) |
| 11. **Summary** | Show everything configured, what was skipped, how to change later via `homie settings` | No |
| 12. **Save & Launch** | Write config, init storage, register model, choose service mode, optionally start daemon | No |

**Skip behavior:** Skipped steps use sensible defaults. The summary screen shows what was skipped with a note: "You can configure these anytime from within Homie."

---

## 3. Microphone Permission Gate (Step 2)

When init detects no microphone access, it runs a structured permission flow:

```
Detect mic via sounddevice.query_devices()
       |
   Found? --yes--> Proceed to voice config (Step 4)
       | no
Try request access via Windows permission prompt
  (launch a brief audio capture to trigger the OS dialog)
       |
   Granted? --yes--> Proceed to voice config
       | no
Open Windows Settings: Privacy > Microphone
  (via: os.startfile('ms-settings:privacy-microphone'))
Show guidance: "Please enable microphone access for Python/Homie, then press Enter"
       |
User presses Enter -> retry detection
       |
   Found? --yes--> Proceed to voice config
       | no
"No microphone detected. Voice will be disabled.
 You can enable it later in Homie settings."
       |
Skip to Step 3
```

**Implementation:**
- Trigger OS prompt: Call `sounddevice.InputStream()` briefly — Windows shows permission dialog on first mic access attempt
- Open Settings: `os.startfile('ms-settings:privacy-microphone')` — native Windows URI, no admin needed
- Max 3 retries with clear messaging between each
- Never blocks init — user can return via `homie settings`

**Voice defaults when mic granted:**

| Setting | Default | Reason |
|---------|---------|--------|
| `enabled` | `true` | User went through permission flow |
| `mode` | `push_to_talk` | Safest starting mode, no accidental captures |
| `hotkey` | `ctrl+8` | Existing default, changeable in Step 4 |
| `wake_word` | `hey homie` | Only active in hybrid/conversational mode |

---

## 4. Screen Reader System (Step 6)

A three-tier screen awareness engine that builds context without interfering with user workflow.

### 4.1 Tier Architecture

| Tier | What It Captures | How | Resource Cost | Default |
|------|-----------------|-----|--------------|---------|
| **T1: Window Tracker** | Active window title, process name, window switches | Win32 API (`GetForegroundWindow`, `GetWindowText`) | Negligible | On |
| **T2: OCR Reader** | Text content from active window region | `Windows.Media.Ocr` (built-in) or `Tesseract` fallback | Low-medium | Off |
| **T3: Visual Analyzer** | Screenshot → LLM description of what user is doing | Screenshot via `mss`/`Pillow`, analyzed by Qubrid cloud (local LLM if capable) | Medium-high | Off |

### 4.2 Capture Strategy (Hybrid)

**Passive polling** (context building):
- T1 runs every 5 seconds — window title/process, almost free
- T2/T3 run every 30-60 seconds (configurable) when enabled

**Event-driven** (proactive suggestions):
- T1 detects window focus change → triggers T2/T3 capture if enabled
- Significant context switch (e.g., IDE → browser → email) triggers Homie to offer help
- Debounced — won't fire if user is rapidly alt-tabbing

### 4.3 PII Filtering Pipeline

Every captured text/image passes through a filter before Homie processes it:

```
Raw capture --> PII Filter --> Sanitized context --> Homie brain
```

**Filter rules:**
- **Strip:** Email addresses, phone numbers, credit card patterns, SSN patterns (regex-based)
- **Redact:** Named entities that look like personal names (lightweight NER or pattern matching)
- **Ignore:** Password fields (detected by window class/title patterns like "Password", login pages)
- **Never capture:** Windows matching blocklist (banking apps, password managers, private browsing)

**Blocklist** ships with sensible defaults and is user-extensible via `homie settings`.

### 4.4 LLM Analysis (T3)

1. Resize screenshot to 720p max (reduce tokens/bandwidth)
2. Run PII filter on any OCR'd text in the image
3. Send to Qubrid cloud vision model with prompt: *"Describe what the user is doing at a high level. Do not extract specific text, names, or personal data."*
4. If user selected local + system is capable: use local multimodal model instead
5. Response stored as short context string (e.g., "User is editing a Python file in VS Code, working on a database migration")

### 4.5 Consent Model (Init Step 6)

```
Screen awareness helps me understand what you're working on so I can help proactively.

Choose your comfort level:
  [1] Window titles only - I see app names, nothing more
  [2] Window titles + text reading - I can read on-screen text (PII filtered)
  [3] Full visual awareness - I can see and understand your screen (PII filtered)
  [4] Off - I only know what you tell me

You can change this anytime. I never capture passwords, banking, or private browsing.
```

### 4.6 Integration with Homie Brain

Screen context feeds into the existing `context/` module alongside system monitoring (CPU/RAM/GPU). The brain uses it for:
- **Proactive suggestions:** "I see you're on GitHub reviewing a PR — want me to summarize the changes?"
- **Conversational context:** When user asks "help me with this," Homie already knows what "this" is
- **Behavioral learning:** Feeds into existing work/browsing observers for routine detection

---

## 5. Windows Service & Notification System

### 5.1 Launch Modes

| Mode | How It Runs | Best For |
|------|------------|----------|
| **On-demand** | User runs `homie start` manually, exits when terminal closes | Users who want full control |
| **Windows Service** | Runs via Windows Task Scheduler, starts on login, survives terminal close | Always-on assistant experience |

### 5.2 Windows Service Implementation

**Registration** (at Step 12, if user opts in):
1. Create a Windows Task Scheduler task (`HomieAI`) that runs `homie daemon --service` at user login
2. Run with least privileges (current user, no admin needed)
3. Restart on failure (max 3 retries, 60s delay)
4. Alternative: `pythonw.exe` based background process for simpler deployment

**Service lifecycle:**
- `homie start` — starts daemon (foreground or service depending on config)
- `homie stop` — gracefully stops the service
- `homie status` — shows running state, uptime, active components
- All also accessible from system tray (existing `pystray`)

### 5.3 Notification System

Windows Toast Notifications (via `plyer` or `win10toast`):

| Notification Type | Trigger | Example |
|---|---|---|
| **Task reminders** | Scheduled events, calendar items | "Meeting with team in 15 minutes" |
| **Email digest** | New important emails (if Gmail connected) | "3 unread emails from your manager" |
| **Social mentions** | @ mentions, DMs (if socials connected) | "New DM on Twitter from @user" |
| **Context suggestions** | Screen reader detects opportunity | "I see you're writing a README — want help?" |
| **System alerts** | Model download complete, connection issues | "Voice model loaded, ready to talk" |

### 5.4 Minimal Popup Widget

A lightweight floating widget (separate from the full overlay):
- Shows Homie's status icon in the corner (listening / thinking / idle)
- Expands on hover to show recent notifications
- Click to open full Homie interaction (chat or voice)
- Hotkey (`ctrl+8`) toggles it
- Built on existing `overlay.py` infrastructure, slimmed down

### 5.5 Privacy Controls

- Notifications never show full email/message content by default — just sender + count
- "Do Not Disturb" mode suppresses all popups (still collects, shows on next check)
- Notification categories individually toggleable in `homie settings`

---

## 6. Command Consolidation

### 6.1 Commands Removed

| Old Command | Where It Moves |
|---|---|
| `homie connect gmail` | Init Step 7 + `homie settings` |
| `homie connect twitter/reddit/...` | Init Step 8 + `homie settings` |
| `homie voice` | Init Step 4 + `homie settings` |
| `homie plugin enable/disable` | Init Step 10 + `homie settings` |

### 6.2 Commands Retained

| Command | Purpose |
|---|---|
| `homie init` | Full setup wizard (12 steps) |
| `homie start` | Launch Homie (foreground or service, based on config) |
| `homie stop` | Stop running service |
| `homie status` | Show running state and active components |
| `homie chat` | Direct text interaction |
| `homie settings` | In-app reconfiguration menu |

### 6.3 `homie settings` Design

Interactive menu mirroring init steps for post-setup changes:

```
Homie Settings
--------------
[1] LLM & Model
[2] Voice
[3] User Profile
[4] Screen Reader
[5] Email & Socials
[6] Privacy
[7] Plugins
[8] Notifications
[9] Service Mode
[0] Back
```

Selecting any option re-presents that section's wizard flow with current values shown. Changes saved immediately to `homie.config.yaml`.

### 6.4 System Tray Integration

Right-clicking the tray icon when running as service:

```
Homie AI
---------
Open Chat
Settings...
---------
Status: Running
Voice: Listening (push-to-talk)
Screen: Window titles + OCR
---------
Do Not Disturb
Pause Screen Reader
---------
Stop Homie
```

"Settings..." opens `homie settings` in a terminal. Quick toggles (DND, pause screen reader) work directly from tray.

---

## 7. Social OAuth Config Updates

### 7.1 Broken/Outdated Configs

| Provider | Issue | Fix |
|---|---|---|
| **LinkedIn** | `r_liteprofile`, `r_emailaddress` deprecated | Replace with `openid`, `profile`, `email`, `w_member_social` |
| **Instagram** | Old scope names, uses Facebook auth URL | New auth URL `instagram.com/oauth/authorize`, `instagram_business_*` prefixed scopes, PKCE required |
| **Facebook** | API version v19.0 outdated | Bump to v24.0 |
| **Twitter** | No PKCE support | Add Authorization Code with PKCE flow |

### 7.2 Configs Unchanged

- **Gmail** — correct as-is
- **Reddit** — correct, add `duration=permanent` for refresh tokens
- **Slack** — correct as-is

### 7.3 Init Step 8 — Guided Social Connection

Each platform gets a guided sub-flow:

```
"Would you like to connect [Platform]?" (y/n/skip all)
       |  yes
Show purpose: "Connecting Twitter lets me read your feed,
post on your behalf, and manage DMs."
       |
Show requirements & warnings (platform-specific)
       |
"Do you have a [Platform] developer app? (y/n)"
       |  no
Open developer console URL in browser automatically
Show step-by-step instructions inline
       |  yes
Prompt: "Paste your Client ID:"
Prompt: "Paste your Client Secret:"
       |
Store credentials -> open browser for OAuth consent -> capture token
       |
"Connected as @username"
```

### 7.4 Per-Platform Guidance

| Platform | Developer Console URL | Warnings |
|---|---|---|
| **Gmail** | console.cloud.google.com | "Requires enabling Gmail API. Restricted scopes need Google verification." |
| **Twitter** | developer.x.com | "Free tier: read + post only. DMs require Basic tier ($200/mo)." |
| **Reddit** | reddit.com/prefs/apps | "Create a 'script' type app. Free, no restrictions." |
| **LinkedIn** | linkedin.com/developers | "Tokens expire in 60 days — Homie will remind you to reconnect." |
| **Facebook** | developers.facebook.com | "Page management requires app review. Personal posting works immediately. API v24.0." |
| **Instagram** | developers.facebook.com | "Business/Creator account required. Personal accounts not supported." |
| **Slack** | api.slack.com/apps | "Create a bot app. Bot tokens don't expire." |
| **Blog/RSS** | N/A | "Just paste your RSS/Atom feed URL. No developer app needed." |

### 7.5 Platform Order (ease-first)

1. Blog/RSS (just a URL)
2. Reddit (free, simple)
3. Telegram (straightforward, officially supported)
4. Slack (straightforward)
5. Gmail (common, well-documented)
6. Phone Link SMS/RCS (auto-detected, read-only)
7. Twitter (free tier limitations)
8. Facebook (app review complexity)
9. Instagram (business account requirement)
10. LinkedIn (token expiration hassle)
11. WhatsApp (experimental, last due to risk)

---

## 8. New Messaging Platforms

### 8.1 Telegram

- **Auth:** User provides `api_id` + `api_hash` from my.telegram.org, authenticates with phone number + code
- **Library:** `telethon` (pure Python, async, MTProto)
- **Capabilities:** Full read + write — all conversations, contacts, media
- **Legal risk:** Low — custom clients officially supported by Telegram
- **Privacy:** Session runs locally, all data stays on-device

### 8.2 WhatsApp (Experimental)

- **Auth:** QR code scan via whatsapp-web.js bridge (Node subprocess)
- **Capabilities:** Full read + write on paired account
- **Legal risk:** High — unofficial protocol, Meta bans AI assistants, frequent account suspensions
- **Warning shown during init:**
  ```
  "WhatsApp connection is experimental. It uses an unofficial protocol
  that Meta may block. There is a risk of temporary account suspension.
  Connect anyway? (y/n)"
  ```
- **Session runs locally**, no third-party relay

### 8.3 Phone Link SMS/RCS (Read-Only)

- **Auth:** None — reads local Phone Link SQLite database
- **Location:** `%LOCALAPPDATA%\Packages\Microsoft.YourPhone_8wekyb3d8bbwe\LocalCache\Indexed\{GUID}\System\Database`
- **Capabilities:** Read-only access to synced SMS/MMS messages. RCS availability depends on phone manufacturer.
- **Auto-detection during init:**
  ```
  "I detected Windows Phone Link is paired with a device.
  Want me to read your synced text messages? (read-only,
  all data stays local) (y/n)"
  ```
- **Fragile:** Undocumented schema, treated as best-effort with graceful failure

### 8.4 RCS Direct

No consumer API exists. RCS messages are only accessible indirectly through Phone Link sync.

---

## 9. Config Schema Changes

### 9.1 New Config Sections

```yaml
# User Profile
user:
  name: "Master"
  language: en
  timezone: auto
  work_hours:
    start: "09:00"
    end: "18:00"
    days: [mon, tue, wed, thu, fri]

# Screen Reader
screen_reader:
  enabled: false
  level: 1                # 1=window titles, 2=+OCR, 3=+visual analysis
  poll_interval_t1: 5     # seconds
  poll_interval_t2: 30
  poll_interval_t3: 60
  event_driven: true
  analysis_engine: cloud  # cloud (Qubrid) or local
  pii_filter: true        # always on
  blocklist:
    - "*password*"
    - "*banking*"
    - "*incognito*"
    - "*private*"
    - "*1Password*"
    - "*KeePass*"
    - "*LastPass*"
  dnd: false

# Service Mode
service:
  mode: on_demand         # on_demand or windows_service
  start_on_login: false
  restart_on_failure: true
  max_retries: 3

# Notifications
notifications:
  enabled: true
  categories:
    task_reminders: true
    email_digest: true
    social_mentions: true
    context_suggestions: true
    system_alerts: true
  dnd_schedule:
    enabled: false
    start: "22:00"
    end: "07:00"

# Connections (state only, credentials stay in vault)
connections:
  gmail: { connected: false }
  twitter: { connected: false }
  reddit: { connected: false }
  telegram: { connected: false }
  slack: { connected: false }
  facebook: { connected: false }
  instagram: { connected: false }
  linkedin: { connected: false }
  whatsapp: { connected: false, experimental: true }
  phone_link: { connected: false, read_only: true }
  blog: { connected: false, feed_url: "" }
```

### 9.2 Modified Sections

```yaml
voice:
  enabled: true           # changed default from false to true
  mode: push_to_talk      # safe default for new installs

privacy:
  screen_reader_consent: false  # explicit consent flag
```

### 9.3 New Pydantic Models

- `UserProfileConfig` — name, language, timezone, work_hours
- `ScreenReaderConfig` — level, intervals, blocklist, engine
- `ServiceConfig` — mode, start_on_login, restart
- `NotificationConfig` — categories, DND schedule
- `ConnectionsConfig` — per-platform connection state

### 9.4 Vault (Unchanged)

Credentials (client ID, secret, access tokens, refresh tokens) remain in the encrypted vault. Config only tracks connection state — no secrets in `homie.config.yaml`.

---

## 10. Architecture — New & Modified Modules

### 10.1 New Modules

| Module | Location | Purpose |
|---|---|---|
| `screen_reader/` | `src/homie_core/screen_reader/` | 3-tier screen awareness engine |
| `screen_reader/window_tracker.py` | | T1: Win32 API window title/process polling |
| `screen_reader/ocr_reader.py` | | T2: Windows OCR / Tesseract text extraction |
| `screen_reader/visual_analyzer.py` | | T3: Screenshot + Qubrid/local LLM analysis |
| `screen_reader/pii_filter.py` | | Regex + pattern-based PII stripping |
| `screen_reader/capture_scheduler.py` | | Hybrid polling + event-driven orchestration |
| `notifications/` | `src/homie_core/notifications/` | Toast notifications + category routing |
| `notifications/toast.py` | | Windows Toast via `plyer` or `win10toast` |
| `notifications/router.py` | | Category filtering, DND logic |
| `service/` | `src/homie_app/service/` | Windows service registration + lifecycle |
| `service/scheduler_task.py` | | Task Scheduler create/remove/status |
| `service/tray_menu.py` | | Extended tray with quick toggles |
| `messaging/` | `src/homie_core/messaging/` | New messaging platform integrations |
| `messaging/telegram_provider.py` | | Telethon-based client API |
| `messaging/whatsapp_provider.py` | | whatsapp-web.js bridge (experimental) |
| `messaging/phone_link_reader.py` | | Phone Link SQLite DB reader |

### 10.2 Modified Modules

| Module | Changes |
|---|---|
| `src/homie_app/init.py` | 7-step → 12-step wizard, all connection logic moved in |
| `src/homie_app/cli.py` | Remove `homie connect`, `homie voice`. Add `homie settings`, `homie stop`, `homie status` |
| `src/homie_core/config.py` | Add new Pydantic models for all new config sections |
| `src/homie_core/social_media/oauth.py` | Update LinkedIn scopes, Instagram scopes/URL, Facebook API version, add PKCE for Twitter/Instagram |
| `src/homie_app/daemon.py` | Integrate screen reader, notifications, service mode, Phone Link reader |
| `src/homie_core/context/` | Receive screen reader context alongside existing system monitoring |
| `src/homie_core/brain/` | Use screen context for proactive suggestions |
| `pyproject.toml` | New dependencies |

### 10.3 Data Flow

```
Screen Reader --> PII Filter --> Context Module --> Brain
     ^                                               |
Window events                                  Proactive suggestions
                                                     |
                                              Notification Router
                                                     |
                                         Toast / Tray / Overlay
```

### 10.4 New Dependencies

| Package | Purpose | Required/Optional |
|---|---|---|
| `telethon` | Telegram client API | Optional (messaging extra) |
| `plyer` or `win10toast` | Windows toast notifications | Required for service mode |
| `mss` | Fast screenshot capture | Optional (screen reader T3) |
| `Pillow` | Image resize before LLM analysis | Optional (screen reader T3) |
| `pywin32` | Win32 API for window tracking | Optional (screen reader T1) |

### 10.5 Unchanged

- Voice pipeline
- Memory system
- Existing plugins (browser, IDE, git, etc.)
- Storage/vault encryption
- Brain/agentic loop architecture
- Website (heyhomie.app)

---

## 11. Testing Strategy

### 11.1 Unit Tests

| Component | What to Test |
|---|---|
| **Init wizard steps** | Each step runs independently, skip logic works, defaults applied correctly |
| **Mic permission flow** | Detection → prompt → settings → retry → fallback (mock `sounddevice` and `os.startfile`) |
| **Screen reader PII filter** | Email, phone, SSN, credit card patterns stripped. Blocklist matching. Named entity redaction |
| **Screen reader tiers** | T1 returns window title/process, T2 returns OCR text, T3 returns LLM description |
| **Config schema** | New Pydantic models validate correctly, defaults applied, backwards-compatible with old configs |
| **OAuth configs** | Updated URLs, scopes, and PKCE flow for each provider |
| **Notification system** | Toast creation, DND suppression, category filtering |
| **Phone Link reader** | SQLite parsing, graceful failure on missing/locked DB |
| **Settings menu** | Each section loads current values, saves changes correctly |

### 11.2 Integration Tests

| Flow | What to Test |
|---|---|
| **Full init wizard** | Steps 1-12 end-to-end with simulated user input |
| **OAuth per platform** | Token exchange, credential storage in vault, token refresh |
| **Telegram auth** | api_id/hash → phone code → session creation via Telethon |
| **WhatsApp bridge** | Node subprocess spawn, QR code flow, session persistence |
| **Screen reader pipeline** | Capture → PII filter → context storage → brain query |
| **Service lifecycle** | Register Task Scheduler → start → status → stop → unregister |
| **Window service notifications** | Toast fires, tray icon updates, DND respects schedule |

### 11.3 What to Mock

- `sounddevice` (mic detection)
- `os.startfile` (Windows Settings launch)
- `win32gui` (window titles)
- External OAuth endpoints (use recorded responses)
- Qubrid cloud API (for T3 analysis)
- Windows Toast API

### 11.4 What NOT to Mock

- SQLite/vault operations (test with real temp DB)
- Config file read/write (test with real temp files)
- Pydantic validation (test actual schema)
- PII regex patterns (test with real-looking data)

---

## 12. Web Scraping Decision

**Decision: OAuth only. No web scraping for social platforms.**

**Rationale:**
- Legal risk too high — LinkedIn sued and won, Meta actively litigates
- Stability terrible — every platform except Reddit breaks scrapers regularly
- Scraping requires user credentials anyway — worse UX than OAuth
- Privacy-first brand conflict — scraping others' platforms contradicts Homie's positioning
- Reddit is the only feasible target, but PRAW/OAuth works better anyway

**Exception considered:** RSS-Bridge as self-hosted read-only feed consumption — deferred as a future "nice to have."
