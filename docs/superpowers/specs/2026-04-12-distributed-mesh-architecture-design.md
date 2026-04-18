# Distributed Mesh Architecture — Hub-and-Spoke with Smart Fallback

**Date:** 2026-04-12
**Status:** Approved
**Scope:** Cross-machine distributed intelligence, unified user model, event-sourced sync

---

## 1. Problem Statement

Homie currently runs as a single-machine assistant. To be a true companion, it must:
- Run on any machine (Windows, Linux, macOS, Android, headless servers)
- Form a mesh of interconnected instances that share intelligence
- Build a complete understanding of the user across all their devices
- Route inference to the best available hardware
- Work offline and sync seamlessly on reconnect
- Self-learn from interactions across the entire mesh

## 2. Architecture: Hub-and-Spoke with Smart Fallback

**Core principle:** Every Homie node is a fully functional standalone instance. Hub/Spoke is an optimization layer, not a dependency.

- **Hub:** The most capable machine (typically desktop with GPU). Handles shared inference, memory aggregation, fine-tuning, and context synthesis.
- **Spoke:** Any other machine. Handles local awareness, voice capture, local caching, and offline operation.
- **Standalone:** A node with no mesh connection. Full functionality, no sync.

Any Spoke can be promoted to Hub. Hub election is automatic based on capability score.

## 3. Node Identity & Topology

### 3.1 Node Identity

Every machine gets a `HomieNode` identity on first run, persisted in `~/.homie/node.json`:

```
HomieNode:
  node_id: UUID (generated once, never changes)
  node_name: str (defaults to hostname, user-configurable)
  role: hub | spoke | standalone
  created_at: ISO8601 UTC
  mesh_id: UUID | null (shared across all nodes in a mesh)
```

### 3.2 Capability Profile

Collected on startup and refreshed periodically:

```
NodeCapabilities:
  gpu: {name: str, vram_gb: float} | null
  cpu_cores: int
  ram_gb: float
  disk_free_gb: float
  os: windows | linux | macos | android
  has_mic: bool
  has_display: bool
  has_model_loaded: bool
  model_name: str | null
```

### 3.3 Hub Election

Automatic, deterministic, no manual config required:

1. On first boot, node starts as **standalone**
2. When discovering other Homie nodes via mDNS, enters **mesh negotiation**
3. Capability score: `(gpu_vram * 10) + (ram_gb * 2) + (cpu_cores) + (has_model_loaded * 50)`
4. Highest score becomes Hub. Ties broken by earliest `created_at`.
5. If Hub goes offline for >60 seconds, next-highest Spoke auto-promotes
6. When original Hub returns, it becomes Spoke (no disruptive re-election)
7. Manual override: `homie node set-role hub` (force this node as Hub)

### 3.4 Topology Data Model

```sql
-- ~/.homie/homie.db (extends existing schema)
CREATE TABLE mesh_nodes (
    node_id TEXT PRIMARY KEY,
    node_name TEXT NOT NULL,
    role TEXT NOT NULL DEFAULT 'standalone',
    mesh_id TEXT,
    capability_score REAL DEFAULT 0,
    capabilities_json TEXT,
    lan_ip TEXT,
    tailnet_ip TEXT,
    last_seen_ts TEXT,
    paired_at TEXT,
    public_key_ed25519 TEXT,
    status TEXT DEFAULT 'offline'  -- online | offline | degraded
);
```

## 4. Network & Discovery Layer

### 4.1 LAN Discovery (Zero-Config)

Builds on existing `src/homie_core/network/discovery.py`:

- Protocol: mDNS/DNS-SD via zeroconf
- Service type: `_homie._tcp.local.`
- TXT records: `node_id`, `role`, `capability_score`, `protocol_version`, `mesh_id`
- Discovery latency: <5 seconds on same LAN
- Continuous browsing: detect nodes joining/leaving in real-time

### 4.2 WAN Connectivity

For nodes on different networks (home <-> office):

- **Primary:** Tailscale (existing architecture) — WireGuard-based, NAT-traversing
- **Secondary:** Direct WebSocket over HTTPS with mutual TLS
- **Config:** Spoke stores Hub address in `~/.homie/mesh.json`:
  ```json
  {
    "hub_address": "100.64.0.1:8721",
    "transport": "tailscale",
    "fallback_address": "home.example.com:8721"
  }
  ```

### 4.3 Pairing Protocol

Secure one-time pairing between nodes:

1. Hub: `homie mesh pair` → generates 6-digit code (valid 5 minutes, single-use)
2. Spoke: `homie mesh join --code 483291`
3. Protocol:
   a. Spoke connects to Hub's pairing endpoint with the code
   b. Hub validates code, both exchange Ed25519 public keys
   c. Shared secret derived via X25519 ECDH key agreement
   d. Secret stored in existing vault (AES-256-GCM encrypted)
   e. Both nodes record each other in `mesh_nodes` table
   f. Hub assigns `mesh_id` to Spoke (or creates new mesh)
4. Post-pairing: HMAC-SHA256 authenticates all messages (existing `node_api.py` pattern)

### 4.4 Transport Layer

Two channels per connection:

| Channel | Protocol | Use Case |
|---------|----------|----------|
| REST | FastAPI HTTPS | Request/response: status, task dispatch, queries |
| Stream | WebSocket (WSS) | Real-time: context events, inference tokens, voice relay |

- Payload format: MessagePack (binary, compact) with JSON fallback for debugging
- Compression: LZ4 for payloads >1KB (memory sync, file transfer)
- Keep-alive: WebSocket ping every 30s, reconnect with exponential backoff (1s, 2s, 4s, 8s, max 60s)

### 4.5 Heartbeat & Health

- Hub polls each Spoke: `GET /v1/health` every 15 seconds (jitter: +/- 3s)
- Spoke pushes status on significant events (activity change, error, low battery)
- Status transitions: `online` → `degraded` (2 missed heartbeats) → `offline` (4 missed)
- On status change: Hub emits `node_status_changed` event to all other Spokes

## 5. Event-Sourced Sync Protocol

### 5.1 Event Schema

Every meaningful action produces an immutable event:

```python
@dataclass
class HomieEvent:
    event_id: str          # ULID (time-sortable, globally unique)
    node_id: str           # Source node UUID
    timestamp: str         # ISO8601 UTC
    category: str          # memory | context | task | preference | learning | system
    event_type: str        # e.g., "fact_learned", "episode_recorded", "activity_changed"
    payload: dict          # Event-specific data
    vector_clock: dict     # {node_id: counter} for causal ordering
    checksum: str          # SHA-256 of payload for integrity
```

### 5.2 Event Categories

| Category | Event Types | Sync Direction |
|----------|-------------|----------------|
| memory | fact_learned, fact_corrected, episode_recorded, memory_consolidated | Spoke ↔ Hub |
| context | activity_changed, session_started, session_ended, flow_state_changed | Spoke → Hub |
| task | task_created, task_updated, task_completed, task_delegated | Spoke ↔ Hub |
| preference | preference_set, feedback_signal, response_rated | Spoke → Hub |
| learning | model_updated, training_started, training_completed | Hub → Spokes |
| system | node_joined, node_left, hub_elected, error_occurred | Hub ↔ Spokes |

### 5.3 Sync Flow

```
Spoke connects to Hub:
  1. Spoke sends: {last_seen_event_id, node_id, vector_clock}
  2. Hub responds: all events after last_seen_event_id (delta sync)
  3. Spoke applies events to local state
  4. Spoke sends: its own events since last sync
  5. Hub merges Spoke events, resolves conflicts, broadcasts to other Spokes
```

### 5.4 Conflict Resolution

Per-category merge strategies:

| Category | Strategy | Rationale |
|----------|----------|-----------|
| Semantic memory (facts) | LWW with Lamport timestamps | Facts update; latest version wins |
| Episodic memory | Append-only, deduplicate by content_hash | Experiences are immutable |
| Preferences | LWW — most recent explicit signal wins | User intent is latest action |
| Context | No merge — node-specific | Each node tracks its own context |
| Tasks | State machine merge (highest state wins) | Tasks progress forward only |

### 5.5 Event Storage

```sql
CREATE TABLE event_log (
    event_id TEXT PRIMARY KEY,
    node_id TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    category TEXT NOT NULL,
    event_type TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    vector_clock_json TEXT NOT NULL,
    checksum TEXT NOT NULL,
    synced_to_hub BOOLEAN DEFAULT FALSE,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_event_log_category ON event_log(category, timestamp);
CREATE INDEX idx_event_log_sync ON event_log(synced_to_hub, timestamp);
```

### 5.6 Offline Resilience

- Spoke accumulates events locally when disconnected (unlimited duration)
- On reconnect: full delta sync with vector clock comparison
- Event log compaction: weekly job merges old events into summary snapshots
- Compaction preserves: facts, preferences, task outcomes. Drops: granular context events older than 7 days

## 6. Distributed Context & Unified User Model

### 6.1 Per-Node Context (Collected Locally)

Each node's awareness engine produces a `NodeContext` (extends existing awareness system):

```python
@dataclass
class NodeContext:
    node_id: str
    last_updated: str
    
    # Current activity (from existing awareness engine)
    active_window: str
    active_process: str
    activity_type: str       # coding | browsing | email | meeting | idle
    activity_confidence: float
    
    # Session state
    session_start: str
    minutes_active: float
    idle_minutes: float
    flow_score: float
    
    # Machine state
    cpu_usage: float
    gpu_usage: float | None
    ram_usage_gb: float
    battery_pct: float | None  # laptops/phones only
    
    # Activity timeline (last 30 minutes, 5-minute buckets)
    activity_timeline: list[ActivityBucket]
```

Context is synced to Hub as `context.activity_changed` events (throttled: max 1 per 30 seconds per node, or on significant change).

### 6.2 Unified User Model (Hub Aggregates)

The Hub builds a `UnifiedUserModel` from all node contexts:

```python
@dataclass
class UnifiedUserModel:
    user_id: str
    
    # Cross-device real-time state
    active_nodes: list[str]       # Currently active node names
    primary_node: str             # Where user is most active right now
    current_activity_summary: str # "Coding on desktop, laptop idle"
    overall_flow_state: str       # deep_work | focused | casual | idle
    energy_level: str             # high | medium | low (from rhythm + activity)
    
    # Cross-device task continuity
    active_tasks: list[CrossDeviceTask]
    
    # Behavioral patterns (learned over weeks)
    daily_rhythm: dict[str, str]  # hour_range -> typical_activity
    device_usage_patterns: dict[str, DevicePattern]  # node_name -> pattern
    
    # Per-device preferences
    device_preferences: dict[str, dict]  # node_name -> {response_style, voice_enabled, ...}
    
    # Global preferences (applies everywhere)
    global_preferences: dict      # {tone, verbosity, proactivity_level, ...}
```

### 6.3 Context Handoff

When user switches devices:

1. Destination node detects user activity (keyboard/mouse input after idle)
2. Queries Hub for source node's recent context
3. Brain's PERCEIVE stage now includes cross-device context:
   ```
   [CROSS-DEVICE CONTEXT]
   Previously: coding in VS Code on desktop (orchestrator.py, 47 min session)
   Switched to: laptop (just became active)
   Recent topics: sync protocol, event sourcing, CRDT merge strategies
   Open tasks: "implement sync protocol" (started on desktop)
   ```
4. If conversation was in progress on source node, offer to continue it
5. Working memory is portable — synced via events

### 6.4 Proactive Cross-Device Intelligence

The unified user model enables proactive suggestions impossible on a single machine:

- "You've been at your desk for 3 hours straight. Your laptop is in the living room — want me to switch there for a break?"
- "Your desktop is rendering a video (GPU at 95%). I'll route your next query through cloud inference."
- "You started a code review on your laptop but your desktop has the full repo. Want me to open it there?"
- "All your machines show idle — it's 11 PM. Generating your daily summary."

## 7. Distributed Inference

### 7.1 Enhanced Inference Router

Extends existing `InferenceRouter` with mesh awareness:

```python
class MeshInferenceRouter(InferenceRouter):
    """Routes inference to the best available compute in the mesh."""
    
    priority_chain = [
        "local",       # Model loaded on this node
        "hub",         # Hub's local model (WebSocket streaming)
        "lan_peer",    # Any node with GPU + loaded model
        "qubrid",      # Qubrid cloud (existing)
        "vertex",      # Vertex AI (existing)
    ]
```

### 7.2 Remote Inference Protocol

When Spoke sends inference to Hub:

1. Spoke builds prompt locally (full cognitive architecture pipeline)
2. Sends to Hub via WebSocket: `{type: "inference_request", prompt, max_tokens, temperature}`
3. Hub queues by priority (IMMEDIATE > BACKGROUND > BATCH)
4. Hub generates using its local model engine
5. Tokens stream back via WebSocket in real-time
6. Spoke receives and presents to user (same UX as local inference)

Latency overhead: ~5-15ms on LAN, ~30-80ms on Tailscale WAN. Acceptable for text; voice pipeline adds buffer.

### 7.3 Inference Queue (Hub)

```python
class InferenceQueue:
    """Priority queue for multi-Spoke inference requests."""
    
    priorities = {
        "IMMEDIATE": 0,   # User waiting (chat, voice)
        "BACKGROUND": 1,  # Proactive tasks, summarization
        "BATCH": 2,       # Synthetic data, bulk processing
    }
    
    max_concurrent: int  # Based on GPU VRAM: 1 for 16GB models, 2 for 8GB
    spillover: str       # "qubrid" | "vertex" | "reject"
```

### 7.4 Model Distribution

- Hub maintains the mesh model registry
- Spokes can request models: `homie model pull <name>` (downloads from Hub over LAN, fast)
- Fine-tuned models auto-deploy: Hub → training complete → notify Spokes → Spokes pull
- Model metadata synced via `learning.model_updated` events

## 8. Distributed Task Execution

### 8.1 Cross-Machine Task Types

| Task Type | Example | Flow |
|-----------|---------|------|
| Remote command | "Organize downloads on laptop" | Hub dispatches to laptop Spoke |
| Coordinated workflow | "Back up projects on all machines" | Hub orchestrates parallel execution |
| Inference delegation | "Summarize this doc" (from phone) | Phone → Hub → result back to phone |
| Awareness query | "What was I working on today?" | Hub queries all nodes' timelines |
| File transfer | "Send that file to my laptop" | Source Spoke → Hub relay → target Spoke |

### 8.2 Task Lifecycle

```
TaskRequest (from any node)
  → Hub validates (auth, safety, target online)
  → Hub plans execution (which node(s), what order)
  → Target Spoke(s) execute via existing executor.py
  → Results stream back to Hub
  → Hub aggregates and presents to requesting node
```

### 8.3 Safety

- All existing safety gates apply to cross-machine commands
- Cross-machine commands require explicit user confirmation unless in allowlist
- Allowlist configurable per node: `~/.homie/mesh_permissions.json`
- Destructive operations (delete, format, kill) always require confirmation regardless of allowlist
- Audit log: all cross-machine actions recorded with source_node, target_node, command, result

## 9. Security & Privacy

### 9.1 Authentication

| Layer | Mechanism |
|-------|-----------|
| Node-to-node | HMAC-SHA256 with shared secret (existing) |
| Pairing | One-time code + Ed25519/X25519 key exchange |
| Key rotation | Automatic every 30 days, manual via `homie mesh rotate-keys` |
| API access | Bearer token for external integrations (Dashboard API) |

### 9.2 Encryption

| Layer | Mechanism |
|-------|-----------|
| In-transit (LAN) | TLS 1.3 (FastAPI + WebSocket) |
| In-transit (WAN) | Tailscale WireGuard (automatic) or mutual TLS |
| At-rest (credentials) | AES-256-GCM vault (existing) |
| At-rest (event log) | SQLCipher or application-level AES |
| Sync payloads | Encrypted with mesh shared key before transit |

### 9.3 Privacy Controls

```yaml
# homie.config.yaml additions
privacy:
  sync:
    share_activity: true        # Share active window/process with Hub
    share_clipboard: false      # Never sync clipboard content
    share_screenshots: false    # Never sync screenshots
    share_file_contents: false  # Only share file metadata
    private_mode: false         # Temporarily disable all sync
    
  retention:
    context_events_days: 7      # Granular context events
    memory_events_days: 365     # Facts, episodes
    task_events_days: 90        # Task history
    event_log_max_mb: 500       # Max event log size per node
```

### 9.4 Enterprise Security

- RBAC: Admin (full control), User (use + configure own node), Viewer (read-only dashboard)
- Audit log: WHO did WHAT on WHICH machine at WHEN
- SSO: OIDC/SAML integration endpoint for enterprise identity providers
- Network policy: Restrict pairing to specific IP ranges or Tailscale ACL tags
- Compliance: Data residency controls — specify which nodes can store what data

## 10. Cross-Platform Packaging & Deployment

### 10.1 Platform Matrix

| Platform | Package | Inference | Voice | UI | Min Resources |
|----------|---------|-----------|-------|----|---------------|
| Windows 10+ | MSI + Windows Service | GGUF (CUDA/Vulkan) | Full | Tray + Tauri | 4GB RAM |
| Linux (Ubuntu 22+, Fedora 38+) | pip + systemd | GGUF (CUDA/ROCm/CPU) | Full (PipeWire/Pulse) | Tray + Tauri | 4GB RAM |
| macOS 13+ | pip + launchd | GGUF (Metal) | Full (CoreAudio) | Tray + Tauri | 4GB RAM |
| Android 12+ | Kotlin companion app | Spoke-only (remote) | Android STT/TTS APIs | Native Material 3 | 2GB RAM |
| Headless / Server | pip + systemd | GGUF or cloud-only | None | REST API only | 1GB RAM |
| Raspberry Pi | pip + systemd | Spoke-only (remote) | Optional USB mic | REST API | 512MB RAM |

### 10.2 Platform Abstraction Layer

Fills the currently empty `src/homie_core/platform/` directory:

```python
# src/homie_core/platform/base.py
class PlatformAdapter(ABC):
    @abstractmethod
    def get_hostname(self) -> str: ...
    
    @abstractmethod
    def get_active_window(self) -> str | None: ...
    
    @abstractmethod
    def get_system_metrics(self) -> SystemMetrics: ...
    
    @abstractmethod
    def get_audio_devices(self) -> list[AudioDevice]: ...
    
    @abstractmethod
    def send_notification(self, title: str, body: str) -> None: ...
    
    @abstractmethod
    def register_hotkey(self, combo: str, callback: Callable) -> None: ...
    
    @abstractmethod
    def get_service_manager(self) -> ServiceManager: ...
    
    @abstractmethod
    def get_credential_store(self) -> CredentialStore: ...
    
    @abstractmethod
    def get_gpu_info(self) -> GPUInfo | None: ...

# Implementations
# src/homie_core/platform/windows.py  -> WindowsAdapter
# src/homie_core/platform/linux.py    -> LinuxAdapter
# src/homie_core/platform/macos.py    -> MacOSAdapter
# src/homie_core/platform/android.py  -> AndroidAdapter (subset)
```

### 10.3 First-Run Setup (`homie init`)

Unified setup wizard that works on any platform:

1. **Detect** OS, hardware, GPU, audio devices
2. **Generate** node identity (UUID, Ed25519 keypair)
3. **Scan** for existing Homie nodes on LAN via mDNS (5-second scan)
4. **If nodes found:** Offer to join mesh → pairing code flow
5. **If no nodes found:** Start as standalone → auto-promotes to Hub when another joins
6. **Model setup:**
   - If GPU with >8GB VRAM: recommend downloading a local model
   - If GPU with 4-8GB VRAM: recommend smaller quantized model
   - If no GPU: recommend Spoke-only mode (remote inference from Hub)
   - Always: configure cloud fallback (Qubrid API key)
7. **Voice setup:** Test mic, select TTS engine based on OS/hardware
8. **Privacy defaults:** Show privacy config, let user adjust
9. **Done** — Homie starts in appropriate mode

### 10.4 Minimal Spoke Mode

For low-power devices (old laptops, Raspberry Pi, phones):

- No local model needed — all inference via Hub
- Runs only: awareness observer, voice capture, context sync, event log
- Memory footprint: <100MB RAM, <50MB disk
- CPU usage: <2% idle, <10% active
- Install: `pip install homie[spoke]` (minimal dependencies)

## 11. Self-Learning Across the Mesh

### 11.1 Distributed Feedback Collection

Every Spoke collects implicit learning signals:

```python
class FeedbackSignal:
    signal_type: str      # accepted | regenerated | corrected | ignored | rated
    query: str            # What the user asked
    response_summary: str # First 200 chars of response
    node_id: str          # Which device
    activity_context: str # What user was doing
    timestamp: str
```

Signals sync to Hub as `preference.feedback_signal` events.

### 11.2 Automated Fine-Tuning Trigger

Hub monitors accumulated feedback and triggers training cycles:

```
Trigger conditions (any one):
  - 500+ new feedback signals since last training
  - 100+ explicit corrections
  - 30+ days since last training cycle
  - User runs `homie model train` manually

Pipeline (runs on Hub, requires GPU):
  1. Filter high-quality training pairs from feedback
  2. Generate synthetic data via existing generator.py
  3. QLoRA SFT + DPO training via existing pipeline.py
  4. Evaluate via existing benchmark suite
  5. If score improved: deploy as new active model
  6. Emit learning.model_updated event
  7. All Spokes with local models pull updated weights
```

### 11.3 Cross-Device Knowledge Evolution

- Facts learned on ANY device propagate to ALL devices via sync
- Behavioral patterns from all devices build a richer user model
- Device-specific preferences stay device-specific (you talk differently to phone vs desktop)
- Global preferences merge across all devices (tone, topic interests, work habits)

## 12. CLI & API Additions

### 12.1 New CLI Commands

```
homie mesh status         # Show mesh topology, node statuses
homie mesh pair           # Generate pairing code (Hub only)
homie mesh join --code X  # Join mesh with pairing code
homie mesh leave          # Leave mesh, become standalone
homie mesh nodes          # List all nodes with capabilities
homie mesh promote <node> # Force a node to become Hub
homie mesh rotate-keys    # Rotate all mesh authentication keys

homie node info           # Show this node's identity + capabilities
homie node set-role <role> # Force role: hub | spoke
homie node set-name <name> # Set friendly name

homie context              # Show unified user model (cross-device)
homie context <node>       # Show specific node's context

homie task run --on <node> <command>  # Run task on specific node
homie task run --all <command>        # Run task on all nodes
```

### 12.2 New FastAPI Endpoints (Node Agent)

```
GET  /v1/mesh/status        # Mesh topology from this node's view
POST /v1/mesh/pair           # Generate pairing code
POST /v1/mesh/join           # Join with pairing code
POST /v1/mesh/leave          # Leave mesh
GET  /v1/mesh/nodes          # List known nodes
GET  /v1/context             # This node's current context
GET  /v1/context/unified     # Unified user model (Hub only)
POST /v1/inference           # Remote inference request
WS   /v1/stream              # WebSocket for real-time events + inference streaming
POST /v1/events/sync         # Push/pull events for sync
GET  /v1/events/since/<id>   # Get events after given ID
```

## 13. Configuration Additions

```yaml
# homie.config.yaml additions
mesh:
  enabled: true
  auto_discover: true           # mDNS discovery on LAN
  auto_elect_hub: true          # Automatic Hub election
  preferred_role: auto          # auto | hub | spoke
  pairing_timeout: 300          # Seconds before pairing code expires
  heartbeat_interval: 15        # Seconds between health checks
  sync_interval: 30             # Seconds between event sync cycles
  max_offline_events: 100000    # Max events to buffer while offline
  
  wan:
    enabled: false
    transport: tailscale        # tailscale | websocket
    hub_address: ""             # For websocket transport
    
  inference:
    allow_remote: true          # Allow other nodes to use this node for inference
    max_concurrent: 2           # Max simultaneous remote inference requests
    queue_spillover: qubrid     # qubrid | vertex | reject
    
  security:
    key_rotation_days: 30
    require_tailscale: false    # If true, only Tailscale IPs allowed
    ip_allowlist: []            # Additional allowed IP ranges
```

## 14. Implementation Order

### Phase 1: Foundation (Node Identity + Platform Layer)
- [ ] `HomieNode` identity generation and persistence (`node.json`)
- [ ] Platform abstraction layer (`platform/base.py` + `windows.py` + `linux.py` + `macos.py`)
- [ ] Capability detection (GPU, audio, OS features)
- [ ] `homie node info` command
- [ ] `homie init` enhanced setup wizard

### Phase 2: Network & Discovery
- [ ] Enhance existing mDNS discovery with capability TXT records
- [ ] Pairing protocol (Ed25519 key exchange, shared secret)
- [ ] WebSocket transport layer (real-time streaming)
- [ ] `homie mesh pair` / `homie mesh join` commands
- [ ] Heartbeat and health monitoring

### Phase 3: Event Sync
- [ ] Event schema and local event log (SQLite)
- [ ] Vector clock implementation
- [ ] Delta sync protocol (push/pull)
- [ ] Conflict resolution per category
- [ ] Offline event accumulation and replay
- [ ] Event log compaction

### Phase 4: Distributed Inference
- [ ] `MeshInferenceRouter` extending existing router
- [ ] Remote inference via WebSocket streaming
- [ ] Inference queue with priority levels
- [ ] Model distribution (Hub → Spoke over LAN)
- [ ] Automatic cloud spillover

### Phase 5: Unified Context
- [ ] `NodeContext` collection and sync
- [ ] `UnifiedUserModel` aggregation on Hub
- [ ] Context handoff on device switch
- [ ] Cross-device awareness in Brain's PERCEIVE stage
- [ ] Proactive cross-device intelligence

### Phase 6: Distributed Tasks
- [ ] Cross-machine task dispatch
- [ ] Coordinated multi-node workflows
- [ ] Safety gates for remote execution
- [ ] Audit logging for cross-machine actions

### Phase 7: Self-Learning Loop
- [ ] Distributed feedback signal collection
- [ ] Automated fine-tuning trigger on Hub
- [ ] Model update propagation to Spokes
- [ ] Cross-device knowledge evolution

### Phase 8: Enterprise & Hardening
- [ ] RBAC (Admin/User/Viewer)
- [ ] SSO integration (OIDC)
- [ ] Admin dashboard
- [ ] Compliance controls (data residency, retention policies)
- [ ] Android companion app (Spoke-only)

## 15. Testing Strategy

### Unit Tests
- Node identity generation and persistence
- Capability detection on each platform
- Event serialization/deserialization
- Vector clock operations
- Conflict resolution for each category
- Inference queue priority ordering

### Integration Tests
- Two-node discovery and pairing (localhost simulation)
- Event sync with simulated offline/reconnect
- Remote inference request/response
- Context handoff between nodes
- Cross-machine task execution

### End-to-End Tests
- Full mesh setup: Hub + 2 Spokes on same machine (different ports)
- User workflow: start task on node A, continue on node B
- Hub failover: kill Hub, verify Spoke promotion
- Offline resilience: disconnect Spoke, accumulate events, reconnect, verify sync

## 16. Dependencies (New)

```
# Core (required)
ulid-py >= 1.1       # Time-sortable unique IDs for events

# Network (mesh feature)
zeroconf >= 0.80     # mDNS discovery (already optional dep)
websockets >= 12.0   # WebSocket transport
msgpack >= 1.0       # Binary serialization
lz4 >= 4.3           # Payload compression
cryptography >= 41   # Ed25519/X25519 key exchange (already a dep)

# Optional
sqlcipher3 >= 0.5    # Encrypted SQLite for event log
```
