# Self-Healing Runtime — Design Spec

**Date:** 2026-03-21
**Status:** Approved
**Sub-project:** 1 of 5 (Self-Healing Runtime → Adaptive Learning → Knowledge Evolution → Performance Optimizer → Model Fine-Tuning)

---

## Vision

Make Homie a self-evolving intelligent assistant that detects failures, heals itself, optimizes its own performance, and evolves its own code and architecture — all autonomously, silently, and without user intervention.

## Design Decisions

- **Tiered autonomy**: All internal fixes are fully autonomous. No user approval needed.
- **Silent operation**: All events logged to internal health log only. No notifications.
- **YOLO self-modification**: Changes apply directly to live system. Rollback is the safety net.
- **Evolving playbook**: Recovery strategies are learned and refined over time, not static.
- **Core lock**: Safety mechanisms (rollback, guardian, security) are immutable.

---

## 1. System Architecture Overview

The Self-Healing Runtime has three layers:

### Layer 1: Module-Level Resilience

A `@resilient` decorator applied to boundary operations across existing modules. Provides:

- **Automatic retries** with exponential backoff (configurable per module)
- **Circuit breaker** — after N failures in a time window, trips open and uses fallback
- **Timeout enforcement** — no operation hangs indefinitely
- **Health reporting** — each decorated method reports state to the HealthWatchdog via the event bus

### Layer 2: HealthWatchdog Service

A background service that starts first on boot, shuts down last:

- **Health probes** — periodic checks on each subsystem
- **Anomaly detection** — tracks metrics, flags drift using moving averages + standard deviation
- **Recovery orchestrator** — routes failures to the appropriate recovery tier
- **Improvement engine** — identifies suboptimality and applies fixes
- **Health log** — structured SQLite table of all events

### Layer 3: Recovery + Improvement Engines

Two peer systems under the watchdog:

- **Recovery Engine** — fixes failures (reactive)
- **Improvement Engine** — fixes suboptimality (proactive)

---

## 2. Module-Level Resilience — The `@resilient` Decorator

### Decorator Interface

```python
@resilient(
    retries=3,
    backoff="exponential",        # 1s, 2s, 4s
    circuit_breaker_threshold=5,
    circuit_breaker_window=60,    # seconds
    timeout=30,
    fallback=None,                # optional fallback callable
    health_report=True            # report status to watchdog
)
```

### Exception Classification

| Category       | Examples                              | Action                              |
|----------------|---------------------------------------|-------------------------------------|
| **Transient**  | Timeout, OOM, connection reset        | Retry                               |
| **Recoverable**| File locked, DB busy                  | Retry with longer backoff           |
| **Permanent**  | FileNotFound, InvalidConfig           | Fail immediately, report to watchdog|
| **Fatal**      | Disk full, GPU driver crash           | Trip circuit, escalate to T3/T4     |

### Application Scope

Applied to boundary operations only:

- Inference calls (model loading, generation)
- Storage operations (SQLite, ChromaDB reads/writes)
- Voice pipeline (STT, TTS, wake word)
- Context observers (screen reader, process monitor)
- Network operations (LAN discovery, Qubrid cloud)
- Knowledge ingestion (document parsing, embedding)

Pure logic functions do not get the decorator.

---

## 3. HealthWatchdog Service

### Lifecycle

Starts as the first service on boot, before any other module. Protected by an OS-level guardian process that restarts it if it crashes.

### Boot Sequence

```
1. Start Watchdog + OS-level guardian process
2. Run all health probes (initial system assessment)
3. If critical failures → attempt T1-T3 recovery before proceeding
4. If unrecoverable → boot in degraded mode, log what's missing
5. Start Homie's main loop
6. Begin continuous monitoring cycle
```

Homie never fails to start — it always comes up, even if degraded.

### Health Probes

Each subsystem registers a probe returning:

```python
class HealthStatus:
    state: Literal["healthy", "degraded", "failed", "unknown"]
    latency_ms: float
    error_count: int
    last_error: Optional[str]
    metadata: dict
```

Registered probes:

| Probe             | Checks                                                         |
|-------------------|----------------------------------------------------------------|
| `inference`       | Model generates response, latency within bounds, GPU memory OK |
| `storage_sqlite`  | Read/write works, integrity check, file size within limits     |
| `storage_chromadb`| Collection accessible, embedding queries return results        |
| `voice_stt`       | Whisper model loaded, can transcribe test clip                 |
| `voice_tts`       | TTS engine responsive, audio output working                   |
| `memory`          | Vector store queryable, embedding model loaded                 |
| `context`         | Observers running, system metrics flowing                      |
| `network`         | LAN discovery alive, WebSocket connections stable              |
| `knowledge`       | Document index intact, session persistence working             |
| `config`          | Config parseable, required fields present, values valid        |

Default probe interval: 30s. Critical systems (inference, storage): 10s.

### Anomaly Detector

Tracks time-series metrics per module using moving averages + standard deviation. Flags:

- Response latency trending upward
- Memory usage growing unbounded (leak detection)
- Error rate spikes
- GPU utilization dropping unexpectedly

Simple statistics, no ML. The LLM is involved only at the diagnosis stage in the Improvement Engine.

### Event Bus

Lightweight in-process pub/sub (Python queues):

- Modules publish: health updates, errors, performance metrics
- Watchdog subscribes: aggregates, detects patterns, triggers recovery/improvement
- Zero network overhead

### Health Log

SQLite table `health_events`:

| Column       | Type     | Purpose                                          |
|--------------|----------|--------------------------------------------------|
| `timestamp`  | datetime | When it happened                                 |
| `module`     | string   | Which subsystem                                  |
| `event_type` | string   | probe_result, anomaly, recovery, improvement, rollback |
| `severity`   | string   | info, warning, error, critical                   |
| `details`    | JSON     | Full context — metrics, actions taken, outcomes  |
| `version_id` | string   | Links to self-modification version if applicable |

Retention: 30 days. Older entries summarized into weekly digests before deletion.

---

## 4. Recovery Engine — Tiered Strategies

### Tier Definitions

| Tier            | Trigger                   | Action                      | Example                                    |
|-----------------|---------------------------|-----------------------------|--------------------------------------------|
| **T1 - Retry**  | Transient failure         | Retry with backoff          | Inference timeout → retry                  |
| **T2 - Fallback** | Repeated failure       | Switch to alternative       | GPU OOM → reduce layers → CPU fallback     |
| **T3 - Rebuild** | Corruption/persistent   | Reconstruct from scratch    | ChromaDB corrupt → rebuild from SQLite     |
| **T4 - Degrade** | Unrecoverable           | Disable module, continue    | Voice crashes → text-only mode             |

Recovery always moves down tiers — T1 first, escalate only on failure.

### Inference Recovery

| Failure              | T1                          | T2                              | T3                            | T4                                  |
|----------------------|-----------------------------|---------------------------------|-------------------------------|-------------------------------------|
| Model timeout        | Retry shorter max_tokens    | Reduce context_length           | Switch to smaller model       | Serve cached responses              |
| GPU OOM              | Retry fewer gpu_layers      | Reduce batch size, CPU offload  | Full CPU inference            | Queue requests, serve cache         |
| Model file corrupt   | Re-verify GGUF hash         | Re-download from source         | Fall back to alt model        | Degraded mode                       |
| GPU driver crash     | Retry after cooldown        | Restart with CPU fallback       | Reload GPU context            | CPU-only until reboot               |

### Storage Recovery

| Failure              | T1                          | T2                              | T3                            | T4                                  |
|----------------------|-----------------------------|---------------------------------|-------------------------------|-------------------------------------|
| SQLite locked        | Retry with backoff          | Force close stale connections   | Copy, repair, swap            | Fresh DB from backup                |
| SQLite corrupt       | integrity_check + auto-fix  | Restore from backup             | Rebuild from available data   | Start fresh, log data loss          |
| ChromaDB unreachable | Restart process             | Rebuild from source embeddings  | Keyword search fallback       | Basic matching only                 |
| Disk full            | Emergency cleanup           | Compress data                   | Aggressive retention (7 days) | Read-only mode                      |

### Voice Recovery

| Failure              | T1                          | T2                              | T3                            | T4                                  |
|----------------------|-----------------------------|---------------------------------|-------------------------------|-------------------------------------|
| STT crash            | Reload Whisper              | Switch quality (medium→small)   | System speech recognition     | Text-only input                     |
| TTS failure          | Retry synthesis             | Switch engine (kokoro→piper→melo)| Reduce audio quality         | Text-only output                    |
| Audio device lost    | Re-enumerate, reconnect     | Switch to default device        | Wait for reconnection         | Text-only mode                      |

### Context Recovery

| Failure              | T1                          | T2                              | T3                            | T4                                  |
|----------------------|-----------------------------|---------------------------------|-------------------------------|-------------------------------------|
| Screen reader crash  | Restart observer            | Window titles only (no OCR)     | Disable screen reader         | No screen context                   |
| Process monitor      | Retry psutil                | Reduce frequency                | Critical processes only       | Disable process context             |
| Git scanner          | Retry git commands          | Cache last known state          | Disable git context           | No git awareness                    |

### Config Recovery

| Failure              | T1                          | T2                              | T3                            | T4                                  |
|----------------------|-----------------------------|---------------------------------|-------------------------------|-------------------------------------|
| Parse error          | Re-read, validate YAML      | Use last known good config      | Merge defaults + parseable    | Full defaults                       |
| Invalid values       | Clamp to valid range        | Reset section to defaults       | Reset entire config           | Factory reset                       |

### Self-Modification Recovery

| Failure              | T1                          | T2                              | T3                            | T4                                  |
|----------------------|-----------------------------|---------------------------------|-------------------------------|-------------------------------------|
| Import error         | Immediate rollback          | Rollback + blacklist change     | Disable Improvement for 24h   | Lock self-modification              |
| Runaway mutation     | Rate limiter                | Halt Improvement Engine         | Rollback to stable checkpoint | Disable self-modification           |

---

## 5. Self-Improvement Engine

### Architecture

```
Improvement Engine
  ├── Performance Analyzer    — profiles execution
  ├── Bottleneck Detector     — identifies hot paths
  ├── Code Patcher            — generates + applies source modifications
  ├── Architecture Evolver    — module creation/restructuring
  └── Rollback Manager        — snapshots, baselines, auto-revert
```

### The Improvement Loop

**Step 1: Observe** — Continuously profiles Homie's execution:
- Response latency per pipeline stage
- Memory/CPU/GPU utilization patterns
- Error frequency and types per module
- Code hot paths and bottlenecks
- Which modules are unused or underperforming

**Step 2: Diagnose** — Uses Homie's own LLM to analyze observations:
- Identifies root causes ("ChromaDB queries slow because collection > 1000 entries")
- Spots optimization opportunities ("stages 2 and 3 are independent, can parallelize")

**Step 3: Prescribe** — Generates a fix at the appropriate level:

| Level                      | Example                                                    |
|----------------------------|------------------------------------------------------------|
| Config tuning              | Increase batch size 32→64, reduce context_length on CPU    |
| Workflow optimization      | Parallelize pipeline stages, add LRU cache                 |
| Code patch                 | Fix inefficient loop, replace blocking call with async     |
| Architecture evolution     | Split oversized module, create caching layer, deprecate dead code |

**Step 4: Apply** — Changes go live immediately:
- Config changes: hot-reload `homie.config.yaml`
- Code changes: write to source files, reload affected modules
- Architecture changes: create/modify/remove module files, update imports

**Step 5: Monitor** — Watches metrics for regression:
- Compares key metrics to pre-change baseline
- Auto-rollback if error rate +20% or latency +50% within monitoring window

### Rollback Manager

- **Snapshots** every changed file before modification (`.homie/snapshots/`)
- **Metric baselines** captured before each change
- **Auto-rollback trigger**: error rate +20% or latency +50% within 5 minutes
- **Rollback log**: records what was tried and why it failed — prevents retrying bad fixes
- **Version chain**: sequential version IDs for every self-modification

### Constraints

- **One change at a time** — apply, monitor, stabilize, then next
- **Core lock** — immutable files: rollback manager, guardian, security vault
- **Rate limit** — max N mutations per day (configurable, default 10)
- **Evolution log** — every change recorded with reasoning, diff, and outcome metrics
- **Append-only recovery history** — Homie cannot delete its own failure records

---

## 6. Evolving Recovery — Adaptive Playbook

### Recovery Memory

Every recovery attempt recorded:

```json
{
    "failure_type": "gpu_oom",
    "module": "inference",
    "tier_attempted": "T1",
    "action_taken": "retry with fewer gpu_layers (32 → 24)",
    "outcome": "success",
    "time_to_recover_ms": 1200,
    "system_state": { "gpu_mem": "11.2GB", "model": "qwen3.5-35b", "context_len": 65536 },
    "timestamp": "2026-03-21T14:30:00Z"
}
```

### Playbook Mutations

The Improvement Engine evolves the recovery playbook over time:

- **Add new strategies** — discovers fixes not in the seed playbook
- **Reorder tiers** — if T2 consistently outperforms T1, swap them
- **Add preemptive rules** — detect conditions and act before failure
- **Remove dead strategies** — deprecate actions that never succeed

### Preemptive Healing

| Signal                                  | Preemptive Action                                    |
|-----------------------------------------|------------------------------------------------------|
| GPU memory trending toward limit        | Reduce gpu_layers or context_length proactively      |
| SQLite file size approaching disk limit | Trigger early cleanup cycle                          |
| Inference latency slowly climbing       | Clear caches, restart model, optimize settings       |
| Error rate ticking up for a module      | Increase monitoring frequency, prepare fallback      |
| Crash pattern after N hours uptime      | Schedule preemptive restart before crash window      |

### Evolution Lifecycle

```
Seed Playbook (static rules shipped with Homie)
       ↓
Observe real failures + recoveries
       ↓
Analyze patterns in recovery history
       ↓
Generate playbook mutations (via LLM)
       ↓
Apply mutation
       ↓
Monitor → if recovery improves, keep
         → if recovery worsens, rollback
       ↓
Repeat forever
```

### Constraints on Playbook Evolution

- Rollback mechanism is immutable
- Recovery history is append-only
- Preemptive rules require 3+ pattern observations before creation

---

## 7. File Structure

### New Module: `homie_core/self_healing/`

```
src/homie_core/self_healing/
├── __init__.py
├── watchdog.py                # HealthWatchdog — boot, lifecycle, probe scheduling
├── probes/
│   ├── __init__.py
│   ├── base.py                # BaseProbe, HealthStatus dataclass
│   ├── inference_probe.py
│   ├── storage_probe.py
│   ├── voice_probe.py
│   ├── context_probe.py
│   ├── network_probe.py
│   ├── knowledge_probe.py
│   └── config_probe.py
├── resilience/
│   ├── __init__.py
│   ├── decorator.py           # @resilient decorator
│   ├── circuit_breaker.py     # CircuitBreaker state machine
│   ├── retry.py               # Retry with backoff strategies
│   ├── timeout.py             # Timeout enforcement
│   └── exceptions.py          # Exception classifier
├── recovery/
│   ├── __init__.py
│   ├── engine.py              # Recovery orchestrator — tier routing
│   ├── playbook.py            # Seed rules + learned mutations
│   ├── strategies/
│   │   ├── __init__.py
│   │   ├── inference.py
│   │   ├── storage.py
│   │   ├── voice.py
│   │   ├── context.py
│   │   └── config.py
│   └── preemptive.py          # Preemptive healing rules engine
├── improvement/
│   ├── __init__.py
│   ├── engine.py              # Observe/diagnose/prescribe/apply/monitor loop
│   ├── analyzer.py            # Performance profiling
│   ├── bottleneck.py          # Hot path detection
│   ├── patcher.py             # Code generation and application
│   ├── evolver.py             # Architecture restructuring
│   └── rollback.py            # Snapshots, baselines, auto-revert
├── event_bus.py               # In-process pub/sub
├── health_log.py              # SQLite health log
├── metrics.py                 # Time-series collection, anomaly detection
└── guardian.py                # OS-level guardian process
```

### Integration Points

| Existing Module                     | Integration                                         |
|-------------------------------------|-----------------------------------------------------|
| `homie_core/inference/router.py`    | Decorate routing methods, register inference probe  |
| `homie_core/model/engine.py`        | Decorate load/generate, GPU health reporting        |
| `homie_core/storage/database.py`    | Decorate read/write, register storage probe         |
| `homie_core/memory/semantic.py`     | Decorate ChromaDB ops, register memory probe        |
| `homie_core/voice/`                 | Decorate STT/TTS, register voice probe              |
| `homie_core/context/`              | Decorate observers, register context probe          |
| `homie_core/network/`              | Decorate LAN/WebSocket, register network probe      |
| `homie_core/knowledge/`            | Decorate indexing/parsing, register knowledge probe |
| `homie_app/cli.py`                 | Boot watchdog first, shutdown last                  |
| `homie.config.yaml`                | New `self_healing:` section                         |

### Config Addition

```yaml
self_healing:
  enabled: true
  probe_interval: 30
  critical_probe_interval: 10
  improvement:
    enabled: true
    max_mutations_per_day: 10
    monitoring_window: 300
    rollback_error_threshold: 0.20
    rollback_latency_threshold: 0.50
  recovery:
    max_tier: 4
    preemptive: true
    pattern_threshold: 3
  health_log:
    retention_days: 30
    digest_enabled: true
  guardian:
    enabled: true
  core_lock:
    - self_healing/rollback.py
    - self_healing/guardian.py
    - security/
```

### Data Storage

| Data             | Location                          | Purpose                              |
|------------------|-----------------------------------|--------------------------------------|
| Health log       | `homie.db` → `health_events`     | All events, recovery actions         |
| Recovery history | `homie.db` → `recovery_history`  | Pattern learning for playbook        |
| Snapshots        | `.homie/snapshots/`              | Pre-modification file backups        |
| Evolution log    | `.homie/evolution/`              | Diffs, reasoning, metrics per change |
| Metric series    | In-memory + periodic SQLite flush | Real-time anomaly detection          |

---

## Future Sub-Projects

This spec covers sub-project 1. The remaining sub-projects will be designed separately:

2. **Adaptive Learning Engine** — response quality + preference learning
3. **Knowledge Evolution System** — knowledge graph + guided intake
4. **Performance Self-Optimizer** — inference tuning, caching, resource management
5. **Model Fine-Tuning Pipeline** — Homie trains its own model on interaction data
