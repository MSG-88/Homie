# Adaptive Learning Engine — Design Spec

**Date:** 2026-03-22
**Status:** Approved
**Sub-project:** 2 of 5 (Self-Healing Runtime → **Adaptive Learning Engine** → Knowledge Evolution → Performance Optimizer → Model Fine-Tuning)

---

## Vision

Make Homie continuously improve its responses, speed, and understanding by learning from every interaction. Three engines work together through integrated feedback loops: PreferenceEngine (response quality), PerformanceOptimizer (speed/efficiency), and KnowledgeBuilder (depth/context).

## Design Decisions

- **Dual signal sources**: Explicit feedback (user corrections) + implicit signals (behavioral inference)
- **Context-aware preferences**: Layered profiles — global defaults + domain/project/temporal overrides
- **Integrated feedback loops**: Engines share data and trigger each other (knowledge informs preferences, preferences inform caching, behavioral patterns inform resource scheduling)
- **Simple algorithms**: Exponential moving averages, frequency counts, cosine similarity — no ML training needed
- **Self-prompt modification**: PreferenceEngine generates a dynamic preference layer prepended to the system prompt

---

## 1. Architecture Overview

### Layer 1: Observation Pipeline

Captures all learning signals into a unified **ObservationStream**:

```python
@dataclass
class LearningSignal:
    signal_type: str       # "explicit", "implicit", "behavioral"
    category: str          # "preference", "engagement", "context", "performance"
    source: str            # "user_feedback", "response_timing", "app_tracker", etc.
    data: dict             # signal-specific payload
    context: dict          # current state — topic, app, time, query complexity
    timestamp: float
```

Signal sources:
- **Explicit** (high confidence, α=0.3): Direct user corrections, preference statements
- **Implicit** (medium confidence, α=0.05): Clarification requests, interruptions, re-asks, response timing, copy/paste detection
- **Behavioral** (continuous background): Active app, time of day, git activity, conversation topic, query complexity

The pipeline hooks into existing infrastructure — context aggregator, CognitiveArchitecture, middleware hooks, working memory. A new **LearningMiddleware** captures implicit signals via the `after_turn` hook.

### Layer 2: Three Learning Engines

**PreferenceEngine** — Learns response style (verbosity, formality, depth, format, explanation style). Context-layered profiles with exponential moving average learning.

**PerformanceOptimizer** — Semantic response cache, context window relevance learning, resource scheduling with usage pattern prediction and pre-fetching.

**KnowledgeBuilder** — Conversation mining (fact/decision extraction), project tracking (git-based knowledge graph), behavioral profiling (work rhythms, tool preferences).

### Layer 3: Shared Learning Memory

All engines read/write to shared SQLite tables and ChromaDB collections, enabling cross-engine feedback loops. Thread-safe with write locks.

---

## 2. Observation Pipeline — Signal Collection

### Explicit Signals

| Signal | Detection | Payload |
|--------|-----------|---------|
| Direct correction | User says "shorter", "more detail", "less formal" | `{"dimension": "verbosity", "direction": "decrease"}` |
| Preference statement | User says "I prefer bullet points" | `{"dimension": "format", "value": "bullets"}` |
| Topic preference | User says "skip the explanation" | `{"dimension": "depth", "direction": "decrease"}` |

### Implicit Signals

| Signal | Detection | Meaning |
|--------|-----------|---------|
| Clarification request | User asks "what do you mean?" or rephrases | Response unclear or wrong depth |
| Interruption | New message before response completes | Response too long or off-track |
| Re-ask | Same intent phrased differently within 2 turns | First response missed the mark |
| Quick follow-up | User responds in < 3 seconds | Engaged, response useful |
| Long pause then topic change | User takes > 60s then switches topic | Response wasn't useful |
| Copy/paste from response | Clipboard contains response text | Response had actionable content |

### Behavioral Signals

| Signal | Source | Purpose |
|--------|--------|---------|
| Active application | Context aggregator | Work context detection |
| Time of day | System clock | Daily rhythm learning |
| Git activity | Git scanner | Active project detection |
| Conversation topic | CognitiveArchitecture classifier | Context for preference layering |
| Query complexity | CognitiveArchitecture classifier | Response depth calibration |

### Integration with Existing Systems

- **Context aggregator** → behavioral signals
- **CognitiveArchitecture** → topic classification, query complexity
- **Middleware hooks** (`after_turn`) → implicit signals
- **Working memory** → conversation state for re-ask detection

---

## 3. PreferenceEngine — Learning Response Style

### PreferenceProfile

Multi-dimensional profile describing how Homie should respond:

```python
@dataclass
class PreferenceProfile:
    verbosity: float        # 0.0 (terse) → 1.0 (verbose), default 0.5
    formality: float        # 0.0 (casual) → 1.0 (formal), default 0.5
    technical_depth: float  # 0.0 (simple) → 1.0 (expert), default 0.7
    format_preference: str  # "prose", "bullets", "code_first", "mixed"
    explanation_style: str  # "bottom_up", "top_down", "example_first"
    confidence: float       # confidence in this profile (0.0 → 1.0)
    sample_count: int       # number of signals that built this profile
```

### Context Layering

Preferences stored in layers, resolved most-specific-first:

| Layer | Scope | Resolution order |
|-------|-------|-----------------|
| **Temporal** | Time-based patterns | 1st (highest priority) |
| **Project** | Specific project context | 2nd |
| **Domain** | Topic category (coding, personal, work, research) | 3rd |
| **Global** | All interactions | 4th (fallback) |

### Learning Algorithm

Exponential moving average per dimension:

1. Signal arrives (e.g., user said "shorter")
2. Map to dimension (verbosity) and direction (decrease)
3. Update: `new_value = α * signal_value + (1-α) * current_value`
4. α depends on confidence: explicit=0.3 (fast), implicit=0.05 (slow drift)
5. Increment sample_count, recalculate profile confidence

### Applying Preferences

PreferenceEngine generates a preference prompt layer prepended to Homie's system prompt before each response:

```
[Learned preferences for this context]
- Response length: concise (30-50% shorter than default)
- Format: prefer bullet points over prose
- Technical depth: expert level, skip basic explanations
- Style: lead with code examples, explain after
```

Injected via middleware `before_turn` hook on the BrainOrchestrator.

### Persistence

- Profiles in `homie.db` → `preference_profiles` keyed by `(layer_type, context_key)`
- All signals logged to `learning_signals` for audit and relearning

---

## 4. PerformanceOptimizer — Speed & Efficiency

### Response Cache

Semantic similarity-based cache using vector embeddings:

1. Every response stored with query embedding
2. On new query, search cache by cosine similarity (threshold: 0.92)
3. On hit, validate context freshness — skip stale cache
4. Eviction: LRU with max 500 entries, per-entry TTL (default 24h, shorter for volatile topics)
5. Invalidation: KnowledgeBuilder signals when relevant knowledge changes

### Context Window Optimizer

Learns which context sources improve response quality per query type:

- Tracks whether retrieved context was referenced in the response
- Builds relevance scores per `(query_type, context_source)` pair
- Prioritizes high-relevance sources, skips low-relevance when approaching context limits

### Resource Scheduler

Learns usage patterns for proactive resource management:

| Pattern | Action |
|---------|--------|
| User starts coding at 9am | Pre-load model at 8:55am |
| No interaction for 30 min | Reduce GPU layers, free VRAM |
| User switches to browser | Keep model warm, reduce priority |
| Heavy inference period | Increase GPU layers, pause background tasks |
| Overnight idle (DND) | Unload model, run maintenance |

Pre-fetch hints from KnowledgeBuilder: project opened → cache project context. Behavioral prediction → pre-load relevant knowledge.

### Persistence

- `response_cache` table: query_embedding, response, context_hash, ttl, hit_count
- `context_relevance` table: query_type, context_source, relevance_score, sample_count
- `resource_patterns` table: time histograms, load patterns, scheduling rules

---

## 5. KnowledgeBuilder — Deep Understanding

### Conversation Miner

Extracts structured knowledge from every conversation turn:

| Type | Example | Storage |
|------|---------|---------|
| Facts | "User's main project is Homie AI" | Semantic memory (existing) |
| Decisions | "Chose Python over Rust" | `decisions_log` table |
| Preferences | "Prefers TDD workflow" | Feeds PreferenceEngine |
| Relationships | "Homie AI uses ChromaDB" | `project_knowledge` table |

Runs asynchronously as `after_turn` hook. Uses Homie's own LLM with structured extraction prompt. Deduplicates against existing knowledge via semantic similarity.

### Project Tracker

Lightweight knowledge graph of user's projects:

| Entity | Source |
|--------|--------|
| Projects | Git repos via git scanner |
| Files | File system + git |
| Dependencies | pyproject.toml, package.json |
| Recent changes | Git log/diff |
| Architecture | Directory structure + imports |

Passive scanning via existing git scanner and context aggregator. On-demand deep scan when user points at new repo. Stored as entity-relationship triples in `project_knowledge` table. Refreshes on git events.

### Behavioral Profiler

Learns work patterns over sliding windows (1 day, 1 week, 1 month):

| Pattern | Source | Consumer |
|---------|--------|----------|
| Work schedule | Active hours | ResourceScheduler |
| Tool preferences | App usage frequency | PreferenceEngine |
| Productivity rhythms | Deep work detection | PreferenceEngine (temporal) |
| Topic patterns | Conversation topics by time | PerformanceOptimizer (pre-fetch) |
| Interaction frequency | Messages per hour | ResourceScheduler |

Simple statistics — moving averages, frequency counts, time-of-day histograms.

### Knowledge Feedback Loops

```
KnowledgeBuilder: "user works on Homie 9am-12pm"
  → PerformanceOptimizer: pre-fetch Homie context at 8:55am
  → PreferenceEngine: activate "coding" preferences automatically

KnowledgeBuilder: "user switched from coding to email"
  → PerformanceOptimizer: swap cached context
  → PreferenceEngine: switch to "communication" preferences

KnowledgeBuilder: "user prefers async over threading"
  → PreferenceEngine: add project-level coding preference
```

---

## 6. File Structure

### New Module: `homie_core/adaptive_learning/`

```
src/homie_core/adaptive_learning/
├── __init__.py
├── learner.py                  # AdaptiveLearner — central coordinator
├── observation/
│   ├── __init__.py
│   ├── stream.py               # ObservationStream — signal collector
│   ├── signals.py              # LearningSignal dataclass, signal types
│   └── learning_middleware.py   # Middleware hook for implicit signal capture
├── preference/
│   ├── __init__.py
│   ├── engine.py               # PreferenceEngine — learning algorithm
│   ├── profile.py              # PreferenceProfile, context layering
│   └── prompt_builder.py       # Generates preference prompt layer
├── performance/
│   ├── __init__.py
│   ├── optimizer.py            # PerformanceOptimizer coordinator
│   ├── response_cache.py       # Semantic similarity cache
│   ├── context_optimizer.py    # Context window relevance learning
│   └── resource_scheduler.py   # Usage pattern learning, pre-load/unload
├── knowledge/
│   ├── __init__.py
│   ├── builder.py              # KnowledgeBuilder coordinator
│   ├── conversation_miner.py   # Fact/decision/relationship extraction
│   ├── project_tracker.py      # Project knowledge graph
│   └── behavioral_profiler.py  # Work pattern learning
├── customization/
│   ├── __init__.py
│   ├── generator.py            # LLM-powered code generation from natural language
│   └── manager.py              # Customization lifecycle (create/list/modify/disable/rollback)
└── storage.py                  # Learning memory — SQLite tables, shared access
```

### Integration Points

| Existing Module | Integration |
|----------------|-------------|
| `homie_core/middleware/` | New LearningMiddleware in stack |
| `homie_core/brain/orchestrator.py` | Preference prompt layer via `before_turn` |
| `homie_core/brain/cognitive_arch.py` | Topic classification → ObservationStream |
| `homie_core/context/aggregator.py` | Behavioral signals → ObservationStream |
| `homie_core/context/git_scanner.py` | Project changes → KnowledgeBuilder |
| `homie_core/memory/semantic.py` | Mined facts stored here |
| `homie_core/storage/vectors.py` | Cache similarity via ChromaDB |
| `homie_core/intelligence/` | Observer loop, flow detector → behavioral profiler |
| `homie_core/self_healing/watchdog.py` | AdaptiveLearner registers health probe |
| `homie_core/self_healing/event_bus.py` | Engines publish learning events |
| `homie_app/cli.py` | Boot AdaptiveLearner after HealthWatchdog |

### Config Addition

```yaml
adaptive_learning:
  enabled: true
  preference:
    learning_rate_explicit: 0.3
    learning_rate_implicit: 0.05
    min_signals_for_confidence: 10
  performance:
    cache_enabled: true
    cache_max_entries: 500
    cache_ttl_default: 86400
    cache_similarity_threshold: 0.92
    context_optimization: true
    resource_scheduling: true
  knowledge:
    conversation_mining: true
    project_tracking: true
    behavioral_profiling: true
    scan_interval: 300
  feedback_loops: true
```

### New SQLite Tables

| Table | Purpose |
|-------|---------|
| `learning_signals` | Raw signal log — all observations, append-only |
| `preference_profiles` | Learned preferences per (layer, context) |
| `response_cache` | Cached query-response pairs with embeddings |
| `context_relevance` | Relevance scores per (query_type, context_source) |
| `resource_patterns` | Usage patterns (time histograms, load patterns) |
| `project_knowledge` | Entity-relationship triples for project graph |
| `behavioral_patterns` | Aggregated work patterns (daily, weekly, monthly) |
| `decisions_log` | Extracted decisions from conversations |

### Self-Healing Integration

- AdaptiveLearner registers a **learning probe** with HealthWatchdog
- Recovery strategies: restart engines, clear stale cache, rebuild profiles from signal log
- `@resilient` decorator on boundary operations (cache writes, LLM extraction calls)

---

## 7. User-Requested Customizations

Users can request customizations in natural language. Homie implements them through full self-modification — generating real code (middleware, tools, hooks, prompt changes) using the self-healing runtime's CodePatcher, ArchitectureEvolver, and RollbackManager.

### Request Flow

1. User describes desired behavior in natural language
2. Homie's LLM analyzes intent — what condition triggers it, what behavior is desired
3. LLM generates implementation code (middleware class, tool, prompt modification, scheduled task)
4. RollbackManager snapshots before modification
5. ArchitectureEvolver/CodePatcher applies the code
6. New code registers in the system (hot-reload — middleware stack, tool registry, or scheduler)
7. Self-healing runtime monitors for regressions → auto-rollback if errors spike
8. Customization logged to `customization_history`

### What Homie Can Generate

| Request type | Generated code |
|-------------|---------------|
| "Greet me with a joke each morning" | New middleware that injects humor prompt on first interaction of the day |
| "When I say /standup, show git + calendar" | New tool registered in ToolRegistry with slash command |
| "Always suggest type hints in Python" | Prompt layer addition in PreferenceEngine |
| "Create a weekly summary of what I worked on" | New scheduled task + tool that aggregates behavioral data |
| "Analyze my PR review patterns" | New tool + knowledge extraction pipeline |

### Safety

All powered by existing self-healing infrastructure:
- **RollbackManager** → snapshots before every change
- **@resilient** → wraps generated code
- **HealthWatchdog** → monitors for regressions
- **Auto-rollback** → if error rate spikes after a customization, it's reverted
- **Core lock** → generated code cannot modify security or rollback systems

### Customization Management

Users manage customizations through natural language:
- "Show my customizations" → lists all generated customizations with status
- "Disable the morning joke" → deactivates (doesn't delete — can re-enable)
- "Change the standup to include PRs" → LLM modifies existing generated code via CodePatcher

### Persistence

- `customization_history` table: request text, generated file paths, status (active/disabled/rolled_back), version_id, creation timestamp
- Generated files tracked by RollbackManager's evolution log
- Each customization gets a version ID for rollback

### New Files

```
src/homie_core/adaptive_learning/
├── customization/
│   ├── __init__.py
│   ├── generator.py           # LLM-powered code generation from natural language requests
│   └── manager.py             # Customization lifecycle (create/list/modify/disable/rollback)
```

### Additional SQLite Table

| Table | Purpose |
|-------|---------|
| `customization_history` | Request text, generated paths, status, version_id, timestamps |
