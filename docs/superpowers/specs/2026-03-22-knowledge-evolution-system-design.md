# Knowledge Evolution System — Design Spec

**Date:** 2026-03-22
**Status:** Approved
**Sub-project:** 3 of 5 (Self-Healing Runtime → Adaptive Learning Engine → **Knowledge Evolution System** → Performance Self-Optimizer → Model Fine-Tuning)

---

## Vision

Upgrade Homie's knowledge from shallow, in-memory fact extraction to a persistent, temporally-versioned knowledge graph with guided intake, automatic entity resolution, relationship inference, and contradiction detection. Homie builds a deep, connected understanding of the user, their projects, their domains, and the world — evolving over time.

## Design Decisions

- **Extend existing KnowledgeBuilder** — don't replace it. The current conversation miner, project tracker, and behavioral profiler become input sources that feed into the new graph.
- **Temporal versioning** — facts have time ranges (valid_from/valid_until). Nothing is deleted, just superseded. Historical queries are supported.
- **Adaptive depth intake** — surface pass on everything, deep LLM analysis only on high-value content (top 20%).
- **Full graph reasoning** — entity resolution, transitive relationship inference (max 2 hops), and automatic contradiction detection/resolution.
- **Three knowledge domains** — personal (user, preferences, goals), domain (codebases, tools, architectures), and world (documents, articles, research).

---

## 1. Architecture Overview

Three new layers on top of the existing KnowledgeBuilder:

### Existing (Sub-project 2)
- **ConversationMiner** — regex-based fact extraction from conversations
- **ProjectTracker** — in-memory project knowledge triples
- **BehavioralProfiler** — hourly pattern aggregation
- **LearningStorage** — project_knowledge and decisions_log tables

### New Layers

**Persistent Knowledge Graph** — entities + typed relationships with temporal versioning, stored in SQLite. Replaces in-memory triples.

**Guided Intake Pipeline** — user points at directories, repos, or documents. Adaptive depth: surface pass on everything, deep LLM analysis on high-value files.

**Knowledge Reasoning Engine** — entity resolution (merge duplicates), relationship inference (derive new facts from existing), contradiction detection (flag and resolve conflicts using temporal versioning).

### Data Flow

```
Input Sources:
  ConversationMiner → extracts facts from chat
  ProjectTracker → detects project activity
  Guided Intake → user-directed source scanning
       ↓
  Knowledge Graph (persistence)
  ├── Entity Store
  ├── Relationship Store
  └── Temporal Versioning
       ↓
  Reasoning Engine (intelligence)
  ├── Entity Resolution
  ├── Relationship Inference
  └── Contradiction Detection
```

---

## 2. Persistent Knowledge Graph

### Entity Model

```python
@dataclass
class Entity:
    id: str                    # unique identifier
    name: str                  # "Python", "Homie AI", "Google"
    entity_type: str           # "person", "project", "technology", "concept", "organization"
    aliases: list[str]         # ["Python 3", "Python3", "CPython"]
    properties: dict           # flexible key-value metadata
    created_at: float
    updated_at: float
```

### Relationship Model

```python
@dataclass
class Relationship:
    id: str
    subject_id: str            # entity ID
    predicate: str             # "uses", "works_at", "depends_on", "created_by", "prefers"
    object_id: str             # entity ID
    confidence: float          # 0.0-1.0
    source: str                # "conversation", "code_scan", "document", "inference"
    valid_from: float          # timestamp — when this became true
    valid_until: float | None  # None = still true, timestamp = superseded
    properties: dict           # extra context
    created_at: float
```

### Temporal Versioning

Every relationship has a time range. When a contradicting fact arrives:
1. Find existing relationship with same subject + predicate
2. Set `valid_until` on old relationship to now
3. Create new relationship with `valid_from` = now, `valid_until` = None
4. Both remain queryable by time range

Example:
```
("user", "works_at", "Google",    valid: 2023-01 → 2025-06)
("user", "works_at", "Anthropic", valid: 2025-06 → present)
```

### Storage

Two SQLite tables in `learning.db`:

**`kg_entities`**: id, name, entity_type, aliases (JSON), properties (JSON), created_at, updated_at

**`kg_relationships`**: id, subject_id, object_id, predicate, confidence, source, valid_from, valid_until, properties (JSON), created_at

Indexed on: subject_id, object_id, predicate, entity_type, valid_until.

---

## 3. Guided Intake Pipeline

### Flow

```
User points at source (directory, file, repo)
  → Source Scanner (enumerate files, detect types)
  → Surface Pass (quick: AST for code, headings for docs)
  → Value Scoring (which files deserve deep analysis?)
  → Deep Pass (LLM reads high-value files, extracts knowledge)
  → Entity Resolution (merge with existing graph)
  → Knowledge Graph updated
```

### Adaptive Depth

**Surface pass** (runs on everything, fast):
- File metadata: path, type, size, last modified
- Code: imports, class/function names, docstrings (AST-based, no LLM)
- Documents: headings, key terms, named entities (regex/heuristic)
- Creates entities for files, modules, classes

**Value scoring** (decides what gets deep analysis):
- Files referenced by many others (imports) → high value
- Recently modified (git) → high value
- User has asked about in conversation → high value
- Large files with sparse surface extraction → worth deeper look
- Top 20% by score get deep pass

**Deep pass** (LLM-powered, selective):
- Reads content, extracts: architecture decisions, design patterns, component relationships
- Creates rich relationships: "module X depends on Y", "class A implements pattern B"
- Queued and rate-limited — doesn't compete with user conversations

### Supported Sources

| Source | Scanner | Surface | Deep |
|--------|---------|---------|------|
| Directory | Walk files | AST/headings | LLM analysis |
| Git repo | Walk + git metadata | AST + git context | LLM analysis |
| Single file | Direct | AST/headings | LLM analysis |
| Document (PDF/DOCX) | RAG parsers | Key terms, structure | LLM summarization |

Reuses existing `RagPipeline._parse_any_document()` for document parsing.

---

## 4. Knowledge Reasoning Engine

### Entity Resolution

Detects when different sources refer to the same entity:

1. New entity arrives (e.g., "Python3" from code scan)
2. Check existing entities: exact name → merge, alias match → merge, fuzzy match (>0.85 similarity) → candidate
3. High-confidence candidates (>0.9) auto-merged. Lower confidence flagged for LLM confirmation.

Merge: aliases consolidated, properties merged (newer wins), relationships repointed to surviving entity.

### Relationship Inference

Derives new facts from graph connections (max 2 hops):

| Rule | Example | Confidence |
|------|---------|------------|
| `(A, works_on, B)` + `(B, uses, C)` → `(A, works_with, C)` | User works on Homie + Homie uses ChromaDB → User works with ChromaDB | parent * 0.7 |
| `(A, depends_on, B)` + `(B, depends_on, C)` → `(A, indirectly_depends_on, C)` | Module A → B → C dependency chain | parent * 0.5 |
| `(A, member_of, B)` + `(B, located_in, C)` → `(A, located_in, C)` | Person in org, org in city | parent * 0.4 |

Inferred relationships: marked `source: "inference"`, lower confidence, re-evaluated when source facts change. Runs in batches after new knowledge arrives, not per-fact.

### Contradiction Detection

1. **Detect:** Same subject + predicate, different object, both current (`valid_until = None`)
2. **Classify:** Soft (version update: "Python 3.11" → "Python 3.12") vs Hard (mutually exclusive: "works at Google" vs "works at Anthropic")
3. **Resolve:** Higher confidence wins as current. Apply temporal versioning — old fact gets `valid_until = now`. Equal confidence → recency wins.
4. **Log:** All contradictions recorded.

### Confidence Propagation

| Source | Base confidence |
|--------|----------------|
| Explicit user statement | 0.95 |
| Code scan (AST) | 0.85 |
| Deep LLM analysis | 0.75 |
| Conversation extraction | 0.7 |
| Document extraction | 0.7 |
| Inference (1 hop) | parent * 0.7 |
| Inference (2 hops) | parent * 0.5 |

Time decay: `effective = base * 0.99^age_days` (~70 days to lose 50%).

---

## 5. File Structure

### New/Modified Files

```
src/homie_core/adaptive_learning/knowledge/
├── __init__.py                    # Update exports
├── builder.py                     # Modify — wire graph + intake + reasoning
├── conversation_miner.py          # Existing — now feeds graph
├── project_tracker.py             # Modify — persist to graph instead of memory
├── behavioral_profiler.py         # Existing — unchanged
├── graph/
│   ├── __init__.py
│   ├── models.py                  # Entity, Relationship dataclasses
│   ├── store.py                   # KnowledgeGraphStore — SQLite CRUD
│   ├── query.py                   # GraphQuery — current/historical facts, traversal
│   └── temporal.py                # TemporalManager — versioning, contradiction detection, decay
├── intake/
│   ├── __init__.py
│   ├── pipeline.py                # IntakePipeline coordinator
│   ├── scanner.py                 # SourceScanner — enumerate files, detect types
│   ├── surface_extractor.py       # SurfaceExtractor — AST/regex quick extraction
│   ├── deep_analyzer.py           # DeepAnalyzer — LLM-powered extraction
│   └── value_scorer.py            # ValueScorer — score files for deep analysis
└── reasoning/
    ├── __init__.py
    ├── entity_resolver.py         # EntityResolver — detect and merge duplicates
    ├── inference_engine.py        # InferenceEngine — derive relationships
    └── contradiction_detector.py  # ContradictionDetector — find and resolve conflicts
```

### Integration Points

| Existing Module | Change |
|----------------|--------|
| `adaptive_learning/knowledge/builder.py` | Wire graph + intake + reasoning |
| `adaptive_learning/knowledge/project_tracker.py` | Persist to graph instead of in-memory dict. Deprecate existing `project_knowledge` table (replaced by `kg_relationships`) |
| `adaptive_learning/storage.py` | Add kg_entities and kg_relationships tables |
| `adaptive_learning/learner.py` | Expose intake and graph query methods |
| `homie_core/rag/pipeline.py` | Reuse `_parse_any_document` for intake |
| `homie_core/self_healing/watchdog.py` | Register new `KnowledgeGraphProbe` (separate from existing `KnowledgeProbe` which monitors RAG) |
| `homie.config.yaml` | Add `knowledge_graph:` top-level section |
| `homie_core/config.py` | Add `KnowledgeGraphConfig` class and field on `HomieConfig` (top-level, not nested under adaptive_learning) |

### Config Addition

```yaml
knowledge_graph:
  enabled: true
  intake:
    surface_pass: true
    deep_pass: true
    deep_pass_top_percent: 20
    max_deep_files_per_batch: 50
  reasoning:
    entity_resolution: true
    relationship_inference: true
    max_inference_hops: 2
    inference_batch_interval: 300
  temporal:
    confidence_decay_rate: 0.99
    contradiction_auto_resolve: true
```

### New SQLite Tables (in LearningStorage's database)

| Table | Purpose |
|-------|---------|
| `kg_entities` | Entity store with type, aliases, properties |
| `kg_relationships` | Typed relationships with confidence, source, temporal range |
