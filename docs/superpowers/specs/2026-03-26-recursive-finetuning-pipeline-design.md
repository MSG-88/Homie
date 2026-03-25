# Recursive Finetuning Pipeline — Design Specification

**Date:** 2026-03-26
**Status:** Approved
**Author:** Muthu Subramanian G + Claude

## Overview

A fully local, recursive self-improving finetuning pipeline for the Homie personal AI assistant model (`PyMasters/Homie`). The pipeline generates synthetic training data (zero user data), finetunes via QLoRA, evaluates against a comprehensive benchmark suite, and loops — each cycle targeting weak areas with harder data — until performance plateaus.

### Base Model

The finetuning base model is `lfm2` (14GB, the same model backing `PyMasters/Homie` on Ollama). The project also uses `GLM-4.7-Flash` (18GB) as the local inference model — these serve different purposes. The finetuning pipeline operates exclusively on `lfm2` because it is the Ollama-registered model that gets pushed to the registry. The inference router may serve either model depending on configuration.

### Registry Note

The Ollama registry was renamed from `MSG-88/Homie` to `PyMasters/Homie` on 2026-03-26 when the Ollama account username changed. All references in this spec use the new name. The old `MSG-88/Homie` registry entry is abandoned. Tests referencing the old name should be updated during implementation.

### Minimum GPU Requirement

**Target:** NVIDIA GPU with >= 12GB VRAM (RTX 3060 12GB or better).

| Stage | VRAM Usage | Notes |
|-------|-----------|-------|
| Data generation | ~0 GPU | Uses cloud API for teacher |
| QLoRA training (lfm2 14B, 4-bit) | ~10-12 GB | With unsloth optimizations |
| Evaluation (inference) | ~8 GB | Single model loaded, 4-bit |
| Merge + quantize | ~10 GB | Peak during merge, then drops |

If VRAM is insufficient at runtime, the pipeline will: (1) reduce batch size to 1, (2) enable CPU offloading for optimizer states, (3) if still OOM, abort cycle with a clear error and skip to next idle window.

### Platform Note (Windows)

The target environment is Windows 11. Key considerations:
- `unsloth` requires WSL2 on Windows. The pipeline detects the OS and runs training inside WSL2 if on Windows, falling back to native `peft` if WSL2 is unavailable.
- `bitsandbytes >= 0.43` has native Windows support. Pin this version.
- GGUF conversion uses `llama.cpp` binary (pre-built Windows release or WSL2).
- Idle detection uses `ctypes` + `GetLastInputInfo` on Windows for keyboard/mouse monitoring.

The model is optimized for 6 capability domains in priority order:
1. Intent Understanding (A) — 25%
2. Personal Context Reasoning (C) — 20%
3. Conversational Intelligence (D) — 20%
4. Task Orchestration (B) — 15%
5. System Self-Awareness (E) — 10%
6. Safety & Privacy (F) — 10%

A cross-cutting behavior — **Proactive Information Gathering** — is woven into all domains, teaching the model to anticipate user needs and have relevant context ready before being asked.

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                 RECURSIVE FINETUNE LOOP              │
│                                                      │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐        │
│  │ GENERATE │──→│  TRAIN   │──→│ EVALUATE │──┐     │
│  │ Synthetic│   │  QLoRA   │   │Benchmark │  │     │
│  │   Data   │   │ Finetune │   │  Suite   │  │     │
│  └──────────┘   └──────────┘   └──────────┘  │     │
│       ↑                              │        │     │
│       │         ┌──────────┐         │        │     │
│       │         │  MERGE   │←────────┘        │     │
│       │         │ & DEPLOY │  (if improved)   │     │
│       │         └──────────┘                  │     │
│       │              │                        │     │
│       └──────────────┘ (harder data           │     │
│         from weak areas)    (if plateau ×3)───┘     │
│                              → STOP                  │
└─────────────────────────────────────────────────────┘
```

### Stages

1. **GENERATE** — Synthetic data generator uses cloud fallback (Qubrid/Vertex) as teacher to create SFT and DPO pairs across 6 domains. All scenarios procedurally generated from templates + randomized contexts.
2. **TRAIN** — QLoRA finetuning locally using unsloth/peft + trl. LoRA rank 16-32.
3. **EVALUATE** — Multi-category benchmark suite scores the finetuned model. Cloud fallback judges open-ended responses.
4. **MERGE & DEPLOY** — Merge LoRA into base, quantize to GGUF, import into Ollama, validate, push to PyMasters/Homie.

### Integration with Existing System

- New method `EvolutionEngine.evolve_finetune()` alongside existing `evolve()` (Modelfile-level).
- Both feed into the same `ModelRegistry` for unified version tracking.
- Modelfile-level updates remain fast for preference/knowledge changes.
- Finetune cycles run less frequently when sufficient data accumulates.

**DataCurator Integration:** The existing `DataCurator` (collects real interaction SFT/DPO data) and the new `SyntheticDataGenerator` are complementary sources. The training pipeline merges both:
- Synthetic data: primary source (no user data constraint applies to synthetic).
- DataCurator data: excluded from finetuning by default (respects the "no user data" rule). If the user explicitly opts in via config (`finetune.data.include_curated: true`), curated data is mixed at a 70/30 synthetic/curated ratio.

**ModelValidator Coexistence:** The existing 5-prompt `ModelValidator` remains active for Modelfile-level `evolve()`. The new 30-test benchmark suite (`finetune/evaluation/benchmark.py`) is used exclusively by the finetune pipeline. They share no state and can run independently.

**Concurrent Model Access:** During deployment, the pipeline uses a staging name `PyMasters/Homie:candidate` for import and validation. Only after validation passes does it atomically swap: `ollama cp PyMasters/Homie:candidate PyMasters/Homie:latest`. This prevents mid-conversation model replacement.

**Teacher Dependency & Fallback:** If cloud APIs are unavailable during data generation:
1. Retry 3 times with exponential backoff (1min, 5min, 15min).
2. If still unavailable, use the current local model as a weaker teacher (self-play mode). Quality filter threshold increases from 4 to 5 to compensate.
3. If local teacher also unavailable, defer the cycle to the next idle window.
- Rate limiting: generation is throttled to 10 requests/minute to avoid cloud API limits. Expected time for 3,000 examples: ~5 hours.

**Stacked QLoRA Drift Prevention:** Training always starts from the **original base model** (`lfm2`), not from the previous cycle's merged weights. Each cycle accumulates all data from cycles 0..N and trains from scratch. This prevents compounding quality degradation across cycles. Trade-off: training time grows with each cycle, but the dataset is small enough (~3K examples × 10 max cycles = 30K max) that this remains feasible locally.

---

## 1. Synthetic Data Generation

### Data Types

| Type | Format | Purpose |
|------|--------|---------|
| SFT | `{system, user, assistant}` | Teach correct behavior patterns |
| DPO | `{system, user, chosen, rejected}` | Teach preference between good and bad responses |

### Domain Allocation Per Cycle

| Domain | Base % | Example Scenarios |
|--------|--------|-------------------|
| Intent Understanding (A) | 25% | Ambiguous requests, multi-part commands, implicit intent, typos/informal language |
| Context Reasoning (C) | 20% | Using project/preference/schedule context to personalize, recalling prior facts |
| Conversational Intelligence (D) | 20% | When to clarify vs act, tone adaptation, multi-turn coherence |
| Task Orchestration (B) | 15% | Multi-step tool chains, dependency ordering, parallel vs sequential |
| System Self-Awareness (E) | 10% | Capability boundaries, plugin/service awareness, suggesting alternatives |
| Safety & Privacy (F) | 10% | Credential leak attempts, prompt injection resistance, privacy boundaries |

### Generation Process

1. **Scenario Templates** — ~50 handcrafted templates per domain (300 total). Each defines scenario structure, expected good behavior, and expected bad behavior.

2. **Context Randomizer** — Generates fictional user contexts (projects, preferences, schedules, connected services) so the model learns to reason over context without memorizing real data.

3. **Teacher Generation** — Cloud fallback generates:
   - SFT: ideal assistant response
   - DPO: ideal (chosen) + plausible-but-worse (rejected) response pair

4. **Quality Filter** — Teacher scores each example 1-5 on relevance, correctness, naturalness. Only 4+ kept.

### Proactive Information Gathering (Cross-Cutting, ~15% of Data)

Synthetic templates that teach anticipatory behavior:
- **Context-aware greeting** — model receives time + profile + recent activity, produces contextually appropriate opening
- **Anticipatory follow-up** — model identifies and addresses the likely next question
- **Information triage** — model selects 2-3 most relevant pieces from 10 available for the current query
- **Context restraint** — model receives lots of context but user asks something simple, must not over-share

### Data Format

SFT and DPO data use ChatML format (`{system, user, assistant}` / `{system, user, chosen, rejected}`). The existing `DataCurator` uses Alpaca format (`{instruction, input, output}`). The training module includes a format adapter that converts Alpaca → ChatML at load time if curated data is opted in.

### Quality Filter Minimums

If quality filtering drops a domain below 50% of its target allocation, the generator retries that domain with relaxed constraints (up to 3 retries). If still insufficient, a warning is logged and the cycle proceeds with available data — the evaluation will catch any domain regression.

### Target Per Cycle

- ~2,000 SFT examples + ~1,000 DPO pairs
- Sufficient for meaningful QLoRA training without overfitting

### Curriculum Escalation

After each cycle, evaluator identifies weakest domain. Next cycle generates 40% data targeting that domain (up from base allocation), with harder variations.

---

## 2. Local QLoRA Training

### Stack

- `unsloth` — 2x faster QLoRA with fused kernels (primary)
- `peft` — LoRA adapter management (fallback)
- `trl` — SFTTrainer + DPOTrainer
- `bitsandbytes` — 4-bit quantization

### Training Phases Per Cycle

| Phase | Trainer | Data | Epochs | Purpose |
|-------|---------|------|--------|---------|
| Phase 1: SFT | SFTTrainer | ~2,000 examples | 3 | Teach capability patterns |
| Phase 2: DPO | DPOTrainer | ~1,000 pairs | 1 | Align preferences, sharpen judgment |

### Hyperparameters

```
LoRA rank:           16 (bump to 32 on plateau)
LoRA alpha:          32
LoRA target modules: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
Learning rate:       2e-4 (SFT), 5e-5 (DPO)
Batch size:          4 (gradient accumulation 4 → effective 16)
Warmup ratio:        0.05
Scheduler:           cosine
Max seq length:      4096
Optimizer:           adamw_8bit
```

### Post-Training Pipeline

1. **Merge** — LoRA adapters merged into base model weights
2. **Quantize** — Convert to GGUF Q4_K_M via llama.cpp
3. **Import** — `ollama create PyMasters/Homie -f Modelfile`
4. **Validate** — Run benchmark suite
5. **Push** — `ollama push PyMasters/Homie` (if validation passes)

### Checkpoints

- LoRA adapters saved at `~/.homie/finetune/adapters/cycle-{N}/`
- Checkpoint every 50 training steps for pause/resume
- Failed validation → adapter discarded, previous version stays active

---

## 3. Benchmark & Evaluation Suite

### Test Cases (30 total)

| Domain | # Tests | Scoring Method |
|--------|---------|---------------|
| Intent Understanding | 8 | Does response correctly identify all intents, disambiguate, address each? |
| Context Reasoning | 6 | Inject synthetic context, verify correct usage without hallucination |
| Conversational Intelligence | 5 | Multi-turn: clarify when needed, act when clear, match tone |
| Task Orchestration | 4 | Tool call sequence correctness, parameter accuracy, dependency handling |
| System Self-Awareness | 4 | Accurate capability representation |
| Safety & Privacy | 3 | Credential extraction, prompt injection, data boundary violations |

### Scoring

- **Automated checks** (60%) — regex/structural for format, tool calls, refusals
- **Judge scoring** (40%) — Cloud fallback grades 1-5 on accuracy, helpfulness, safety
- **Final score** = weighted average (A=25%, C=20%, D=20%, B=15%, E=10%, F=10%)

### Promotion Gate

- Overall score must exceed current active model by **>= 2%**
- No individual domain may **regress by > 5%**
- Safety domain must score **>= 0.85** always (hard floor)

### Plateau Detection

- 3 consecutive cycles with < 2% improvement → escalate LoRA rank
- If already at rank 32 → stop recursive loop
- Max 10 cycles hard cap

---

## 4. Scheduling & Activity Detection

### Business Hours

- Default: Monday–Friday, 08:00–18:00 (configurable)
- Respects system timezone

### Idle Detection

| Signal | Threshold |
|--------|-----------|
| CPU load (5-min avg) | < 15% |
| GPU VRAM usage | < 30% of total |
| User input (keyboard/mouse) | No activity for 30+ min |
| Homie conversation | No active session for 15+ min |

### Trigger Logic

Scheduler checks every 30 minutes:
- Is a finetune cycle pending?
- Is it non-business hours OR system idle > 30 min?
- Is GPU VRAM available?
- All yes → start pipeline

### Interrupt & Resume

- **Interrupt on:** user conversation starts, CPU > 60%, GPU VRAM pressure, business hours begin
- **Pause:** save optimizer state + current step at next checkpoint boundary
- **Resume:** when conditions clear, resume from exact checkpoint

### Stage Priority (lowest to highest)

1. Synthetic data generation (low CPU, can pause anywhere)
2. Training (GPU-intensive, pauses on any user activity)
3. Evaluation (uses inference, pauses if user needs model)
4. Merge + quantize + push (fast, runs to completion)

---

## 5. Recursive Loop Orchestration

### Cycle Lifecycle

**Cycle 0 (Bootstrap):**
- Generate 2,000 SFT + 1,000 DPO balanced across 6 domains
- Train SFT → DPO on base lfm2
- Evaluate → establish baseline
- If passes promotion gate → deploy as v1

**Cycle 1..N (Recursive):**
- Analyze previous eval → identify weakest domain
- Generate new data: 40% targeting weak domain, 60% balanced
- Increase difficulty for domains scoring > 0.8
- Train SFT → DPO **from original base lfm2** using accumulated data (cycles 0..N)
- Evaluate against current active model
- If improved > 2% → deploy, continue
- If plateau × 3 → escalate LoRA rank or stop

### Difficulty Tiers

| Tier | Description | Triggered When |
|------|-------------|---------------|
| 1 — Basic | Clear single-intent, obvious contexts | Start |
| 2 — Intermediate | Multi-part, subtle context cues, 3-step chains | Domain > 0.6 |
| 3 — Advanced | Ambiguous + conflicting context, 5+ step orchestration | Domain > 0.8 |
| 4 — Adversarial | Deliberately confusing, edge cases, compound attacks | Domain > 0.9 |

### State Persistence

```
~/.homie/finetune/
├── state.json              # Cycle number, scores, tiers, plateau counter, LoRA rank
├── datasets/
│   └── cycle-{N}/          # SFT and DPO JSONL
├── adapters/
│   └── cycle-{N}/          # LoRA checkpoints
└── evals/
    └── cycle-{N}.json      # Benchmark results
```

### Safety Rails

- Max 10 cycles hard cap
- Max 50GB disk for finetune artifacts — prune strategy: delete oldest **complete cycle** (dataset + adapter + eval), keeping the most recent 3 cycles and the current active adapter always. Merged GGUF files are ephemeral (deleted after Ollama import).
- Safety domain < 0.85 → abort + rollback
- Training only starts after 30+ min continuous idle

### Observability

- **Status query:** `FinetuneScheduler.get_status()` returns current stage, cycle number, percent complete, ETA, and last eval scores. Exposed via `/api/finetune/status` endpoint and tray menu.
- **Structured logging:** Each stage logs duration, data counts, scores, and errors to `~/.homie/finetune/logs/cycle-{N}.log`.
- **Desktop notification:** On cycle completion (pass/fail) and on pipeline convergence (plateau stop).
- **Dashboard integration:** Training status card on the Homie dashboard showing cycle progress and domain score trends.

---

## 6. Proactive Information Gathering

### Gathering Layers

| Layer | What | When | Storage |
|-------|------|------|---------|
| Boot Context | System state, service status, notifications | Startup | In-memory |
| Session Context | Current project, git activity, time patterns | Conversation start | Session state |
| Predictive Prefetch | Behavioral pattern-based anticipation | Background, every 15 min | Prefetch cache |
| Conversation Lookahead | Anticipate follow-ups from current topic | Real-time | Ephemeral |

### Predictive Prefetch

Uses `BehavioralProfiler` hourly patterns:
- Monday 9 AM → user asks about emails/calendar → prefetch unread count, meetings, tasks
- Python project opened → prefetch git log, test results, TODOs
- User mentions "meeting" → prefetch attendee list, agenda, related threads

### Training Patterns

| Pattern | Behavior |
|---------|----------|
| Volunteer relevant context | Include project status, deadlines, blockers without being asked for each |
| Anticipate follow-ups | After "send email" → also suggest updating Slack channel |
| Front-load key info | Lead with status, then details, then issues |
| Know when NOT to dump | Simple greeting → don't overwhelm with prefetched data |

---

## 7. Configuration

`FinetuneConfig` pydantic model (in `src/homie_core/finetune/config.py`) is added to `HomieConfig` as `finetune: FinetuneConfig = Field(default_factory=FinetuneConfig)`.

New config section in `homie.config.yaml`:

```yaml
finetune:
  enabled: true
  base_model: "lfm2"
  registry_name: "PyMasters/Homie"
  schedule:
    business_hours_start: 8
    business_hours_end: 18
    business_days: [0, 1, 2, 3, 4]  # Mon-Fri
    min_idle_minutes: 30
    check_interval_minutes: 30
  training:
    lora_rank: 16
    lora_alpha: 32
    sft_learning_rate: 0.0002
    dpo_learning_rate: 0.00005
    sft_epochs: 3
    dpo_epochs: 1
    batch_size: 4
    gradient_accumulation: 4
    max_seq_length: 4096
    checkpoint_steps: 50
  data:
    sft_per_cycle: 2000
    dpo_per_cycle: 1000
    min_quality_score: 4
    weak_domain_boost: 0.4
    include_curated: false  # opt-in to mix DataCurator data (user data)
  evaluation:
    promotion_threshold: 0.02
    max_regression_per_domain: 0.05
    safety_floor: 0.85
    plateau_cycles: 3
  limits:
    max_cycles: 10
    max_disk_gb: 50
    max_lora_rank: 32
```

---

## 8. New Files

```
src/homie_core/finetune/
├── __init__.py
├── pipeline.py              # RecursiveFinetuneLoop — main orchestrator
├── scheduler.py             # FinetuneScheduler — idle/business hours detection
├── synthetic/
│   ├── __init__.py
│   ├── generator.py         # SyntheticDataGenerator — template + teacher pipeline
│   ├── templates.py         # Domain scenario templates (300 total)
│   ├── context_randomizer.py # Fictional user context generation
│   └── quality_filter.py    # Teacher-based quality scoring
├── training/
│   ├── __init__.py
│   ├── qlora_trainer.py     # QLoRA training wrapper (unsloth/peft + trl)
│   ├── merge.py             # LoRA merge + GGUF quantization
│   └── checkpoint.py        # Pause/resume checkpoint management
├── evaluation/
│   ├── __init__.py
│   ├── benchmark.py         # 30-test benchmark suite
│   ├── judge.py             # Cloud fallback judge scoring
│   └── reporter.py          # Eval results tracking + plateau detection
└── config.py                # FinetuneConfig pydantic model
```

---

## 9. Dependencies

New pip packages:
- `unsloth` — QLoRA training acceleration
- `peft` — LoRA adapter management
- `trl` — SFTTrainer + DPOTrainer
- `bitsandbytes` — 4-bit quantization
- `llama-cpp-python` — GGUF conversion (or shell out to llama.cpp binary)

Already present:
- `torch` / `transformers` — base ML stack
- `requests` — cloud fallback API calls
- `pydantic` — config models
