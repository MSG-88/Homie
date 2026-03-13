# Voice Integration Design — Speech-to-Speech for Homie

**Date:** 2026-03-13
**Status:** Approved
**Approach:** Cherry-pick from HuggingFace speech-to-speech repo, keep Homie's architecture

## Overview

Add full voice interaction to Homie — the user can speak to Homie and hear spoken responses, with live transcription in the overlay. Voice feeds through the existing brain/cognitive pipeline with zero changes to the intelligence layer.

The design adopts the best components and patterns from [huggingface/speech-to-speech](https://github.com/huggingface/speech-to-speech) (queue-based threading, Silero VAD, multi-engine TTS) while preserving Homie's architecture. No separate LLM — all voice queries go through `BrainOrchestrator.process_stream()`.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        VoiceManager                                 │
│  (orchestrates modes, owns component lifecycle, config-driven)      │
│                                                                     │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐      │
│  │ AudioIn  │──▶│ SileroVAD│──▶│   STT    │──▶│  Brain   │      │
│  │ (sounddev│    │ (neural) │    │ (faster- │    │ (existing│      │
│  │  ice)    │    │          │    │  whisper) │    │ cognitive│      │
│  └──────────┘    └──────────┘    └──────────┘    │ pipeline)│      │
│                                                   └────┬─────┘      │
│                                                        │            │
│  ┌──────────┐    ┌─────────────────────────────┐       │            │
│  │ AudioOut │◀──│ TTS Engine (switchable)      │◀─────┘            │
│  │ (sounddev│    │  ├─ PiperTTS   (fast mode)  │                    │
│  │  ice)    │    │  ├─ KokoroTTS  (quality)    │                    │
│  │          │    │  └─ MeloTTS    (multilang)  │                    │
│  └──────────┘    └─────────────────────────────┘                    │
│                                                                     │
│  Threading: Queue-based producer-consumer per component             │
│  Barge-in: should_listen Event flag stops TTS + flushes queues      │
└─────────────────────────────────────────────────────────────────────┘
```

**Key principles:**

- **No separate LLM** — voice queries go through `BrainOrchestrator.process_stream()`, same as text.
- **Queue-based threading** — each component (VAD, STT, TTS, AudioOut) runs in its own thread, connected by `queue.Queue`. Adopted from HF repo's `BaseHandler` pattern.
- **Barge-in** — a `should_listen` threading.Event. When VAD detects speech during TTS playback, it sets the flag, which immediately stops audio output and flushes the TTS queue.
- **Mode-agnostic core** — pipeline components don't know about modes. `VoiceManager` handles mode logic by controlling when the pipeline starts/stops listening.

## Voice Modes & State Machine

Three modes unified under a single state machine:

```
                    ┌─────────────────────────────┐
                    │          IDLE                │
                    │  (pipeline warm, not listening)│
                    └──────┬──────────┬───────────┘
                           │          │
                   wake word detected  hotkey (ctrl+8)
                           │          │
                    ┌──────▼──────────▼───────────┐
                    │        LISTENING              │
                    │  (VAD active, waiting for     │
                    │   speech segments)             │
                    └──────────┬───────────────────┘
                               │
                        speech detected
                               │
                    ┌──────────▼───────────────────┐
                    │        RECORDING              │
                    │  (accumulating audio,          │
                    │   VAD tracking silence)        │
                    └──────────┬───────────────────┘
                               │
                     silence threshold reached
                               │
                    ┌──────────▼───────────────────┐
                    │       PROCESSING              │
                    │  (STT → Brain → response)     │
                    └──────────┬───────────────────┘
                               │
                    ┌──────────▼───────────────────┐
                    │        SPEAKING               │
                    │  (TTS playback, barge-in       │
                    │   monitoring active)           │
                    └──────┬──────────┬───────────┘
                           │          │
                    barge-in!    playback done
                           │          │
                    ┌──────▼──┐  ┌────▼────────────┐
                    │LISTENING│  │ mode check       │
                    └─────────┘  └────┬────────────┘
                                      │
                          ┌───────────┼───────────┐
                          │           │           │
                    conversational  push-to-talk  wake word
                          │           │           │
                    ┌─────▼───┐ ┌─────▼───┐ ┌────▼────┐
                    │LISTENING│ │  IDLE   │ │  IDLE   │
                    └─────────┘ └─────────┘ └─────────┘
```

### Mode behaviors

| Mode | Activation | After response | Exit |
|------|-----------|----------------|------|
| **Wake word** | "Hey Homie" detected | → IDLE (single turn) | Automatic |
| **Push-to-talk** | Hotkey held/toggled | → IDLE | Release key / toggle |
| **Conversational** | `homie voice` CLI or "let's talk" | → LISTENING (stays in loop) | "Goodbye" / silence timeout → confirmation |

### Conversational exit flow

1. Silence timeout (configurable, default 2min) or exit phrase ("goodbye", "stop", "that's all")
2. Homie asks: *"Would you like to end our conversation?"*
3. User confirms → IDLE. User says no → LISTENING resumes.
4. Timer only runs during LISTENING state — pauses during PROCESSING/SPEAKING.

### Hotkey behavior (Ctrl+8)

| Current state | Hotkey press | Result |
|---|---|---|
| IDLE, voice enabled | Ctrl+8 | Open overlay in voice mode, start LISTENING |
| IDLE, voice disabled | Ctrl+8 | Open overlay in text mode (unchanged) |
| LISTENING/RECORDING | Ctrl+8 | Cancel current recording, return to IDLE |
| SPEAKING | Ctrl+8 | Stop TTS playback, return to LISTENING |
| Conversational active | Ctrl+8 | Toggle mute/unmute |

## Component Design

### VAD: Silero VAD (replacing energy-based)

- Neural network-based via `torch.hub.load("snakers4/silero-vad")`
- Speech probability score (0.0–1.0), configurable threshold (default 0.5)
- Hysteresis: trigger at 0.5, release at 0.35 to prevent flickering
- `min_silence_duration_ms`: 300ms (conversational), 600ms (wake word) — configurable per mode
- Tiny model (~2MB), runs on CPU — no GPU contention with LLM
- Falls back to energy-based VAD if torch unavailable

### STT: faster-whisper (existing, upgraded config)

- Language auto-detection with code passthrough for TTS routing
- Model hot-switching per mode:
  - `tiny.en` for push-to-talk (speed)
  - `medium` for conversational (accuracy)
- Multilingual: `medium` and `large-v3` support Tamil, Telugu, Malayalam, French, Spanish
- English-only modes use `tiny.en` or `small.en` for speed

### TTS: Three switchable engines

| Engine | Role | Languages | Latency | When used |
|--------|------|-----------|---------|-----------|
| **Piper** (existing) | Fast mode | English + limited | ~100ms | Short replies, push-to-talk |
| **Kokoro** (new) | Quality mode | 8 languages (EN, FR, ES, +5) | ~400ms | Longer responses, conversational |
| **MeloTTS** (new) | Multilingual mode | Broad coverage including Indic | ~300ms | Tamil, Telugu, Malayalam, auto-detected |

**Auto-selection logic (default):**
1. Response < 20 words → Piper (fast)
2. Detected language not English → MeloTTS (multilingual)
3. Otherwise → Kokoro (quality)

Manual override via config or voice command ("use quality voice"). Config key: `voice.tts_mode: auto | fast | quality | multilingual`

### Audio I/O: Queue-based threading

- `AudioInThread`: reads from sounddevice `RawInputStream` (16kHz, mono, int16, 512-sample chunks), pushes to `vad_queue`
- `AudioOutThread`: reads from `playback_queue`, writes to `RawOutputStream`
- Dither strategy: low-level noise when queue empty to keep audio device responsive
- Barge-in: VAD detects speech during SPEAKING → `should_listen` event set → AudioOutThread stops, TTS queue flushed, pipeline returns to LISTENING

## Daemon & Overlay Integration

### Daemon changes (minimal)

```python
# In HomieDaemon.__init__():
if self._config.voice.enabled:
    self._voice_manager = VoiceManager(
        config=self._config.voice,
        on_query=self._on_user_query_stream,   # existing brain callback
        on_state_change=self._on_voice_state,   # for overlay updates
    )

# In HomieDaemon.start():
if self._voice_manager:
    self._voice_manager.start()

# In HomieDaemon.stop():
if self._voice_manager:
    self._voice_manager.stop()
```

The brain receives text and returns streamed tokens — it doesn't know input came from voice.

### Voice-aware prompting

When input comes from voice, a lightweight hint is injected into the system prompt:

```
User is speaking via voice. Keep responses concise and conversational.
Avoid markdown, code blocks, or visual formatting — the response will be read aloud.
```

Passed as metadata alongside the query. Text queries are unaffected.

### Overlay changes

Voice-first overlay with live transcript:

```
┌─────────────────────────────────┐
│  Listening...                   │  ← state indicator
│                                 │
│  You: "What's the weather like  │  ← live STT transcript
│        in Chennai?"             │
│                                 │
│  Homie: "It's currently 32°C   │  ← streamed response text
│   and humid in Chennai..."      │
│                                 │
│  [Type instead]    [End voice]  │  ← fallback controls
└─────────────────────────────────┘
```

- State indicator follows pipeline state
- Live transcript updates as STT produces results
- Response text streams as brain generates tokens (simultaneous with TTS)
- "Type instead" pauses voice pipeline, switches to text input
- "End voice" stops session gracefully

## Configuration

### homie.config.yaml additions

```yaml
voice:
  enabled: false
  hotkey: ctrl+8
  wake_word: "hey homie"
  mode: hybrid                      # hybrid | wake_word | push_to_talk | conversational

  stt_engine: faster-whisper
  stt_model_fast: tiny.en
  stt_model_quality: medium
  stt_language: auto                # auto | en | ta | te | ml | fr | es

  tts_mode: auto                    # auto | fast | quality | multilingual
  tts_voice_fast: piper
  tts_voice_quality: kokoro
  tts_voice_multilingual: melo

  vad_engine: silero                # silero | energy
  vad_threshold: 0.5
  vad_silence_ms: 300

  barge_in: true
  conversation_timeout: 120
  exit_phrases:
    - "goodbye"
    - "stop"
    - "that's all"

  device: auto                      # auto | cuda | cpu
  audio_sample_rate: 16000
  audio_chunk_size: 512
```

### CLI additions

```bash
homie voice                         # conversational session
homie voice --mode push-to-talk     # override mode
homie voice --tts quality           # override TTS
homie voice --lang en               # force language
homie voice status                  # show component status
homie voice enable / disable        # toggle voice
```

### New dependencies (pyproject.toml)

```toml
voice = [
    "faster-whisper>=1.0",
    "openwakeword>=0.6",
    "piper-tts>=1.2",
    "pyaudio>=0.2",
    "sounddevice>=0.4",
    "torch>=2.0",
    "torchaudio>=2.0",
    "kokoro>=0.9",
    "melo-tts>=0.1",
]
```

## File Structure

### New and modified files

```
src/homie_core/voice/
├── __init__.py                    # exports VoiceManager
├── audio_io.py                    # MODIFIED: add AudioInThread, AudioOutThread
├── stt.py                         # MODIFIED: language detection, model hot-switching
├── tts.py                         # MODIFIED: rename to PiperTTS, add TTS base class
├── tts_kokoro.py                  # NEW: Kokoro TTS engine
├── tts_melo.py                    # NEW: MeloTTS engine
├── tts_selector.py                # NEW: auto-selects TTS engine
├── vad.py                         # MODIFIED: add SileroVAD, keep energy fallback
├── vad_silero.py                  # NEW: Silero VAD implementation
├── wakeword.py                    # UNCHANGED
├── voice_pipeline.py              # MODIFIED: queue-based threading, barge-in
├── voice_manager.py               # NEW: mode orchestration, state machine
├── voice_prompts.py               # NEW: voice-aware prompt hints
└── base_handler.py                # NEW: BaseHandler queue/thread pattern

src/homie_app/
├── cli.py                         # MODIFIED: add `homie voice` command group
├── daemon.py                      # MODIFIED: instantiate VoiceManager
├── overlay.py                     # MODIFIED: voice mode panel
├── hotkey.py                      # MODIFIED: ctrl+8, mode-aware behavior

homie.config.yaml                  # MODIFIED: expanded voice section
pyproject.toml                     # MODIFIED: new voice dependencies
```

### Untouched

- `src/homie_core/brain/` — zero changes
- `src/homie_core/memory/` — unchanged
- `src/homie_core/intelligence/` — unchanged
- `src/homie_core/behavioral/` — unchanged
- All plugins, security, vault, RAG — unchanged
- All existing tests continue to pass

## Error Handling & Degradation

### Graceful degradation chain

```
Silero VAD unavailable (no torch)  → energy-based VAD fallback
Kokoro unavailable                 → Piper fallback
MeloTTS unavailable                → Piper fallback
faster-whisper fails to load       → voice disabled, text-only mode
Audio device not found             → voice disabled, text-only mode
```

`VoiceManager` probes each component at startup and builds an availability map.

### Performance safeguards

| Concern | Mitigation |
|---|---|
| GPU memory contention | STT on CPU by default. TTS uses GPU only during SPEAKING, releases after. LLM has priority. |
| Audio latency spikes | Queue depth monitoring — if queue exceeds 50 items, log warning, drop oldest chunks. |
| Barge-in race condition | `should_listen` is `threading.Event` (atomic). TTS thread checks every chunk (~32ms). |
| Wake word false positives | Two-stage: Silero VAD detects speech → STT transcribes → text-based wake word match. |
| Timeout during processing | Timer only runs in LISTENING state, pauses during PROCESSING/SPEAKING. |
| Thread cleanup on crash | `stop_event` + queue sentinel (`b"END"`) pattern. `atexit` handler as safety net. |
| Config hot-reload | Mid-session TTS swap: queue drain → swap engine → resume. |

## Out of Scope (Future Work)

- WebSocket/network audio transport
- Progressive/streaming STT (partial transcripts during speech)
- Voice cloning / custom voice training
- Multi-speaker detection
- Noise cancellation (DeepFilterNet)
- Additional STT engines (Parakeet, Paraformer)
