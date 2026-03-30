# AGENTS.md

Guide for AI coding assistants working in the SpindL codebase.

## Quick Reference

| What | Command |
|------|---------|
| Start everything (recommended) | `python scripts/dev.py` |
| Start backend (headless, no GUI) | `python scripts/launcher.py` |
| Run backend tests | `conda run -n spindl python -m pytest tests/ --tb=short -q` |
| Run frontend tests | `cd gui && npm run test:run` |
| Run specific test file | `conda run -n spindl python -m pytest tests/path/to/test_file.py -v` |
| Install backend deps | `pip install -e ".[dev]"` |
| Install frontend deps | `cd gui && npm install` |

## Project Layout

```
src/spindl/             Python backend (~34,400 lines, 135 files)
gui/src/                Next.js frontend (~32,800 lines, 157 files)
tests/                  Backend unit tests (116 test modules, ~29,000 lines)
tests_e2e/              E2E tests (Playwright, 8 modules, 5-config matrix)
scripts/                dev.py (unified launcher), launcher.py (headless), standalone GUI
config/                 spindl.yaml.example (runtime config template)
spindl-avatar/          Standalone avatar renderer (Tauri 2 + Three.js + VRM)
spindl-subtitles/       Stream subtitle overlay (Tauri 2, OBS-compositable)
characters/             User character data (gitignored)
```

## Backend Architecture

### Module Map

| Module | Purpose | Key ABCs/Classes |
|--------|---------|-----------------|
| `orchestrator/` | Central voice agent loop, config (Pydantic v2) | `VoiceAgentOrchestrator`, `OrchestratorConfig`, `OrchestratorCallbacks` |
| `core/` | State machine, event bus, context | `AudioStateMachine`, `EventBus`, `ContextManager` |
| `audio/` | Mic capture, speaker playback, VAD, RMS level emission, stream health watchdog | `AudioCapture` (includes `_watchdog_loop` for dead-stream detection + auto-restart), `AudioPlayback`, `SileroVAD`, `VADTracker` |
| `avatar/` | Emotion classification + avatar bridge | `ONNXEmotionClassifier`, `AvatarToolMoodSubscriber`, `AvatarConfig` |
| `llm/` | Prompt building, LLM dispatch, plugins | `LLMProvider` (ABC), `LLMPipeline`, `PromptBuilder`, `PromptBlock` |
| `llm/builtin/` | LLM backend implementations | `LlamaProvider`, `DeepSeekProvider`, `OpenRouterProvider` |
| `llm/providers/` | Pipeline content providers | `HistoryProvider`, `PersonaProvider`, `InputProvider`, `VoiceStateProvider`, etc. |
| `llm/plugins/` | Pre/post-processing pipeline | `PreProcessor` (ABC), `PostProcessor` (ABC), `BudgetEnforcer`, `HistoryInjector`, `ReasoningExtractor`, `SummarizationTrigger`, `CodexActivator`, `CodexCooldown`, `TTSCleanup` |
| `stt/` | Speech-to-text | `STTProvider` (ABC), `WhisperProvider`, `ParakeetProvider` |
| `tts/` | Text-to-speech | `TTSProvider` (ABC), `KokoroProvider` |
| `vision/` | Screen capture + VLM (local, cloud, unified) | `VLMProvider` (ABC), `ScreenCapture`, `VisionProvider`, `LLMVisionProvider` |
| `memory/` | ChromaDB vector store (global + per-character), RAG with composite scoring, reflection, summaries | `MemoryStore` (tiered: global + general + flashcards + summaries), `RAGInjector`, `ReflectionSystem`, `SessionSummaryGenerator`, `compute_score()` |
| `characters/` | SillyTavern V2 card models | `Character` (Pydantic), `CharacterLoader` |
| `codex/` | Lorebook/character book | `CodexActivationManager`, `CodexManager` |
| `tools/` | Function calling framework | `Tool` (ABC), `ToolExecutor`, `ToolRegistry` |
| `stimuli/` | Autonomous behavior engine (idle timer, Twitch chat) | `StimulusModule` (ABC), `StimuliEngine`, `PatienceModule`, `TwitchModule` |
| `vts/` | VTubeStudio WebSocket driver | `VTSDriver` |
| `launcher/` | Service process management | `ServiceRunner`, `HealthChecker`, `LogAggregator` |
| `gui/` | Socket.IO server, response models | Socket handlers in `server.py`, `response_models.py` (Pydantic) |
| `history/` | JSONL conversation persistence, prompt snapshots | `JSONLStore`, `SnapshotStore` |
| `config/` | YAML config loading | `config_loader.py`, `get_config_path()` |
| `utils/` | Shared utilities (paths, ring buffer) | `paths.py`, `ring_buffer.py` |

### Patterns to Follow

**Registry pattern.** All pluggable backends use registry factories. To add a new provider:
1. Implement the relevant ABC (`LLMProvider`, `STTProvider`, `TTSProvider`, `VLMProvider`, `StimulusModule`, `Tool`)
2. Register it in the corresponding registry module

**Plugin pipeline.** LLM pre/post-processing uses `PreProcessor` and `PostProcessor` ABCs. Plugins are registered on the `LLMPipeline` and run in order.

**Event bus.** Components communicate via `EventBus` (pub/sub). Event types are defined in `core/events.py`. Use `EventBus.subscribe()` / `EventBus.publish()`. Supports priority ordering and one-shot subscriptions. High-frequency events (e.g., `AUDIO_LEVEL` at 50ms intervals) flow through the same bus — EventBridge forwards them to Socket.IO for frontend visualization.

**State machine.** Agent states: `IDLE → LISTENING → USER_SPEAKING → PROCESSING → SYSTEM_SPEAKING`. Defined as an enum in `core/state_machine.py`. All transitions are thread-safe (Lock-protected). **Gotcha:** During TTS playback, the state machine re-enters `LISTENING` for barge-in (interrupt) support. Frontend components that need to know "system is speaking" should check `audioLevel > 0` instead of `state === "system_speaking"` (see `character-portrait.tsx`).

**Prompt blocks.** The prompt builder supports three modes: legacy (template string), provider (registered `ContextProvider` plugins), and block (configurable `PromptBlock` dataclasses). Block mode is default. Block definitions live in `llm/prompt_block.py`.

**Config-driven.** All settings flow from `config/spindl.yaml`. Environment variable substitution (`${ENV_VAR}`) is supported — secrets go in `.env`, not YAML. Config loading uses `python-dotenv` + a custom `resolve_env_vars()` function.

**Runtime-swappable providers.** LLM and VLM providers use a `ProviderHolder` indirection layer (`llm/provider_holder.py`). Five consumers (pipeline, tools, etc.) hold a reference to the holder, not the provider directly — swapping the inner provider updates all consumers automatically. Swaps are state-machine gated (rejected during PROCESSING). Config persistence uses section-level YAML surgery, never full-file `yaml.dump()` (which introduces PyYAML continuation lines on strings >80 chars). Socket handlers: `set_llm_provider`, `set_vlm_provider`, `set_tools_config` — all emit `_emit_health()` after their config update event.

**Thread safety.** Audio capture, playback, VAD, stimuli engine, and VTS driver all run in daemon threads. Use `SimpleQueue` for inter-thread command dispatch (see `vts/driver.py`). Use `threading.Lock` for shared state (see `core/state_machine.py`).

**Event types.** 28 `EventType` enum values in `core/events.py`:
- Speech pipeline: `TRANSCRIPTION_READY`, `RESPONSE_READY`, `TTS_STARTED`, `TTS_COMPLETED`, `TTS_INTERRUPTED`
- State: `STATE_CHANGED`
- Audio viz: `AUDIO_LEVEL` (speaker RMS), `MIC_LEVEL` (mic RMS)
- Context: `CONTEXT_UPDATED`
- Tokens: `TOKEN_USAGE`, `PROMPT_SNAPSHOT`
- Tools: `TOOL_INVOKED`, `TOOL_RESULT`
- Stimuli: `STIMULUS_FIRED`
- Avatar: `AVATAR_MOOD`, `AVATAR_TOOL_MOOD`
- Error: `PIPELINE_ERROR`

Key event data: `ResponseReadyEvent` carries `emotion`, `emotion_confidence`, `stimulus_source`. `ToolInvokedEvent`/`ToolResultEvent` carry `tool_call_id` and `iteration`.

### Prompt Composition (Block Model)

The prompt is assembled from 15 configurable blocks:

```
persona_name (0) → persona_appearance (1) → persona_personality (2) →
scenario (3) → example_dialogue (4) → modality_context (5) →
voice_state (6) → codex_context (7) → rag_context (8) →
twitch_context (9) → persona_rules (10) → modality_rules (11) →
conversation_summary (12) → recent_history (13) → closing_instruction (14)
```

Each block has: `order`, `enabled`, `section_header`, `content_wrapper`, `user_override`. Blocks can be reordered, disabled, or overridden via `spindl.yaml` under `prompt_blocks:`. The dashboard's Prompt Workshop page provides a visual editor. Deferred blocks (`codex_context`, `rag_context`, `twitch_context`, `recent_history`) have their content populated at runtime by pipeline injection.

### Memory System

Four ChromaDB collections — one global, three per character:
- `global_memories` — Cross-character, user-curated facts (manual entry only, shared across all characters)
- `{id}_general` — Per-character durable facts (manual entry + escalated entries)
- `{id}_flashcards` — Auto-generated reflection output (every N messages, configurable interval), session-scoped (only retrieved for the current session)
- `{id}_summaries` — On-demand session summaries (triggered from GUI), session-scoped

RAG injection: `RAGInjector` (a `PreProcessor`) queries relevant memories from global + per-character collections and injects them into the prompt. Retrieval uses composite scoring via `compute_score()` — four signals: relevance (L2 distance inverted), Ebbinghaus recency decay (exponential), importance rating (1–10), and log-scaled access count. Tier weights boost curated content (global 1.15×, general 1.10×, flashcards 1.0×, summaries 0.95×). Scoring weights are configurable in `MemoryConfig` (`scoring_w_relevance`, `scoring_w_recency`). `reinforce()` bumps `access_count` + `last_accessed` at injection time, not query time.

Reflection prompts are user-editable: `reflection_prompt` (extraction question with `{transcript}` placeholder), `reflection_system_message`, and `reflection_delimiter` are configurable fields in `MemoryConfig`. The parser is format-agnostic — accepts any delimiter.

Embedding: External llama.cpp server running with `--embedding` flag on a dedicated port. `EmbeddingClient` calls `/v1/embeddings` (OpenAI-compatible).

### Character Cards

Uses [SillyTavern V2 spec](https://github.com/SillyTavern/SillyTavern). Key models in `characters/models.py`:
- `Character` — name, description, personality, scenario, greetings, avatar
- `CharacterBook` / `CharacterBookEntry` — lorebook with keyword activation, cooldowns, sticky duration
- `SpindlExtensions` — voice config, TTS settings, appearance, rules, summarization prompt
- `GenerationConfig` — temperature, top_p, max_tokens, etc.

All validated via Pydantic at load time.

### Avatar System

Two components: a backend emotion pipeline and a standalone desktop renderer.

**Backend (`avatar/`):**
- `ONNXEmotionClassifier` — DistilBERT GoEmotions via ONNX Runtime (~67MB, lazy-downloaded). 28 labels → 5 buckets (happy/sad/angry/surprised/neutral) → VRM-aligned mood names (amused/melancholy/annoyed/curious). 5× neutral weighting. CPU-only, <15ms inference.
- `AvatarToolMoodSubscriber` — Subscribes to `TOOL_INVOKED` events, maps tool names to visual categories (search/execute/memory), re-emits as `AVATAR_TOOL_MOOD` events.
- `AvatarConfig` — Pydantic model: enabled, emotion_classifier mode, show_emotion_in_chat, confidence_threshold, expression_fade_delay, subtitles_enabled, subtitle_fade_delay.

**Standalone renderer (`spindl-avatar/`):**
- Tauri 2 + Three.js + @pixiv/three-vrm desktop app.
- Procedural idle animations (blink, breathe, saccade, fidget, contrapposto), cursor tracking, lipsync, 20 mood presets.
- Mixamo FBX retargeting via AnimationMixer. 5 base animation slots (`idle`, `happy`, `sad`, `angry`, `curious`) mapped from classifier moods via `MOOD_TO_BASE_SLOT` (`amused→happy`, `melancholy→sad`, `annoyed→angry`, `curious→curious`). Per-character expression composites — custom blend shape recipes keyed by mood name (e.g., `avatar_expressions.curious` on the character card), with per-emotion confidence thresholds.
- Socket.IO bridge to orchestrator. Events: state, amplitude (lipsync), TTS status, avatar_mood, avatar_tool_mood.
- Transparent window mode (alpha stash + force-opaque materials), 7-light rig + bloom + color grading.

**Event flow:**
```
LLM response text → ONNXEmotionClassifier.classify()
    → AvatarMoodEvent (mood, confidence)
    → Socket.IO → spindl-avatar renderer
    → expression composites + animation crossfade
```

### Subtitle Overlay

Standalone Tauri 2 app (`spindl-subtitles/`). Connects to the orchestrator via Socket.IO, listens for `response_chunk` and `tts_playback` events. Duration-synced typewriter reveal with sentence boundary cropping. Chroma key background (black/green/magenta) via right-click context menu.

**Process management:** Spawned/killed by `GUIServer._subtitle_spawn()` / `_subtitle_kill()`. Auto-spawns on startup when `subtitles_enabled: true` in config. Uses `kill_process_tree` (psutil) for clean shutdown.

### Process Shutdown Chain

`dev.py` manages the full process tree. On Ctrl+C or dashboard Shutdown button:

1. **Unix:** SIGINT → backend `finally` block → `shutdown_services()` (orchestrator, avatar, subtitles, services) → uvicorn exits → `dev.py` detects exit → tree-kills frontend
2. **Windows:** `dev.py` uses `kill_process_tree` (psutil, `recursive=True`) to walk and terminate the entire process tree — backend, all services, avatar, subtitles, frontend

`shutdown_services()` and `_shutdown_backend_async()` both call `_avatar_kill()` and `_subtitle_kill()` as defense-in-depth.

## Frontend Architecture

**Stack:** Next.js 16 + React 19 + TypeScript + Tailwind CSS v4 + Zustand + Socket.IO client + Radix UI

### Pages

| Route | Purpose |
|-------|---------|
| `/` | Dashboard — hero character portrait (audio-reactive glow), transcription, responses, tool calls |
| `/launcher` | Service configuration wizard (LLM, STT, TTS, VLM, embedding) |
| `/prompt` | Prompt Workshop — block editor, token breakdown, injection wrappers |
| `/characters` | Character manager — CRUD, avatar cropping, import/export |
| `/memories` | Memory curation — general, flashcards, summaries, search |
| `/codex` | Knowledge base — global + per-character lorebook entries |
| `/sessions` | Conversation history viewer |
| `/settings` | Configuration — personas, VAD, pipeline settings, Twitch credentials, avatar/subtitle toggles |

### State Management

11 Zustand stores in `gui/src/lib/stores/`:

| Store | Scope |
|-------|-------|
| `agent-store` | Pipeline state, transcription, responses, health, tools, audio/mic level, shutdown |
| `chat-store` | Chat message history, hydration, metadata (emotion, codex, memories, reasoning) |
| `connection-store` | Socket.IO connection status |
| `launcher-store` | Service config, launch progress, model lists, validation |
| `character-store` | Character CRUD, avatar data, VRM binding, import/export |
| `codex-store` | Global + per-character lorebook entries |
| `memory-store` | Memory collections (general/flashcards/summaries), search, CRUD |
| `prompt-store` | Current prompt snapshot (messages + token breakdown) |
| `block-editor-store` | Block config editing state, drag-to-reorder, field overrides |
| `settings-store` | VAD, pipeline, memory, generation params, stimuli, tools, LLM/VLM runtime, avatar config (emotion classifier, fade delay, subtitles, connection status) |
| `session-store` | Conversation history, session resume/delete/summarize |

**Data flow:** `SocketProvider` (root context) listens to 100+ socket events → calls Zustand store setters → components re-render via selectors.

**Zustand gotcha:** Selectors that return new objects via spread (`{ ...state.foo }`) must NOT be passed to `useStore()` — causes infinite re-render. Call `selector(store)` directly instead.

### Backend Communication

- **Socket.IO** (port 8765) — Real-time bidirectional. Agent state, transcription, responses, config updates, memory CRUD, stimuli events.
- **REST API** (`/api/`) — File operations. Character CRUD, avatar upload, codex entries, config read/write. Next.js API routes → filesystem.

Event types are defined in `gui/src/types/events.ts`. Type-safe client/server contracts.

### Styling

- Tailwind CSS v4 with PostCSS
- Dark theme hardcoded (`html dark` class in root layout)
- Radix UI primitives for accessible components
- **Never** use `${variable}` interpolation for Tailwind classes — JIT purge can't detect them. Use complete class strings in config objects.

## Testing

### Backend (pytest)

```bash
# All tests
conda run -n spindl python -m pytest tests/ --tb=short -q

# Specific module
conda run -n spindl python -m pytest tests/memory/ -v

# Skip slow/cloud tests
conda run -n spindl python -m pytest tests/ -m "not slow and not cloud" --tb=short -q
```

**Important:** Always specify `tests/` directory explicitly. Running bare `python -m pytest` from project root can hang (discovers files outside tests/).

`pytest-timeout` is NOT installed. If a test might hang, run it in the background.

Test markers: `@pytest.mark.slow`, `@pytest.mark.vision`, `@pytest.mark.cloud`.

### Frontend (Vitest)

```bash
cd gui
npm run test:run          # Single run
npm run test              # Watch mode
npm run test:coverage     # Coverage report
```

jsdom environment. Global `fetch` mock in `gui/src/test/setup.ts`. Socket.IO mock in `gui/src/test/mocks/socket.ts`.

22 test files across stores, components, API routes, and schemas:
- **Stores (9):** agent, character, chat, connection, memory, prompt, session, settings, vts
- **Components (6):** block-detail, block-list, prompt-viewer, session-list, session-viewer, llm-config, vtubestudio-card
- **API routes (2):** fetch-models, write-config
- **Schemas (1):** config-schemas (28 test suites covering all Zod schemas)
- **Pages (2):** codex, memories
- **Test infrastructure:** `gui/src/test/setup.ts` (cleanup, mocks), `gui/src/test/mocks/socket.ts` (createMockSocket factory with `_simulateEvent()` helper), `gui/src/test/utils.tsx` (render helpers)

### E2E (Playwright)

```bash
conda run -n spindl python -m pytest tests_e2e/ -v
```

Requires running services. E2E config fixtures at `tests_e2e/fixtures/config/` (5 named configs covering local/cloud LLM × VLM combos + unified mode: `spindl_e2e_local_unified`, `spindl_e2e_local_local`, `spindl_e2e_local_cloud`, `spindl_e2e_cloud_local`, `spindl_e2e_cloud_cloud`).

## Windows-Specific Gotchas

- **Bash paths:** Always use forward slashes (`c:/Users/...`) in shell commands. A trailing `\` before a closing `"` is interpreted as an escaped quote.
- **cmd.exe quoting:** npm `.cmd` shims route through `cmd.exe /c` which mangles nested quotes. For subprocess calls to npm CLI tools, bypass the `.cmd` wrapper and call `node path/to/cli.js` directly.
- **shell=True for native services:** `service_runner.py` uses `shell=True` on Windows for commands with quoted path segments. This invokes `cmd /S /c` which correctly preserves inner quotes.
- **mss thread safety:** Create a fresh `mss.mss()` instance per capture on Windows (thread safety requirement).

## Configuration Reference

Primary config: `config/spindl.yaml` (copy from `spindl.yaml.example`).

Key sections:
- `audio:` — capture_rate (16000), chunk_size (512)
- `character:` — default character, character directory
- `services:` — per-service config (enabled, platform, command, health_check, depends_on)
- `providers:` — LLM, STT, TTS, VLM provider settings
- `memory:` — enabled, relevance_threshold, top_k, reflection_interval, reflection_prompt, reflection_system_message, reflection_delimiter, scoring_w_relevance, scoring_w_recency
- `stimuli:` — enabled, patience config (idle threshold, message count)
- `avatar:` — enabled, emotion_classifier, show_emotion_in_chat, confidence_threshold, expression_fade_delay
- `vtubestudio:` — enabled, host, port, plugin name
- `prompt_blocks:` — per-block overrides (order, enabled, content)

Secrets use `${ENV_VAR}` syntax. Put API keys in `.env` (loaded via `python-dotenv`).

**YAML boolean gotcha:** YAML 1.1 parses bare `on`, `off`, `yes`, `no` as booleans. In `extra_args` lists, always quote these values:
```yaml
extra_args:
  - -fa
  - "on"    # without quotes, YAML parses this as boolean True
```
The backend coerces `True`→`"on"` and `False`→`"off"` as a safety net (NANO-080), but quoting is the correct fix.
