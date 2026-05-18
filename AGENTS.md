# AGENTS.md

Guide for AI coding assistants working in the SpindL codebase.

## Git Workflow

**Branch strategy:** Feature branches + pull requests. Never push directly to `main`.

```
main (protected, always green CI)
 └── NANO-112-stt-tts-disable-toggle        ← feature branch
 └── NANO-111-streaming-tts-pipeline        ← feature branch
 └── NANO-110-addressing-others-stream-deck ← feature branch
 └── NANO-082-e2e-launch-matrix             ← feature branch
```

**Flow:**
1. `git checkout main && git pull`
2. `git checkout -b NANO-XXX-short-description`
3. Commit as you go (small, logical commits)
4. `git push -u origin NANO-XXX-short-description`
5. `gh pr create` — PR against `main` with summary + test plan
6. CI must pass (pytest + Vitest + `next build`)
7. Merge (squash or regular — either is fine)
8. Delete the branch after merge

**Branch naming:** `NANO-XXX-short-description` — matches the ticket system. Examples:
- `NANO-108-repetition-params`
- `NANO-082-e2e-launch-matrix`
- `MTX-002-group-chat`

**Commit messages:** Concise, imperative. Include ticket number when relevant.
- `NANO-108: wire repeat_penalty through pipeline and provider`
- `NANO-108: add repetition control sliders to dashboard`
- `fix: socket handler missing validation for presence_penalty`

**PR description format:**
```
## Summary
- 1-3 bullet points

## Test plan
- [ ] Backend: pytest passes
- [ ] Frontend: Vitest + next build pass
- [ ] CI: green check on PR
```

**CI pipeline:** GitHub Actions (`.github/workflows/ci.yml`). Two parallel jobs on `ubuntu-latest`:
- Backend (Python 3.12, `pip install -e ".[dev]"`, 15 min timeout): `python -m pytest tests/ --tb=short -q -m "not cloud and not vision and not slow and not hardware" --junitxml=test-results.xml`
- Frontend (Node 22, `npm ci` in `gui/`, 10 min timeout): `npm run test:run` then `npx next build`

CI runs bare `python` inside `actions/setup-python@v5`. Local dev uses `conda run -n spindl python` — the marker filter and command are otherwise identical. CI excludes four marker groups (`cloud`, `vision`, `slow`, `hardware`); locally you can drop the filter to run everything.

**Not in CI:** `tests_e2e/` (Playwright) requires running services and is invoked manually. The Tauri Rust crates (`spindl-avatar`, `spindl-subtitles`, `spindl-stream-deck`) are not built in CI either.

Runs on every push to `main` and every PR targeting `main`.

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
src/spindl/             Python backend (~45,500 lines, 162 files)
gui/src/                Next.js frontend (~38,200 lines, 166 files)
tests/                  Backend unit tests (106 test modules, ~33,500 lines)
tests_e2e/              E2E tests (Playwright, 8 modules, 5-config matrix)
scripts/                dev.py (unified launcher), launcher.py (headless), standalone GUI
config/                 spindl.yaml.example (runtime config template)
spindl-avatar/          Standalone avatar renderer (Tauri 2 + Three.js + VRM)
spindl-subtitles/       Stream subtitle overlay (Tauri 2, OBS-compositable)
spindl-stream-deck/     Addressing-others button overlay (Tauri 2, hold-to-activate)
Cargo.toml              Workspace root — shared target/ across all 3 Tauri apps
characters/             User character data (gitignored)
```

## Backend Architecture

### Module Map

| Module | Purpose | Key ABCs/Classes |
|--------|---------|-----------------|
| `orchestrator/` | Central voice agent loop, config (Pydantic v2), service capabilities registry | `VoiceAgentOrchestrator`, `OrchestratorConfig`, `OrchestratorCallbacks`, `ServiceCapabilities` |
| `core/` | State machine, event bus, context | `AudioStateMachine`, `EventBus`, `ContextManager` |
| `audio/` | Mic capture, speaker playback, VAD, RMS level emission, stream health watchdog | `AudioCapture` (includes `_watchdog_loop` for dead-stream detection + auto-restart), `AudioPlayback`, `SileroVAD`, `VADTracker` |
| `avatar/` | Emotion classification + avatar bridge | `ONNXEmotionClassifier`, `AvatarToolMoodSubscriber`, `AvatarConfig` |
| `llm/` | Prompt building, LLM dispatch, plugins, sentence segmentation | `LLMProvider` (ABC), `LLMPipeline`, `PromptBuilder`, `PromptBlock`, `SentenceSegmenter`, `LlamaClient` |
| `llm/builtin/` | LLM backend implementations | `LlamaProvider`, `DeepSeekProvider`, `OpenRouterProvider` |
| `llm/providers/` | Pipeline content providers | `HistoryProvider`, `PersonaProvider`, `InputProvider`, `VoiceStateProvider`, etc. |
| `llm/plugins/` | Pre/post-processing pipeline | `PreProcessor` (ABC), `PostProcessor` (ABC), `BudgetEnforcer`, `HistoryInjector`, `TwitchHistoryInjector` (NANO-115), `ConversationHistoryManager` (with empty-assistant guard, NANO-115), `ReasoningExtractor`, `SummarizationTrigger`, `CodexActivatorPlugin` (keyword activation + `tool_choice_override`), `CodexCooldown`, `TTSCleanupPlugin` (dual-output: raw→chat, cleaned→TTS/subtitles/history) |
| `stt/` | Speech-to-text | `STTProvider` (ABC), `WhisperSTTProvider`, `ParakeetSTTProvider` |
| `stt/builtin/` | STT backend implementations | `whisper/` (provider + client), `parakeet/` (provider + client) |
| `tts/` | Text-to-speech | `TTSProvider` (ABC), `KokoroTTSProvider`, `Qwen3TTSProvider` |
| `tts/builtin/` | TTS backend implementations | `kokoro/` (provider + client + server), `qwen3/` (provider) |
| `vision/` | Screen capture + VLM (local, cloud, unified) | `VLMProvider` (ABC), `ScreenCapture`, `VisionProvider`, `LLMVisionProvider` |
| `vision/builtin/` | VLM backend implementations | `llama/` (provider + gemma3 model handler), `llm/` (unified LLM-as-VLM), `openai/` (cloud OpenAI-compatible) |
| `memory/` | ChromaDB vector store (global + per-character), RAG with composite scoring, reflection, summaries | `MemoryStore`, `RAGInjector`, `ReflectionSystem`, `ReflectionMonitor`, `SessionSummaryGenerator`, `CurationClient`, `EmbeddingClient`, `compute_score()` |
| `characters/` | SillyTavern V2 card models, import/export | `Character` (Pydantic), `CharacterLoader`, `CharacterImporter`, `CharacterExporter` |
| `codex/` | Lorebook/character book | `CodexActivationManager`, `CodexManager`, `CodexEntry` (models) |
| `tools/` | Function calling framework + tool executor | `Tool` (ABC), `ToolExecutor` (`TOOL_PATH_TIMEOUT`, `forced_tool_choice` break), `ToolRegistry` |
| `tools/builtin/` | Built-in tool implementations | `screen_vision/` (VLM screenshot describe), `game_state_query/` (query game bridge state), `invoke_hack/` (send hack_walk command to game bridge) |
| `stimuli/` | Autonomous behavior engine (weighted lottery, idle, Twitch, game state, addressing-others) | `StimulusModule` (ABC), `StimuliEngine` (`_select_stimulus` with weighted lottery), `PatienceModule`, `TwitchModule` (`on_message_accepted`), `TwitchSelector`, `WeightedRotator` (decay + recovery) |
| `stimuli/game_state/` | Game bridge consumer — TCP event stream, dialogue buffering, summarization | `GameStateModule` (TCP `asyncio.open_connection`, `send_command_and_wait`), `DialogueBuffer`, `DialogueStore`, `DialogueSummarizer`, `EventValidator` |
| `personas/` | Persona loading | `PersonaLoader` |
| `vts/` | VTubeStudio WebSocket driver | `VTSDriver` |
| `launcher/` | Service process management | `ServiceRunner`, `HealthChecker`, `LogAggregator` |
| `gui/` | Socket.IO server, response models | `server.py` (core: connect, lifecycle, emit API), `server_memory.py`, `server_sessions.py`, `server_config.py`, `server_providers.py`, `server_stimuli.py`, `server_vts.py`, `server_avatar.py` (domain handlers — NANO-113), `response_models.py` (Pydantic) |
| `history/` | JSONL conversation persistence, prompt snapshots | `JSONLStore`, `SnapshotStore` |
| `config/` | YAML config loading | `config_loader.py`, `get_config_path()` |
| `utils/` | Shared utilities | `paths.py`, `ring_buffer.py`, `tokens.py` |

### Patterns to Follow

**Registry pattern.** All pluggable backends use registry factories. To add a new provider:
1. Implement the relevant ABC (`LLMProvider`, `STTProvider`, `TTSProvider`, `VLMProvider`, `StimulusModule`, `Tool`)
2. Register it in the corresponding registry module

**Plugin pipeline.** LLM pre/post-processing uses `PreProcessor` and `PostProcessor` ABCs. Plugins are registered on the `LLMPipeline` and run in order. `TTSCleanupPlugin` is a dual-output `PostProcessor` — it stashes TTS-safe text in `context.metadata["tts_text"]` and returns the original response unchanged. Three consumers get different text: chat display (raw), TTS+subtitles (cleaned), conversation history (cleaned — steers LLM away from generating RP prose). The cleaned text is also stored as `content` in JSONL history, with the raw response preserved as `display_content`. **Streaming gotcha:** When the LLM streams (`generate_stream()`), post-processors run *after* the full response is collected — TTS delivery happens sentence-by-sentence during streaming, but history injection, codex activation, and memory reinforcement are deferred until the stream completes.

**Event bus.** Components communicate via `EventBus` (pub/sub). Event types are defined in `core/events.py`. Use `EventBus.subscribe()` / `EventBus.publish()`. Supports priority ordering and one-shot subscriptions. High-frequency events (e.g., `AUDIO_LEVEL` at 50ms intervals) flow through the same bus — EventBridge forwards them to Socket.IO for frontend visualization.

**State machine.** Agent states: `IDLE → LISTENING → USER_SPEAKING → PROCESSING → SYSTEM_SPEAKING`. Defined as an enum in `core/state_machine.py`. All transitions are thread-safe (Lock-protected). **Gotcha:** During TTS playback, the state machine re-enters `LISTENING` for barge-in (interrupt) support. Frontend components that need to know "system is speaking" should check `audioLevel > 0` instead of `state === "system_speaking"` (see `character-portrait.tsx`). **TTS-disabled gotcha (NANO-112):** When TTS is off, there's no playback to trigger `_on_playback_complete`, so the state machine would get stuck in `PROCESSING`. The `_on_tts_skipped` callback handles this with a direct `PROCESSING → LISTENING` transition — do NOT use `finish_system_speaking()` (that only transitions from `SYSTEM_SPEAKING`). **Barge-in truncation (NANO-111):** On interrupt, only the sentences that were actually delivered via TTS are stored in conversation history (`_delivered_sentences` list). The full LLM response is discarded. This prevents phantom context from unheard text leaking into subsequent turns.

**Prompt blocks.** The prompt builder supports three modes: legacy (template string), provider (registered `ContextProvider` plugins), and block (configurable `PromptBlock` dataclasses). Block mode is default. Block definitions live in `llm/prompt_block.py`.

**Config-driven.** All settings flow from `config/spindl.yaml`. Environment variable substitution (`${ENV_VAR}`) is supported — secrets go in `.env`, not YAML. Config loading uses `python-dotenv` + a custom `resolve_env_vars()` function.

**Runtime-swappable providers.** LLM and VLM providers use a `ProviderHolder` indirection layer (`llm/provider_holder.py`). Five consumers (pipeline, tools, etc.) hold a reference to the holder, not the provider directly — swapping the inner provider updates all consumers automatically. Swaps are state-machine gated (rejected during PROCESSING). Config persistence uses section-level YAML surgery, never full-file `yaml.dump()` (which introduces PyYAML continuation lines on strings >80 chars). Socket handlers: `set_llm_provider`, `set_vlm_provider`, `set_tools_config` — all emit `_emit_health()` after their config update event.

**GUI handler modules (NANO-113).** Socket.IO handlers are split by domain into `gui/server_*.py` modules. Each module exports a `register_*_handlers(server)` function that receives the `GUIServer` instance and registers closures on `server.sio`. To add a new handler: find the right domain module, add a `@sio.event` closure inside the registration function, reference `server.` (not `self.`). Helper functions that only serve one domain live as standalone functions in that module. The core `server.py` retains lifecycle handlers (connect, disconnect, launch, shutdown) and all `emit_*` public API methods consumed by `bridge.py`.

**Thread safety.** Audio capture, playback, VAD, stimuli engine, and VTS driver all run in daemon threads. Use `SimpleQueue` for inter-thread command dispatch (see `vts/driver.py`). Use `threading.Lock` for shared state (see `core/state_machine.py`).

**Event types.** 22 `EventType` enum values in `core/events.py`:
- Speech pipeline: `TRANSCRIPTION_READY`, `RESPONSE_READY`, `TTS_STARTED`, `TTS_COMPLETED`, `TTS_INTERRUPTED`
- State: `STATE_CHANGED`
- Audio viz: `AUDIO_LEVEL` (speaker RMS), `MIC_LEVEL` (mic RMS)
- Context: `CONTEXT_UPDATED`
- Tokens: `TOKEN_USAGE`, `PROMPT_SNAPSHOT`
- Tools: `TOOL_INVOKED`, `TOOL_RESULT`
- Stimuli: `STIMULUS_FIRED`
- Avatar: `AVATAR_MOOD`, `AVATAR_TOOL_MOOD`
- Streaming (NANO-111): `LLM_CHUNK` (sentence-level, carries text + index), `LLM_TOKEN` (token-level raw text for dashboard display), `BARGE_IN_TRUNCATED` (response truncated to delivered sentences)
- Twitch overlay (NANO-131/132): `TWITCH_MESSAGE_APPROVED` (selection pass picked a chat message for overlay), `TWITCH_FOLLOW_EVENT` (channel follow via EventSub)
- Error: `PIPELINE_ERROR`

Key event data: `ResponseReadyEvent` carries `emotion`, `emotion_confidence`, `stimulus_source`, `tts_text`, and `chunks` (list of `{text}` per sentence — `None` for blocking path). Emotion is single-per-response (classified on full text). `LLMChunkEvent` carries `text`, `index`, `is_final`. `ToolInvokedEvent`/`ToolResultEvent` carry `tool_call_id` and `iteration`.

### Streaming TTS Pipeline (NANO-111)

The voice response path streams LLM output and delivers TTS sentence-by-sentence in parallel, rather than waiting for the full response before synthesizing audio.

**Flow:**
```
LLM generate_stream() → token stream
    → SentenceSegmenter (stateless regex, handles abbreviations/ellipsis/reasoning tags)
    → sentence boundary detected
    → _parallel_tts_delivery() thread:
        per sentence (3-thread semaphore cap):
            TTSCleanupPlugin.clean(sentence) → TTS-safe text
            TTSProvider.synthesize(cleaned) → audio bytes
            LLMChunkEvent emitted (sentence text)
        ordered audio concatenation → play_streaming() (starts on first chunk)
    → stream completes:
        ONNXEmotionClassifier.classify(full_response) → single mood
        AvatarMoodEvent emitted (one per response)
        post-processors run (history, codex, memory)
    → ResponseReadyEvent with chunks[] metadata + emotion
```

**Single emotion per response (NANO-111, Session 608):** Per-sentence emotion classification was implemented then reverted — rapid-fire `AvatarMoodEvent`s corrupted THREE.js `crossFadeTo()` chains in the avatar mixer (orphaned action weights, visual stuttering). The fix was architectural: one classify call on the full response text, one `AvatarMoodEvent` per response. `LLMChunkEvent` carries text only (no emotion fields).

**Key classes:**
- `SentenceSegmenter` (`llm/sentence_segmenter.py`) — stateless regex boundary detector on token stream. Handles abbreviations (Mr., Dr., etc.), ellipsis, `<think>` reasoning tags, numbered lists.
- `_parallel_tts_delivery()` (`orchestrator/callbacks.py`) — spawns TTS synthesis per sentence with `ThreadPoolExecutor` (max 3 workers). Results are ordered by index regardless of completion order.
- `_delivered_sentences` (`orchestrator/callbacks.py`) — tracks which sentences have been played back. On barge-in, only delivered sentences are stored in conversation history.

**Text-only mode (TTS disabled, NANO-112):** When TTS is off, `_emit_text_only_chunks()` segments the response into sentences without synthesis. Chunks are delivered via `fallback_chunks` in the `response` event (NOT via `LLMChunkEvent` — those are tied to the streaming playback lifecycle and cause duplicate parent bubbles without it).

### STT/TTS Disable Flow (NANO-112)

STT and TTS are independently toggleable from the launcher GUI. The disable chain spans four layers:

| Layer | What happens when disabled |
|-------|---------------------------|
| **Launcher GUI** | Switch toggle in STT/TTS section headers. Card content collapses. `sttEnabled`/`ttsEnabled` in POST body. |
| **write-config route** | Writes `enabled: true/false` to `launcher.services.stt.enabled` / `launcher.services.tts.enabled` in YAML. (Previously hardcoded `true` — this was the original bug.) |
| **Pydantic config** | `STTConfig.enabled` / `TTSConfig.enabled` propagated from `launcher.services` in `OrchestratorConfig._from_dict()`. |
| **Voice agent** | Provider initialization skipped. Audio capture not started (STT). Playback config skipped (TTS). State machine speech callback not wired (STT). `_on_tts_skipped` handles `PROCESSING → LISTENING` transition (TTS). |

**Health check three-way status:** `HealthStatusEvent.stt` and `.tts` are `boolean | "disabled"`. Backend returns `"disabled"` string (not `False`) for intentionally off services — distinguishes "off by choice" from "down by failure." Frontend badges: `"disabled"` → secondary/"OFF", `true` → default/"OK", `false` → destructive/"DOWN".

**Downstream effects when disabled:**
- STT off: mic toggle disabled on dashboard, MIC badge hidden, Stream Deck toggle disabled (no point without voice input), `launcher-store.selectIsFormComplete` skips STT validation
- TTS off: character editor TTS fields show "TTS disabled — field ignored" badge, text-only chunk emission via `fallback_chunks`

### Twitch Audience Memory + Source Labeling (NANO-115)

Persistent audience transcript, structural source tags on every user-role turn, and an explicit splice/flatten history-mode control.

**Audience transcript (`llm/plugins/twitch_history.py`):**
- `TwitchTranscriptManager` — per-stream JSONL sibling file `conversations/{persona}_{timestamp}.twitch.jsonl`. LRU cache (100 entries). Schema: `{turn_id, role: "audience"|"assistant", username, text, timestamp, channel, responding_to}`. Internal-only — never sent to any LLM API.
- `TwitchHistoryInjector` (`PreProcessor`) — renders the `[AUDIENCE_CHAT]` block (order 9) with chronological + pinned messages, char-cap truncated, self-replies interleaved as `[Spindle]: ...`. Collapses to nothing when transcript is empty.
- **Dual-write:** `TwitchModule.on_message_accepted` persists every accepted viewer message; `_on_twitch_response` in `orchestrator/callbacks.py` writes Spindle's reply to both the conversation JSONL and the audience transcript with `responding_to: [usernames]`.
- **Sliders:** `twitch_audience_window` (25–300, default 25 messages) and `twitch_audience_char_cap` (50–500, default 150 chars) — live-swappable from the dashboard Twitch Chat card. Truncated lines get a `...` suffix.

**Twitch fresh batch as user-role message:**
- The fresh batch *is* the user message. Directive shape: bold markdown line on top, fenced ```chat``` block with `[mmddyyyy-hh-mm-ss:ms]` timestamps per line. Timestamps captured from `ChatMessage.sent_timestamp` on the `TwitchMessage` dataclass.
- `[TWITCH_CONTEXT]` slot, `_inject_twitch_content()`, and the `twitch_content` metadata field were retired. The `audience_chat` block at order 9 covers rolling memory; the user-role payload covers fresh stimulus.

**Source labeling tags:**
- `tag_user_input(text, input_modality, stimulus_source)` in `orchestrator/callbacks.py` is the single chokepoint. Applied at three call sites: voice `run()`, voice `run_stream()`, and `process_text_input`.
- Final taxonomy: `[Message Type - Voice | Direct Keyboard | Twitch Chat | Stimuli]`. ASCII hyphen (JSONL/YAML-safe). Tags are part of message content — they persist verbatim into JSONL and into the `### Conversation` block of subsequent prompts. Old sessions without tags load fine (purely additive).
- `MODALITY_CONTEXT` in `prompt_library.py` documents the tag grammar once and instructs the model to read the tag when asked about a past message's source. Per-turn modality strings retired — they were deadweight (never persisted into history).

**Empty assistant response guard:**
- `ConversationHistoryManager.store_turn` in `llm/plugins/conversation_history.py` skips the assistant turn when computed `history_content` is empty or whitespace-only (R1 timeout class of failure). User turn still written. `_next_turn_id` increments by 1 (not 2). `logger.warning` emitted.

**History mode (`force_role_history`):**
- Two values: `splice` (role-array history outside `[system]:`) and `flatten` (history embedded in system prompt). **Default is `flatten`.** The legacy `auto` value was removed entirely — it always behaved as flatten for every cloud provider (`supports_role_history=False` is the base) and only ever splice for local llama.cpp. Legacy `auto` in YAML or socket payloads is silently coerced to `flatten` with an info log.
- **Pre-launch control (Launcher page):** `historyMode` is a top-level field on `LauncherConfigSchema`. Toggle writes through `/api/launcher/write-config` POST → Node `fs` writes `llm.force_role_history` to `config/spindl.yaml`. Pre-launch toggles never go through the socket.
- **Runtime control (Dashboard page):** History Mode segmented toggle in Generation Parameters card emits `set_generation_params` over Socket.IO. The handler is gated on `server._orchestrator is not None` — only safe to use post-launch.
- **Pipeline override:** `Pipeline._force_role_history` is set from `voice_agent.py` on pipeline creation. `_stash_provider_capabilities` honors it: `splice` forces `provider_supports_role_history=True`, `flatten` forces `False`.

**Gotchas:**
- **First-turn diagnostic trap.** Splice and flatten produce structurally identical first-turn prompts (no history yet to splice). Verification requires at least a second turn to see the role-array shape outside the `[system]:` block.
- **Toggle ≠ slider.** Discrete toggles must emit immediately. The Launcher history toggle was briefly debounced like a slider — debounce cancelled emits on page unmount and dropped clicks. Every toggle/button in the codebase emits immediately; only sliders debounce.
- **`.twitch.jsonl` filtering.** Session viewer + session discovery (`emit_sessions()`, `get_latest_session()`, `.last_session` marker) all explicitly exclude `.twitch.` files. The audience transcript uses `"text"` not `"content"` and `role: "audience"` — would crash session-loading code paths if treated as a regular conversation file.
- **Pre-existing YAML stickiness.** Code defaults only apply on fresh config. When the C.3 Twitch directive shape was introduced, an existing `config/spindl.yaml` with the old `twitch.prompt_template` value silently kept the old template. For default-template migrations, consider a one-time pass that detects the old default and overwrites it on load.

### Prompt Composition (Block Model)

The prompt is assembled from 15 configurable blocks:

```
persona_name (0) → persona_appearance (1) → persona_personality (2) →
scenario (3) → example_dialogue (4) → modality_context (5) →
voice_state (6) → codex_context (7) → rag_context (8) →
audience_chat (9) → persona_rules (10) → modality_rules (11) →
conversation_summary (12) → recent_history (13) → closing_instruction (14)
```

Each block has: `order`, `enabled`, `section_header`, `content_wrapper`, `user_override`. Blocks can be reordered, disabled, or overridden via `spindl.yaml` under `prompt_blocks:`. The dashboard's Prompt Workshop page provides a visual editor. Deferred blocks (`codex_context`, `rag_context`, `audience_chat`, `recent_history`) have their content populated at runtime by pipeline injection.

**NANO-115:** `audience_chat` (order 9) is the rolling Twitch audience transcript. The retired `twitch_context` slot — formerly a fresh-batch system block — was eliminated when fresh Twitch batches became the actual user-role message (see "Twitch Audience Memory" below).

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
- `GenerationConfig` — temperature, top_p, max_tokens, repeat_penalty, repeat_last_n, frequency_penalty, presence_penalty

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

**Event flow (NANO-111):**
```
LLM response completes → ONNXEmotionClassifier.classify(full_response)
    → AvatarMoodEvent (mood, confidence) — one per response
    → Socket.IO → spindl-avatar renderer
    → expression composites + animation crossfade
```
Classification is one-per-response, not per-sentence. Per-sentence was implemented then reverted (Session 608) — rapid-fire mood events corrupted THREE.js crossfade chains. The single-emotion approach eliminates orphaned action weights in the animation mixer.

### Subtitle Overlay

Standalone Tauri 2 app (`spindl-subtitles/`). Connects to the orchestrator via Socket.IO, listens for `response` and `tts_status` events. Prefers `tts_text` (formatting-stripped) over raw `text` for display (NANO-109). Duration-synced typewriter reveal with sentence boundary cropping. Chroma key background (black/green/magenta) via right-click context menu.

**Process management:** Spawned/killed by `GUIServer._subtitle_spawn()` / `_subtitle_kill()`. Auto-spawns on startup when `subtitles_enabled: true` in config. Uses `kill_process_tree` (psutil) for clean shutdown.

### Stream Deck Overlay

Standalone Tauri 2 app (`spindl-stream-deck/`). Dynamic button grid — one hold-to-activate button per addressing-others context. Connects to the orchestrator via Socket.IO on port 8765. Emits `addressing_others_start { context_id }` on hold, `addressing_others_stop` on release. Receives `stimuli_config_updated` for button grid rebuilds, `addressing_others_state` for multi-client sync, and `config_loaded` for initial hydration.

**Addressing-others pipeline:** `set_addressing_others()` stops TTS + suppresses voice pipeline + pauses stimuli. `clear_addressing_others()` resolves a prompt from the active context config and stores it as a one-shot. Next pipeline call picks up the prompt via `BuildContext.addressing_others_prompt` → `ModalityContextProvider` appends it to the `### Context` block. One-shot is consumed after use.

**Cargo workspace:** Root `Cargo.toml` declares all three Tauri apps as workspace members. Shared `target/` directory (~6.5GB total). `cargo build -p spindl-stream-deck --release` builds only the requested crate, reuses cached shared deps.

**Tauri 2 capabilities:** Each Tauri app needs a `src-tauri/capabilities/default.json` granting window API permissions (e.g., `core:window:allow-set-size`). Without it, JS calls to `win.setSize()` are silently denied.

**First-time install:** Settings page detects missing binaries via `check_tauri_install`. Shows Install button with live crate-by-crate progress. Builds all three apps sequentially — first app compiles shared deps, subsequent apps reuse cache. Toggles disabled until install completes.

**Process management:** Spawned/killed by `GUIServer._stream_deck_spawn()` / `_stream_deck_kill()`. Auto-spawns on startup when `stream_deck_enabled: true` in config. Runs as compiled release binary (no Vite dev server needed — unlike avatar/subtitle which need Vite for runtime VRM/FBX loading).

### Process Shutdown Chain

`dev.py` manages the full process tree. On Ctrl+C or dashboard Shutdown button:

1. **Unix:** SIGINT → backend `finally` block → `shutdown_services()` (orchestrator, avatar, subtitles, services) → uvicorn exits → `dev.py` detects exit → tree-kills frontend
2. **Windows:** `dev.py` uses `kill_process_tree` (psutil, `recursive=True`) to walk and terminate the entire process tree — backend, all services, avatar, subtitles, frontend

`shutdown_services()` and `_shutdown_backend_async()` both call `_avatar_kill()`, `_subtitle_kill()`, and `_stream_deck_kill()` as defense-in-depth.

### Game-State Bridge Integration

SpindL connects to game processes via the [SpindL Game Bridge](https://github.com/JChan2787/spindl-game-bridge) — a REFramework Lua plugin + C++ native DLL that emits game events over TCP.

**Architecture:**
```
Game (REFramework) → Lua hooks → C++ TcpServer → TCP 127.0.0.1:53817
    → GameStateModule._connect_and_consume() (Python asyncio)
    → reader.readline() → _process_line() → _buffer_event() → stimulus pipeline

Consumer → writer.write(JSON + \n) → C++ recv_thread → cmd_queue → Lua poll
    → commands.lua dispatch → game API calls → response via event send path
```

**Key files:**
- `stimuli/game_state/module.py` — `GameStateModule` (extends `StimulusModule`). Runs its own `asyncio.new_event_loop()` in a daemon thread. Connects to bridge, reads newline-delimited JSON events, buffers dialogue, builds game-state context for the LLM.
- `stimuli/game_state/dialogue_buffer.py` — Buffers and deduplicates game dialogue events.
- `stimuli/game_state/dialogue_store.py` — Persistent dialogue history.
- `stimuli/game_state/dialogue_summarizer.py` — LLM-driven summarization of dialogue windows.
- `stimuli/game_state/validator.py` — JSON schema validation for bridge events.
- `stimuli/game_state/models.py` — Pydantic models for game events.

**Command/response channel (bidirectional):** `GameStateModule` stores the TCP `writer` from `asyncio.open_connection`. `send_command_and_wait(command, timeout)` dispatches a JSON command to the bridge and awaits a `command_response` event matched by `command_id`. Cross-thread dispatch uses `asyncio.run_coroutine_threadsafe()` (module runs its own event loop in a daemon thread; tool executor calls from the main thread). `asyncio.wrap_future()` bridges the result back.

**Response interception:** `_process_line()` catches `event_type == "command_response"` before `_buffer_event` — command responses are routed to the pending futures dict, not into the stimulus pipeline.

**Gotchas:**
- GameStateModule runs its own event loop — NOT the main asyncio loop. `send_command_and_wait` must use `run_coroutine_threadsafe`, not direct `await`.
- `_pending_commands` cleanup on disconnect: all pending futures get a `disconnected` error result (guard with `fut.done()` to avoid `InvalidStateError` on timeout races).
- Command responses have sequence numbers from the bridge's event send path — they're regular events that happen to be intercepted early.

### Tool Executor + Forced Tool Choice (NANO-134, Session 737)

The tool executor (`tools/executor.py`) runs an iterative loop: LLM call → extract tool calls → execute tools → feed results back → repeat until the LLM produces a text-only response or `max_iterations` (5) is reached.

**Forced tool choice:** When `tool_choice` is set (e.g., by `CodexActivatorPlugin` detecting a lorebook keyword trigger), the executor detects `forced_tool_choice` and **breaks after the first tool execution round**. Without this, the LLM sees tool definitions + a successful result and re-invokes the tool on every iteration (6 API calls instead of 2). After break, a single text-only narration call produces the final response.

**Per-call timeout:** `TOOL_PATH_TIMEOUT = 60.0` (class constant). Injected into provider kwargs when tools are present. The OpenRouter provider accepts `timeout` via `kwargs.get()` (not `pop` — persists across iterations). Worst case: 120s total (2 × 60s), matching the existing single-call timeout. Without this, two sequential 120s timeouts = 240s of dead air.

**Built-in tools:**
- `screen_vision` — captures screenshot, sends to VLM, returns description.
- `game_state_query` — queries the GameStateModule for current game context (hack status, combat state, player info).
- `invoke_hack` — sends `hack_walk` command to the game bridge via `GameStateModule.send_command_and_wait()`. Returns success/error with LLM-friendly descriptions.

### Weighted Stimulus Arbitration (NANO-117)

`StimuliEngine._select_stimulus()` uses weighted lottery scheduling instead of rigid priority ordering. Each stimulus module sets `metadata["weight"]` — e.g., game_state dialogue 2.0, priority events 5.0, Twitch 1.0, Patience 0.5. `WeightedRotator` (`stimuli/weighted_rotator.py`) handles decay after firing and recovery per cycle, preventing starvation of lower-weight modules. Dashboard sliders allow runtime tuning.

## Frontend Architecture

**Stack:** Next.js 16 + React 19 + TypeScript + Tailwind CSS v4 + Zustand + Socket.IO client + Radix UI

### Pages

| Route | Purpose |
|-------|---------|
| `/` | Dashboard — hero character portrait (audio-reactive glow), transcription, responses, tool calls |
| `/launcher` | Service configuration wizard (LLM, STT, TTS, VLM, embedding) — STT/TTS sections have enable/disable toggles |
| `/prompt` | Prompt Workshop — block editor, token breakdown, injection wrappers |
| `/characters` | Character manager — CRUD, avatar cropping, import/export |
| `/memories` | Memory curation — general, flashcards, summaries, search |
| `/codex` | Knowledge base — global + per-character lorebook entries |
| `/sessions` | Conversation history viewer |
| `/settings` | Configuration — personas, VAD, pipeline settings, Twitch credentials, avatar/subtitle/stream-deck toggles, Tauri install flow |

### State Management

13 Zustand stores in `gui/src/lib/stores/`:

| Store | Scope |
|-------|-------|
| `agent-store` | Pipeline state, transcription, responses, health, tools, audio/mic level, shutdown |
| `chat-store` | Chat message history, hydration, metadata (emotion, codex, memories, reasoning) |
| `connection-store` | Socket.IO connection status |
| `launcher-store` | Service config, launch progress, model lists, validation, STT/TTS enable toggles, `historyMode` (NANO-115 pre-launch splice/flatten) |
| `character-store` | Character CRUD, avatar data, VRM binding, import/export |
| `codex-store` | Global + per-character lorebook entries |
| `memory-store` | Memory collections (general/flashcards/summaries), search, CRUD |
| `prompt-store` | Current prompt snapshot (messages + token breakdown) |
| `block-editor-store` | Block config editing state, drag-to-reorder, field overrides |
| `settings-store` | VAD, pipeline, memory, generation params (`force_role_history`: `splice | flatten`, NANO-115), stimuli (patience, twitch with `audience_window`/`audience_char_cap`, addressing-others contexts), tools, LLM/VLM runtime, avatar config (emotion classifier, fade delay, subtitles, stream deck, Tauri install state, connection status) |
| `session-store` | Conversation history, session resume/delete/summarize |
| `vts-store` | VTubeStudio connection state, plugin auth, hotkey/expression/parameter/model lists |
| `use-service-status` | Service capability hooks — `useServiceDisabled()`, `useServiceStatus()` for runtime toggle gating (NANO-116) |

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

Test markers: `@pytest.mark.slow`, `@pytest.mark.vision`, `@pytest.mark.cloud`, `@pytest.mark.hardware`.

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
- **Components (8):** block-detail, block-list, prompt-viewer, token-breakdown, session-list, session-viewer, llm-config, vtubestudio-card
- **API routes (2):** fetch-models, write-config
- **Schemas (1):** config-schemas (28 test suites covering all Zod schemas)
- **Pages (2):** codex, memories
- **Test infrastructure:** `gui/src/test/setup.ts` (cleanup, mocks), `gui/src/test/mocks/socket.ts` (createMockSocket factory with `_simulateEvent()` helper), `gui/src/test/utils.tsx` (render helpers)

### E2E (Playwright)

```bash
conda run -n spindl python -m pytest tests_e2e/ -v
```

Requires running services. E2E config fixtures at `tests_e2e/fixtures/config/` (5 named configs covering local/cloud LLM × VLM combos + unified mode: `spindl_e2e_local_unified`, `spindl_e2e_local_local`, `spindl_e2e_local_cloud`, `spindl_e2e_cloud_local`, `spindl_e2e_cloud_cloud`).

## Lessons Learned

### Cross-Thread Socket.IO Event Ordering (NANO-111, Session 606)

Async events from different threads do NOT have guaranteed ordering across a Socket.IO bridge. When multiple backend threads emit events via `EventBus → bridge → run_coroutine_threadsafe → Socket.IO`, the frontend receives them in whatever order the asyncio event loop schedules them — not the order they were emitted.

**The pattern:** When two event types serve different purposes (e.g., `llm_chunk` for real-time incremental display, `response` for authoritative metadata finalization), design them so neither depends on arriving first. The `response` event is the source of truth for bubble creation and metadata. The `llm_chunk` event is the source of truth for sequential sentence-by-sentence display during TTS playback. If `response` arrives before any `llm_chunk`, it creates the bubble with all data. If `llm_chunk` events arrive first, they build incrementally, and `response` finalizes without overwriting the incremental state. Either ordering produces the correct result.

**Anti-pattern:** Relying on event A to set state that event B reads. If A and B cross thread boundaries through async scheduling, B may arrive first.

### State Machine Transitions and TTS-Off (NANO-112, Session 609)

`finish_system_speaking()` only transitions from `SYSTEM_SPEAKING`. If TTS is disabled, the state machine is in `PROCESSING` when the LLM response completes — calling `finish_system_speaking()` is a no-op and the agent gets stuck. Use a direct `_transition(AudioState.PROCESSING, AudioState.LISTENING)` via a dedicated `_on_tts_skipped` callback.

**The pattern:** When bypassing a pipeline stage that normally triggers a state transition, wire a dedicated callback for the bypass path instead of reusing the normal-path callback. The normal callback may have preconditions (like "must be in state X") that don't hold in the bypass case.

### Text-Only Chunk Delivery (NANO-112, Session 609)

`LLMChunkEvent`s are tied to the streaming TTS playback lifecycle — `currentAssistantMsgId` in the frontend's `SocketProvider` tracks which parent bubble owns the chunks. Emitting `LLMChunkEvent`s without that lifecycle (e.g., synchronous text-only chunking) creates a second parent bubble.

**The fix:** Route text-only chunks through `fallback_chunks` on the `response` event instead. The frontend renders them as sub-bubbles within the response bubble, not as standalone chunks.

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
- `services:` — per-service config (enabled, platform, command, health_check, depends_on). STT and TTS have `enabled` flags that gate provider initialization — these are the flags the launcher GUI toggles write to (NANO-112)
- `providers:` — LLM, STT, TTS, VLM provider settings
- `llm:` — generation params (temperature, top_p, max_tokens, repeat_penalty, etc.) + `force_role_history`: `splice | flatten` (NANO-115, default `flatten`; legacy `auto` coerced to `flatten` on load)
- `memory:` — enabled, relevance_threshold, top_k, reflection_interval, reflection_prompt, reflection_system_message, reflection_delimiter, scoring_w_relevance, scoring_w_recency
- `stimuli:` — enabled, patience config (enabled, seconds, prompt), twitch config (enabled, channel, credentials, `twitch_audience_window` 25–300, `twitch_audience_char_cap` 50–500, prompt_template — NANO-115), addressing_others config (contexts list with id/label/prompt)
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
