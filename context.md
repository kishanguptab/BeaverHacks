# The Decider — Codebase Context

Real-time multi-agent debate app. Two AI podcast co-hosts argue over a user's dilemma and reach a consensus. Built for BeaverHacks hackathon.

---

## Quickstart

```bash
# Backend (port 8000)
cd backend
uvicorn app.main:socket_app --port 8000 --reload

# Frontend (port 3000)
cd frontend
npm run dev
```

Requires `backend/.env` with `GEMINI_API_KEY`, `NVIDIA_API_KEY` (and optionally `OPENAI_API_KEY` for provider TTS). Copy from `backend/.env.example`.

---

## Architecture

```
frontend (Next.js 14, App Router)
  └── Socket.IO client  ──→  backend (FastAPI + python-socketio)
                                  ├── SessionManager (orchestration.py)
                                  │     └── _conversation_loop()
                                  ├── LLMClients (llm.py)
                                  │     ├── Optimizer  → Gemini (google-generativeai)
                                  │     ├── Vibe-Check → Gemini (google-generativeai)
                                  │     └── Orchestrator → NVIDIA Nemotron (OpenAI SDK)
                                  └── social_debt.json  (persistent debt ledger)
```

The ASGI app is `socket_app = socketio.ASGIApp(sio, other_asgi_app=fastapi_app)` — start with `app.main:socket_app`, not `app.main:app`.

---

## Backend File Map (`backend/app/`)

| File | Responsibility |
|---|---|
| `main.py` | FastAPI + Socket.IO event handlers. REST: `GET /health`, `GET /sessions/{id}`, `GET /tts/available`, `POST /tts`. Socket.IO events: `create_room`, `join_room`, `user_interjection` |
| `models.py` | Pydantic models: `ActiveSession`, `OrchestratorDecision`, `TranscriptMessage`, `SocialDebt`, payloads. `SessionStatus` is a 5-value Literal |
| `config.py` | `Settings` via pydantic-settings. Reads `.env`. Keys: `GEMINI_API_KEY`, `NVIDIA_API_KEY`, `NVIDIA_ORCHESTRATOR_MODEL`, `OPENAI_API_KEY` (optional) |
| `llm.py` | All LLM calls. See section below |
| `orchestration.py` | `SessionManager` — state machine loop, interjection handling, idle nudge timer. See section below |
| `parsing.py` | Regex helpers: `parse_interrogation`, `parse_consensus`, `extract_constraints`, `choose_winner` |
| `social_debt.py` | Read/write `data/social_debt.json`. `get_social_debt`, `record_consensus`, `debt_modifier_for` |

---

## `llm.py` — Key Details

### Models
- **Optimizer**: `google.generativeai` — `GenerativeModel(gemini_model, system_instruction=system)` + `start_chat(history).send_message(...)`. Constructed fresh per call (system_instruction is session-specific).
- **Vibe-Check**: same Gemini pattern.
- **Orchestrator**: `self.nvidia` (AsyncOpenAI pointed at `https://integrate.api.nvidia.com/v1`), model `nvidia/llama-3.3-nemotron-super-49b-v1`. Non-streaming, `temperature=0.05`, returns JSON only.

### Prompts
- `OPTIMIZER_PROMPT` — all-lowercase humanizer persona, filler words, ellipsis pauses, no markdown, no consensus rule (Orchestrator owns consensus). `{social_debt_modifier}` placeholder.
- `VIBE_PROMPT` — same structure, different persona.
- `ORCHESTRATOR_PROMPT` — strict JSON referee with 6 hard rules (no pile-ons, question cap ≥2→user, constraint fidelity, consensus requires both agents to agree, alternation, force consensus at max turns).
- `WRAP_UP_SUFFIX`, `REACTION_ONLY_SUFFIX`, `FORCED_DECISION_PROMPT` — appended to system prompt for special turn types.

### Transcript → native message arrays
- `_build_gemini_history(transcript, own_speaker)` — maps transcript to Gemini `model`/`user` roles. Merges consecutive same-role entries (Gemini strict alternation). Inserts `(conversation start)` if history starts with `model`.
- `_build_openai_history(transcript, own_speaker)` — maps to OpenAI `assistant`/`user` roles (kept for future use; Optimizer no longer uses NVIDIA).

### Temperature
Both agents: `1.15` (high creativity for humanized output). Wrap-up/reaction turns use the same value.

### TTS (optional)
`text_to_speech(agent, text)` uses `AsyncOpenAI` (OpenAI TTS-1). Only active if `OPENAI_API_KEY` is set. Frontend falls back to browser `speechSynthesis` otherwise.

---

## `orchestration.py` — Key Details

### `SessionManager`
```python
self.sessions: dict[str, ActiveSession]
self.tasks: dict[str, asyncio.Task]      # conversation loop tasks
self.idle_tasks: dict[str, asyncio.Task] # idle nudge timers
self.pending_fields: dict[str, list[str]]
```

### Session Status State Machine
```
speaking  ──(agent turn done, Orchestrator says user)──→  awaiting_user_answer
                                                               │
                  ←──(user interjection)────────────────────────┘
speaking  ──(user breaks in mid-speech)──→  user_interrupting  ──→  speaking
speaking  ──(Orchestrator: consensus_reached)──→  consensus_reached
```

### Conversation Loop Flow (per turn)
1. Emit `agent_typing` + `room_state_update`
2. Stream agent turn via `_run_agent_turn` (chunks → `message_chunk` events)
3. Append completed text to `session.transcript`
4. Fire `asyncio.create_task(self.llm.orchestrate(session))` immediately (latency overlap)
5. Do question detection
6. `await` orchestrator task
7. Apply constraints, route next state:
   - `consensus_reached` → `_finalize_consensus`
   - `next_speaker == "user"` → set `awaiting_user_answer`, emit `interrogation_triggered`, start idle nudge
   - Otherwise → pick next agent, loop

No `asyncio.sleep` between turns — Orchestrator API latency provides natural pacing.

### First Speaker
`random.choice(["Optimizer", "Vibe-Check"])` when `session.last_speaker is None`.

### Idle Nudge Timer
`_start_idle_nudge(session_id, emit)` → `_idle_nudge_loop` sleeps 12s, checks status is still `awaiting_user_answer`, injects a random nudge line from `_IDLE_NUDGES`, repeats. Cancelled in `_process_interjection` (user replied) and `_finalize_consensus`/`_force_consensus`.

### Interjection Flow
`handle_interjection` → if mid-speech, sets `user_interrupting`, cancels loop, runs `_run_wrap_up` (1-sentence finish), then `_process_interjection`. If already idle, goes straight to `_process_interjection`.
`_process_interjection` cancels idle nudge, appends user message, calls Orchestrator, resumes loop.

---

## Frontend File Map (`frontend/src/`)

| File | Responsibility |
|---|---|
| `store/socket-store.ts` | Zustand store. All Socket.IO state. `connect`, `createRoom`, `joinRoom`, `interject`. Exports `ActiveSession`, `SessionStatus` types |
| `lib/useTTS.ts` | TTS hook. Reads `session.transcript` for completed messages. Sequential queue. Cancels on `user_interrupting`. Supports browser `speechSynthesis` or provider (`POST /tts`) |
| `lib/utils.ts` | `API_URL` constant, `cn` classname helper |
| `components/LiveTranscript.tsx` | Transcript bubbles + streaming in-progress bubble. `StatusBanner` for non-speaking states. Mute/unmute button wired to `useTTS` |
| `components/InterjectInput.tsx` | Text input + browser STT (SpeechRecognition API). "Break in" button (amber/Zap) during `speaking`. "Send" button (sky-blue) during `awaiting_user_answer`. Idle hint strip |
| `components/AvatarNode.tsx` | Per-agent avatar with status chip (Speaking / Asked–waiting / Wrapping up / On hold) |
| `components/ConstraintPanel.tsx` | Displays `session.known_constraints` as a live panel |
| `components/DebtMeter.tsx` | Displays `session.debt_balance` |
| `app/page.tsx` | Landing — create room form |
| `app/room/[session_id]/page.tsx` | Room view, mounts all components |

### Socket.IO Events (backend → frontend)
| Event | Payload | Meaning |
|---|---|---|
| `room_state_update` | `ActiveSession` | Full session state sync |
| `agent_typing` | `{ speaker }` | Agent started turn |
| `message_chunk` | `{ speaker, chunk }` | Streaming text chunk |
| `interrogation_triggered` | `{ missing_fields }` | Agents waiting for user |
| `consensus_reached` | `{ final_decision, winner, new_debt_balance }` | Done |

### Socket.IO Events (frontend → backend)
| Event | Payload |
|---|---|
| `create_room` | `{ group_id, initial_dilemma }` |
| `join_room` | `{ session_id, user_name }` |
| `user_interjection` | `{ session_id, text }` |

---

## Data Models (Pydantic, `models.py`)

```python
SessionStatus = Literal["speaking", "user_interrupting", "awaiting_user_answer",
                         "resuming_with_new_context", "consensus_reached"]

ActiveSession:
  session_id, group_id, dilemma
  known_constraints: dict  # keys: time, budget, audience, audience_vibe, team_energy
  current_turn, max_turns (default 8)
  transcript: list[TranscriptMessage]
  social_debt_modifier: str
  status: SessionStatus
  debt_balance: float
  winner, final_decision
  pending_question: bool
  pending_question_asker, last_speaker

OrchestratorDecision:
  next_speaker: "optimizer" | "vibe_check" | "user"
  status: SessionStatus
  updated_constraints: dict[str, str | None]
  reasoning: str
  final_decision: str | None
```

---

## Social Debt System

Persisted to `backend/data/social_debt.json` keyed by `group_id`. Each consensus win increments/decrements `debt_balance` by 1.0. `debt_modifier_for(balance)` returns a string injected into agent prompts to bias the losing agent toward conceding in future debates.

---

## Known Constraints

The five tracked constraint fields are: `time`, `budget`, `audience`, `audience_vibe`, `team_energy`. Default value `"Unknown"`. Orchestrator extracts them from user messages and merges non-null values into `session.known_constraints`. Regex fallback in `parsing.extract_constraints` catches anything the Orchestrator misses.
