from __future__ import annotations

import asyncio
import random
from collections.abc import Awaitable, Callable
from uuid import uuid4

from .llm import LLMClients, fallback_decision
from .models import ActiveSession, AgentName, OrchestratorDecision, TranscriptMessage
from .parsing import choose_winner, extract_constraints, parse_consensus
from .social_debt import debt_modifier_for, get_social_debt, record_consensus

EmitFn = Callable[[str, dict, str | None], Awaitable[None]]

# Maps Orchestrator's lowercase speaker token → AgentName used throughout the app
_SPEAKER_MAP: dict[str, AgentName] = {
    "optimizer": "Optimizer",
    "vibe_check": "Vibe-Check",
}

_IDLE_NUDGES = [
    "uh... hello? you still there?",
    "i mean... we're waiting.",
    "like, no pressure, but... we kind of need an answer here.",
    "um, did we lose you?",
]


class SessionManager:
    def __init__(self, llm: LLMClients):
        self.llm = llm
        self.sessions: dict[str, ActiveSession] = {}
        self.tasks: dict[str, asyncio.Task] = {}
        self.idle_tasks: dict[str, asyncio.Task] = {}
        self.pending_fields: dict[str, list[str]] = {}
        self.lock = asyncio.Lock()

    async def create_session(self, group_id: str, dilemma: str) -> ActiveSession:
        debt = get_social_debt(group_id)
        session = ActiveSession(
            session_id=str(uuid4()),
            group_id=group_id,
            dilemma=dilemma,
            social_debt_modifier=debt_modifier_for(debt.debt_balance),
            debt_balance=debt.debt_balance,
        )
        self.sessions[session.session_id] = session
        return session

    def get(self, session_id: str) -> ActiveSession | None:
        return self.sessions.get(session_id)

    async def start_loop(
        self,
        session_id: str,
        emit: EmitFn,
        first_agent: AgentName | None = None,
    ) -> None:
        await self.cancel_loop(session_id)
        task = asyncio.create_task(
            self._conversation_loop(session_id, emit, first_agent=first_agent)
        )
        self.tasks[session_id] = task

    async def cancel_loop(self, session_id: str) -> None:
        task = self.tasks.pop(session_id, None)
        if task and not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    # ------------------------------------------------------------------
    # Idle nudge helpers
    # ------------------------------------------------------------------

    def _start_idle_nudge(self, session_id: str, emit: EmitFn) -> None:
        """Cancel any existing idle timer and start a fresh 12s nudge loop."""
        self._cancel_idle(session_id)
        self.idle_tasks[session_id] = asyncio.create_task(
            self._idle_nudge_loop(session_id, emit)
        )

    def _cancel_idle(self, session_id: str) -> None:
        task = self.idle_tasks.pop(session_id, None)
        if task and not task.done():
            task.cancel()

    async def _idle_nudge_loop(self, session_id: str, emit: EmitFn) -> None:
        """Every 12s of user silence, inject a short nudge from the last speaker."""
        while True:
            await asyncio.sleep(12)
            session = self.sessions.get(session_id)
            if not session or session.status != "awaiting_user_answer":
                return
            agent: AgentName = session.last_speaker or random.choice(["Optimizer", "Vibe-Check"])
            nudge = random.choice(_IDLE_NUDGES)
            session.transcript.append(TranscriptMessage(speaker=agent, text=nudge))
            await emit("message_chunk", {"speaker": agent, "chunk": nudge}, session_id)
            await emit("sentence_audio_ready", {"speaker": agent, "text": nudge}, session_id)
            await emit("room_state_update", session.model_dump(mode="json"), session_id)

    # ------------------------------------------------------------------
    # Public interjection handler
    # ------------------------------------------------------------------

    async def handle_interjection(self, session_id: str, text: str, emit: EmitFn) -> ActiveSession | None:
        session = self.get(session_id)
        if not session:
            return None

        # Ignore if we're already in the middle of a wrap-up
        if session.status == "user_interrupting":
            return session

        has_active_task = session_id in self.tasks and not self.tasks[session_id].done()

        if session.status == "speaking" and has_active_task:
            # Mid-speech: signal wrap-up, then process
            session.status = "user_interrupting"
            await emit("room_state_update", session.model_dump(mode="json"), session_id)
            await self.cancel_loop(session_id)
            await self._run_wrap_up(session_id, emit)
        else:
            await self.cancel_loop(session_id)

        await self._process_interjection(session_id, text, emit)
        return session

    # ------------------------------------------------------------------
    # Wrap-up: interrupted agent finishes thought in one sentence
    # ------------------------------------------------------------------

    async def _run_wrap_up(self, session_id: str, emit: EmitFn) -> None:
        session = self.sessions[session_id]
        agent = session.last_speaker
        if not agent:
            return
        await emit("agent_typing", {"speaker": agent}, session_id)

        full_text = ""
        try:
            async for chunk in self.llm.stream_wrap_up(agent, session):
                if agent == "Optimizer":
                    chunk = self.llm._strip_parentheticals(chunk)
                full_text += chunk
                if chunk:
                    await emit("message_chunk", {"speaker": agent, "chunk": chunk}, session_id)
        except Exception:
            pass  # silently skip; conversation continues regardless

        if full_text.strip():
            session.transcript.append(TranscriptMessage(speaker=agent, text=full_text.strip()))
        await emit("room_state_update", session.model_dump(mode="json"), session_id)

    # ------------------------------------------------------------------
    # Shared interjection processing: Orchestrator extracts constraints
    # and decides who resumes
    # ------------------------------------------------------------------

    async def _process_interjection(self, session_id: str, text: str, emit: EmitFn) -> None:
        # User replied — cancel any idle nudge timer
        self._cancel_idle(session_id)

        session = self.sessions[session_id]
        session.transcript.append(TranscriptMessage(speaker="User", text=text))
        session.pending_question = False
        session.pending_question_asker = None

        # --- Orchestrator extracts constraints + decides who resumes ---
        try:
            decision = await self.llm.orchestrate(session)
            _apply_constraints(session, decision)
        except Exception:
            decision = fallback_decision(session)

        # Regex fallback for anything Orchestrator missed
        pending = self.pending_fields.pop(session_id, [])
        regex_extracted = extract_constraints(text)
        for field, value in regex_extracted.items():
            if session.known_constraints.get(field, "Unknown") == "Unknown" and value:
                session.known_constraints[field] = value
        _ = pending  # consumed above via Orchestrator; kept for legacy fallback path

        session.status = "speaking"
        await emit("room_state_update", session.model_dump(mode="json"), session_id)
        await self.start_loop(session_id, emit, first_agent=_SPEAKER_MAP.get(decision.next_speaker))

    # ------------------------------------------------------------------
    # Main conversation loop – Orchestrator drives every transition
    # ------------------------------------------------------------------

    async def _conversation_loop(
        self,
        session_id: str,
        emit: EmitFn,
        first_agent: AgentName | None = None,
    ) -> None:
        session = self.sessions[session_id]

        # Pick the starting agent — randomise on fresh sessions
        if first_agent:
            next_agent: AgentName = first_agent
        elif session.last_speaker is None:
            next_agent = random.choice(["Optimizer", "Vibe-Check"])
        elif session.last_speaker == "Vibe-Check":
            next_agent = "Optimizer"
        else:
            next_agent = "Vibe-Check"

        while session.current_turn < session.max_turns and session.status == "speaking":
            session.current_turn += 1
            agent = next_agent
            session.last_speaker = agent

            # Is this agent reacting to a pending question (reaction-only turn)?
            reaction_only = (
                session.pending_question
                and session.pending_question_asker is not None
                and session.pending_question_asker != agent
            )
            force = session.current_turn >= session.max_turns

            await emit("agent_typing", {"speaker": agent}, session_id)
            await emit("room_state_update", session.model_dump(mode="json"), session_id)

            # --- Stream agent turn ---
            full_text, aborted = await self._run_agent_turn(
                session_id, agent, emit, force=force, reaction_only=reaction_only
            )
            if aborted:
                return

            if full_text.strip():
                session.transcript.append(TranscriptMessage(speaker=agent, text=full_text))

            # Detect if the agent introduced a new question
            if "?" in full_text and not reaction_only:
                session.pending_question = True
                session.pending_question_asker = agent
                missing = [k for k, v in session.known_constraints.items() if v == "Unknown"]
                if missing:
                    self.pending_fields[session_id] = missing

            # --- Fire Orchestrator concurrently — overlaps with emit/state work ---
            orchestrator_task = asyncio.create_task(self.llm.orchestrate(session))

            # Apply any constraints the Orchestrator extracted
            try:
                decision = await orchestrator_task
            except Exception:
                decision = fallback_decision(session)

            _apply_constraints(session, decision)

            # --- State transitions based on Orchestrator ---
            if decision.status == "consensus_reached":
                await self._finalize_consensus(session_id, decision, agent, emit)
                return

            if decision.next_speaker == "user":
                session.status = "awaiting_user_answer"
                session.pending_question = True
                missing = [k for k, v in session.known_constraints.items() if v == "Unknown"]
                await emit("interrogation_triggered", {"missing_fields": missing}, session_id)
                await emit("room_state_update", session.model_dump(mode="json"), session_id)
                # Start 12s idle nudge timer
                self._start_idle_nudge(session_id, emit)
                return

            # Map Orchestrator's decision to the next agent
            mapped = _SPEAKER_MAP.get(decision.next_speaker)
            next_agent = mapped if mapped else (
                "Vibe-Check" if agent == "Optimizer" else "Optimizer"
            )
            session.status = decision.status if decision.status != "consensus_reached" else "speaking"
            session.pending_question = decision.next_speaker == "user"

            await emit("room_state_update", session.model_dump(mode="json"), session_id)
            # No explicit sleep — Orchestrator API latency already provides natural turn pacing

        # Failsafe: max turns hit without consensus
        if session.status == "speaking":
            await self._force_consensus(session_id, emit)

    # ------------------------------------------------------------------
    # Single agent turn (streaming)
    # ------------------------------------------------------------------

    async def _run_agent_turn(
        self,
        session_id: str,
        agent: AgentName,
        emit: EmitFn,
        *,
        force: bool = False,
        reaction_only: bool = False,
    ) -> tuple[str, bool]:
        """Stream one agent turn. Returns (full_text, was_aborted).

        Maintains a sentence buffer: flushes via sentence_audio_ready whenever
        a sentence-ending character (.?!\n) appears in the incoming chunk.
        """
        session = self.sessions[session_id]
        full_text = ""
        sentence_buf = ""
        try:
            stream = self.llm.stream_agent(agent, session, force=force, reaction_only=reaction_only)
            while True:
                if session.status == "user_interrupting":
                    if sentence_buf.strip():
                        await emit("sentence_audio_ready", {"speaker": agent, "text": sentence_buf.strip()}, session_id)
                    return full_text, True
                try:
                    chunk = await asyncio.wait_for(stream.__anext__(), timeout=45)
                except StopAsyncIteration:
                    break
                if agent == "Optimizer":
                    chunk = self.llm._strip_parentheticals(chunk)
                full_text += chunk
                if chunk:
                    await emit("message_chunk", {"speaker": agent, "chunk": chunk}, session_id)
                    sentence_buf += chunk
                    if any(c in chunk for c in ".?!\n"):
                        if sentence_buf.strip():
                            await emit("sentence_audio_ready", {"speaker": agent, "text": sentence_buf.strip()}, session_id)
                        sentence_buf = ""
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            error_text = f"{agent} error: {type(exc).__name__}: {exc}"
            session.transcript.append(TranscriptMessage(speaker="System", text=error_text))
            await emit("message_chunk", {"speaker": "System", "chunk": error_text}, session_id)
            await emit("room_state_update", session.model_dump(mode="json"), session_id)
            return full_text, True  # treat as aborted so loop exits cleanly

        # Flush any trailing text that didn't end with punctuation
        if sentence_buf.strip():
            await emit("sentence_audio_ready", {"speaker": agent, "text": sentence_buf.strip()}, session_id)

        return full_text, False

    # ------------------------------------------------------------------
    # Consensus finalisation
    # ------------------------------------------------------------------

    async def _finalize_consensus(
        self,
        session_id: str,
        decision: OrchestratorDecision,
        last_agent: AgentName,
        emit: EmitFn,
    ) -> None:
        self._cancel_idle(session_id)
        session = self.sessions[session_id]

        # Prefer the Orchestrator's explicit decision text; fall back to regex scan
        final_text = decision.final_decision
        if not final_text:
            for msg in reversed(session.transcript):
                candidate = parse_consensus(msg.text)
                if candidate:
                    final_text = candidate
                    break
        if not final_text:
            final_text = session.transcript[-1].text if session.transcript else "Decision reached."

        winner = choose_winner(session.transcript[-1].text if session.transcript else "", fallback=last_agent)
        debt = record_consensus(session.group_id, session.dilemma, winner)

        session.status = "consensus_reached"
        session.final_decision = final_text
        session.winner = winner
        session.debt_balance = debt.debt_balance
        session.social_debt_modifier = debt_modifier_for(debt.debt_balance)

        await emit(
            "consensus_reached",
            {"final_decision": final_text, "winner": winner, "new_debt_balance": debt.debt_balance},
            session_id,
        )
        await emit("room_state_update", session.model_dump(mode="json"), session_id)

    async def _force_consensus(self, session_id: str, emit: EmitFn) -> None:
        """Failsafe: max turns hit. Force the last agent to declare a decision."""
        self._cancel_idle(session_id)
        session = self.sessions[session_id]
        agent: AgentName = "Optimizer" if session.last_speaker == "Vibe-Check" else "Vibe-Check"
        session.last_speaker = agent

        await emit("agent_typing", {"speaker": agent}, session_id)
        full_text, _ = await self._run_agent_turn(session_id, agent, emit, force=True)
        if full_text.strip():
            session.transcript.append(TranscriptMessage(speaker=agent, text=full_text))

        decision = OrchestratorDecision(
            next_speaker="user",
            status="consensus_reached",
            updated_constraints={},
            reasoning="Forced consensus at max turns.",
            final_decision=parse_consensus(full_text) or full_text.strip() or "Decision reached.",
        )
        await self._finalize_consensus(session_id, decision, agent, emit)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _apply_constraints(session: ActiveSession, decision: OrchestratorDecision) -> None:
    """Merge non-null Orchestrator constraint updates into the session."""
    for field, value in decision.updated_constraints.items():
        if value and str(value).lower() not in ("unknown", "null", "none", ""):
            session.known_constraints[field] = str(value)
