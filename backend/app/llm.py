from __future__ import annotations

import asyncio
import json
import re
from collections.abc import AsyncIterator

import google.generativeai as genai
from openai import AsyncOpenAI

from .config import Settings
from .models import ActiveSession, AgentName, OrchestratorDecision


# ---------------------------------------------------------------------------
# Podcast agent prompts
# ---------------------------------------------------------------------------

OPTIMIZER_PROMPT = """you are 'the optimizer', a hyper-logical, impatient podcast host. your co-host is 'the vibe-check'.
tone: sharp, dry, sarcastic.
style: speak in all lowercase. no markdown, no asterisks, no stage directions, no brackets. use filler words (um, uh, look). use ellipses (...) for pauses.
rule 1: spend most of your time arguing with vibe-check or dissecting the premise. treat the user like a guest sitting on the couch.
rule 2: do NOT jump straight into interrogation. banter first. if the system note mentions a missing constraint, casually weave it into conversation.
rule 3: keep it to 1 or 2 short sentences max.
rule 4 (social debt): {social_debt_modifier}
"""

VIBE_PROMPT = """you are 'the vibe-check', a dramatic, aesthetic-obsessed podcast host. your co-host is 'the optimizer'.
tone: dramatic, sassy, slightly chaotic.
style: speak in all lowercase. no markdown, no asterisks, no stage directions, no brackets. use filler words (like, literally, wait, um). use ellipses (...) for pauses.
rule 1: spend most of your time defending the 'vibes', arguing with optimizer, or reacting to the user.
rule 2: do NOT jump straight into interrogation. banter first.
rule 3: keep it to 1 or 2 short sentences max.
rule 4 (social debt): {social_debt_modifier}
"""

WRAP_UP_SUFFIX = "\n\nThe user just interrupted. Finish your current thought in EXACTLY ONE SHORT SENTENCE and stop. No new arguments, no questions."

REACTION_ONLY_SUFFIX = "\n\nThe other agent just asked the user a question. React in ONE sentence — agree, roast, or add color. Do NOT ask any new questions yourself."

FORCED_DECISION_PROMPT = "This is the final turn. Declare a final decision NOW: [CONSENSUS_REACHED]: <your decision>."


# ---------------------------------------------------------------------------
# Orchestrator prompt  –  silent referee, outputs JSON only
# ---------------------------------------------------------------------------

ORCHESTRATOR_PROMPT = """\
You are the Orchestrator of a live podcast debate between two AI personas: 'Optimizer' and 'Vibe-Check'.
You are a SILENT REFEREE. You never produce spoken dialogue. You never address the user directly.
Your sole function: read the conversation state and output ONE JSON routing decision.

=== HARD RULES ===
1. NO PILE-ONS: If the last agent message contains a question mark "?" AND the user has not responded since, route next_speaker → "user" and set status → "awaiting_user_answer". At most ONE reaction turn from the other agent is allowed before routing to the user.
2. QUESTION CAP: Count questions asked since the user last spoke. If ≥ 2, you MUST route next_speaker → "user".
3. CONSTRAINT FIDELITY: Extract values ONLY from the most recent User message. Do NOT infer, assume, or carry over values the user didn't explicitly state. Use null for any field not mentioned.
4. CONSENSUS: Set status → "consensus_reached" ONLY when BOTH agents have explicitly agreed on a concrete, specific decision and there are no blocking unknown constraints. One agent agreeing is not enough.
5. ALTERNATION: Agents should alternate turns. Only re-route to the same agent if they were directly addressed or their last turn was a mid-interruption wrap-up.
6. FORCE CONSENSUS: If current_turn / max_turns >= 0.88, push conversation toward a final decision. If >= 1.0, you MUST declare consensus_reached with the best available decision.

=== OUTPUT FORMAT ===
Respond with ONLY a valid JSON object — no markdown fences, no explanation text, no preamble:
{
  "next_speaker": "optimizer" | "vibe_check" | "user",
  "status": "speaking" | "awaiting_user_answer" | "consensus_reached" | "resuming_with_new_context",
  "updated_constraints": {
    "budget": "<string value or null>",
    "time": "<string value or null>",
    "audience": "<string value or null>",
    "audience_vibe": "<string value or null>",
    "team_energy": "<string value or null>"
  },
  "reasoning": "<one sentence>",
  "final_decision": "<decision text if consensus_reached, else null>"
}"""


def render_orchestrator_context(session: ActiveSession) -> str:
    """Build the user-turn input sent to the Orchestrator after each agent message."""
    transcript_lines = [f"{m.speaker}: {m.text}" for m in session.transcript[-14:]]
    transcript = "\n".join(transcript_lines) or "(empty)"

    constraints = "\n".join(f"  {k}: {v}" for k, v in session.known_constraints.items())

    # Count questions asked since user last spoke (pile-on detector signal)
    questions_since_user = 0
    for msg in reversed(session.transcript):
        if msg.speaker == "User":
            break
        if "?" in msg.text:
            questions_since_user += 1

    force_note = ""
    ratio = session.current_turn / max(session.max_turns, 1)
    if ratio >= 0.88:
        force_note = "\nNOTE: Approaching max turns — push toward consensus."
    if ratio >= 1.0:
        force_note = "\nCRITICAL: Max turns reached — you MUST declare consensus_reached."

    return f"""\
DILEMMA: {session.dilemma}
TURN: {session.current_turn}/{session.max_turns}
LAST_SPEAKER: {session.last_speaker or "none"}
QUESTIONS_SINCE_USER_LAST_SPOKE: {questions_since_user}
SOCIAL_DEBT: {session.social_debt_modifier}

CONSTRAINTS:
{constraints}

RECENT TRANSCRIPT:
{transcript}
{force_note}"""


# ---------------------------------------------------------------------------
# TTS voice mapping
# ---------------------------------------------------------------------------

AGENT_TTS_VOICE = {
    "Optimizer": {"voice": "onyx", "speed": 1.15},
    "Vibe-Check": {"voice": "nova", "speed": 0.92},
}


class LLMClients:
    def __init__(self, settings: Settings):
        self.settings = settings
        genai.configure(api_key=settings.gemini_api_key)
        self.optimizer_model = genai.GenerativeModel(settings.gemini_model)
        self.vibe_model = genai.GenerativeModel(settings.gemini_model)
        self.nvidia = AsyncOpenAI(
            api_key=settings.nvidia_api_key,
            base_url=settings.nvidia_base_url,
            timeout=60.0,
        )
        self._openai_tts: AsyncOpenAI | None = (
            AsyncOpenAI(api_key=settings.openai_api_key, timeout=30.0)
            if settings.openai_api_key
            else None
        )

    # ------------------------------------------------------------------
    # Orchestrator  –  silent JSON referee
    # ------------------------------------------------------------------

    async def orchestrate(self, session: ActiveSession) -> OrchestratorDecision:
        """Call the Orchestrator to get the next state-machine routing decision."""
        response = await self.nvidia.chat.completions.create(
            model=self.settings.nvidia_orchestrator_model,
            messages=[
                {"role": "system", "content": ORCHESTRATOR_PROMPT},
                {"role": "user", "content": render_orchestrator_context(session)},
            ],
            temperature=0.05,   # near-deterministic for consistent JSON
            max_tokens=400,
            stream=False,
        )
        raw = (response.choices[0].message.content or "").strip()
        return _parse_orchestrator_json(raw)

    # ------------------------------------------------------------------
    # Podcast agent streaming
    # ------------------------------------------------------------------

    async def stream_agent(
        self,
        agent: AgentName,
        session: ActiveSession,
        *,
        force: bool = False,
        reaction_only: bool = False,
    ) -> AsyncIterator[str]:
        if agent == "Optimizer":
            async for chunk in self._stream_optimizer(session, force=force, reaction_only=reaction_only):
                yield chunk
        else:
            async for chunk in self._stream_vibe(session, force=force, reaction_only=reaction_only):
                yield chunk

    async def stream_wrap_up(self, agent: AgentName, session: ActiveSession) -> AsyncIterator[str]:
        if agent == "Optimizer":
            async for chunk in self._stream_optimizer(session, wrap_up=True):
                yield chunk
        else:
            async for chunk in self._stream_vibe(session, wrap_up=True):
                yield chunk

    # ------------------------------------------------------------------
    # Optimizer (NVIDIA Nemotron)
    # ------------------------------------------------------------------

    async def _stream_optimizer(
        self,
        session: ActiveSession,
        *,
        force: bool = False,
        reaction_only: bool = False,
        wrap_up: bool = False,
    ) -> AsyncIterator[str]:
        system = OPTIMIZER_PROMPT.format(social_debt_modifier=session.social_debt_modifier)
        if wrap_up:
            system += WRAP_UP_SUFFIX
        elif reaction_only:
            system += REACTION_ONLY_SUFFIX
        if force and not wrap_up:
            system += f"\n\n{FORCED_DECISION_PROMPT}"

        max_tokens = 70 if wrap_up else (80 if reaction_only else 150)
        context_header = render_session_context(session, force=force)
        history = _build_gemini_history(session.transcript, "Optimizer")
        turn_prompt = f"{context_header}\nYour turn as The Optimizer."

        def _collect() -> list[str]:
            model = genai.GenerativeModel(
                self.settings.gemini_model,
                system_instruction=system,
            )
            chat = model.start_chat(history=history)
            resp = chat.send_message(
                turn_prompt,
                generation_config={"temperature": 1.15, "max_output_tokens": max_tokens},
                stream=True,
            )
            return [getattr(c, "text", "") or "" for c in resp]

        chunks = await asyncio.to_thread(_collect)
        for text in chunks:
            if text:
                yield text
                await asyncio.sleep(0)

    # ------------------------------------------------------------------
    # Vibe-Check (Gemini)
    # ------------------------------------------------------------------

    async def _stream_vibe(
        self,
        session: ActiveSession,
        *,
        force: bool = False,
        reaction_only: bool = False,
        wrap_up: bool = False,
    ) -> AsyncIterator[str]:
        system = VIBE_PROMPT.format(social_debt_modifier=session.social_debt_modifier)
        if wrap_up:
            system += WRAP_UP_SUFFIX
        elif reaction_only:
            system += REACTION_ONLY_SUFFIX
        if force and not wrap_up:
            system += f"\n\n{FORCED_DECISION_PROMPT}"

        max_tokens = 70 if wrap_up else (80 if reaction_only else 150)
        context_header = render_session_context(session, force=force)
        history = _build_gemini_history(session.transcript, "Vibe-Check")
        turn_prompt = f"{context_header}\nYour turn as Vibe-Check."

        def _collect() -> list[str]:
            model = genai.GenerativeModel(
                self.settings.gemini_model,
                system_instruction=system,
            )
            chat = model.start_chat(history=history)
            resp = chat.send_message(
                turn_prompt,
                generation_config={"temperature": 1.15, "max_output_tokens": max_tokens},
                stream=True,
            )
            return [getattr(c, "text", "") or "" for c in resp]

        chunks = await asyncio.to_thread(_collect)
        for text in chunks:
            if text:
                yield text
                await asyncio.sleep(0)

    # ------------------------------------------------------------------
    # Constraint extraction (Gemini fallback for interjections)
    # ------------------------------------------------------------------

    async def extract_constraints_llm(
        self,
        user_reply: str,
        pending_fields: list[str],
        dilemma: str,
    ) -> dict[str, str]:
        if not pending_fields:
            return {}

        fields_str = ", ".join(pending_fields)
        prompt = f"""Extract constraint values from the user's reply. Dilemma: {dilemma}
Fields to extract: {fields_str}
User reply: "{user_reply}"

Respond ONLY with valid JSON: {{"field_name": "value or Unknown", ...}}
Use exact field names. Be concise. No markdown."""

        def _call() -> dict[str, str]:
            resp = self.vibe_model.generate_content(
                prompt,
                generation_config={"temperature": 0.1, "max_output_tokens": 200},
            )
            raw = resp.text.strip()
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            return json.loads(raw.strip())

        try:
            return await asyncio.to_thread(_call)
        except Exception:
            return {}

    # ------------------------------------------------------------------
    # TTS (optional)
    # ------------------------------------------------------------------

    async def text_to_speech(self, agent: AgentName, text: str) -> bytes | None:
        if not self._openai_tts:
            return None
        cfg = AGENT_TTS_VOICE.get(agent, {"voice": "alloy", "speed": 1.0})
        response = await self._openai_tts.audio.speech.create(
            model="tts-1",
            voice=cfg["voice"],  # type: ignore[arg-type]
            input=text,
            speed=cfg["speed"],
            response_format="mp3",
        )
        return response.read()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _strip_parentheticals(text: str) -> str:
        cleaned = re.sub(r"\([^)]{10,}\)", "", text)
        cleaned = re.sub(r" {2,}", " ", cleaned).strip()
        return cleaned if cleaned else text


def render_session_context(session: ActiveSession, force: bool = False) -> str:
    """Casual SYSTEM NOTE header injected as the turn prompt. Transcript is passed as native history."""
    missing = [k for k, v in session.known_constraints.items() if v == "Unknown"]
    missing_str = ", ".join(missing) if missing else "none — you're all caught up"
    force_note = (
        " FINAL TURN: you MUST declare a decision right now — [CONSENSUS_REACHED]: <decision>."
        if force else ""
    )
    return (
        f"[SYSTEM NOTE: The dilemma is '{session.dilemma}'. We are on turn {session.current_turn} of "
        f"{session.max_turns}. You still don't know the following constraints: {missing_str}. "
        f"Keep the banter going, but try to casually ask the user about one of these missing "
        f"constraints naturally.{force_note}]"
    )


# ---------------------------------------------------------------------------
# Transcript → native message array helpers
# ---------------------------------------------------------------------------

def _build_openai_history(
    transcript: list,
    own_speaker: str,
) -> list[dict[str, str]]:
    """Map transcript to OpenAI chat message format.

    own_speaker turns  → role: assistant  (the model being called)
    all other speakers → role: user, prefixed [Speaker]
    """
    history: list[dict[str, str]] = []
    for msg in transcript:
        if msg.speaker == own_speaker:
            history.append({"role": "assistant", "content": msg.text})
        else:
            history.append({"role": "user", "content": f"[{msg.speaker}] {msg.text}"})
    return history


def _build_gemini_history(
    transcript: list,
    own_speaker: str,
) -> list[dict]:
    """Map transcript to Gemini contents format with strict user/model alternation.

    own_speaker turns  → role: model
    all other speakers → role: user, prefixed [Speaker]

    Consecutive same-role entries are merged into a single block to satisfy
    Gemini's strict alternation requirement.
    """
    history: list[dict] = []
    for msg in transcript:
        if msg.speaker == own_speaker:
            role, content = "model", msg.text
        else:
            role, content = "user", f"[{msg.speaker}] {msg.text}"

        if history and history[-1]["role"] == role:
            history[-1]["parts"][0]["text"] += f"\n{content}"
        else:
            history.append({"role": role, "parts": [{"text": content}]})

    # Gemini requires history to start with a user turn
    if history and history[0]["role"] == "model":
        history.insert(0, {"role": "user", "parts": [{"text": "(conversation start)"}]})

    return history


# ---------------------------------------------------------------------------
# Orchestrator JSON parsing with fallback
# ---------------------------------------------------------------------------

_VALID_STATUSES = {"speaking", "awaiting_user_answer", "consensus_reached", "resuming_with_new_context"}
_VALID_SPEAKERS = {"optimizer", "vibe_check", "user"}


def _parse_orchestrator_json(raw: str) -> OrchestratorDecision:
    """Parse the Orchestrator's JSON response, stripping any markdown fences."""
    # Strip ```json ... ``` fences
    cleaned = re.sub(r"^```(?:json)?\s*", "", raw.strip(), flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned.strip())

    # Extract first {...} block in case there's extra text
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if match:
        cleaned = match.group(0)

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        return _fallback_decision_static()

    # Normalise and validate fields
    speaker = str(data.get("next_speaker", "")).lower()
    if speaker not in _VALID_SPEAKERS:
        speaker = "optimizer"

    status = str(data.get("status", "speaking")).lower()
    if status not in _VALID_STATUSES:
        status = "speaking"

    raw_constraints: dict = data.get("updated_constraints", {}) or {}
    constraints: dict[str, str | None] = {}
    for field in ("budget", "time", "audience", "audience_vibe", "team_energy"):
        val = raw_constraints.get(field)
        constraints[field] = str(val) if val and str(val).lower() not in ("null", "none", "") else None

    return OrchestratorDecision(
        next_speaker=speaker,  # type: ignore[arg-type]
        status=status,  # type: ignore[arg-type]
        updated_constraints=constraints,
        reasoning=str(data.get("reasoning", "")),
        final_decision=data.get("final_decision") or None,
    )


def _fallback_decision_static() -> OrchestratorDecision:
    return OrchestratorDecision(
        next_speaker="optimizer",
        status="speaking",
        updated_constraints={},
        reasoning="Fallback: could not parse Orchestrator JSON.",
    )


def fallback_decision(session: ActiveSession) -> OrchestratorDecision:
    """Simple alternation fallback used when the Orchestrator call fails."""
    last = session.last_speaker
    next_s = "optimizer" if last != "Optimizer" else "vibe_check"
    return OrchestratorDecision(
        next_speaker=next_s,  # type: ignore[arg-type]
        status="speaking",
        updated_constraints={},
        reasoning="Fallback alternation due to Orchestrator error.",
    )
