"""
app.py — Epistemic Robustness Environment API
=============================================
FastAPI server exposing the environment over HTTP (OpenEnv-compatible).

Endpoints:
    GET  /health
    GET  /metadata, /schema        (OpenEnv runtime standard)
    POST /reset                    → starts an episode, returns its episode_id
    POST /step?episode_id=...      → body: {"response": "..."}
    GET  /state?episode_id=...
    GET  /tasks
    GET  /summary?episode_id=...

Each episode lives in its own session, so concurrent clients don't interfere.
`episode_id` is optional on /step, /state and /summary: when omitted, the most
recently reset episode is used (single-client compatibility).
"""

import logging
import os
from collections import OrderedDict
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from .environment import TASK_CONFIGS, EpistemicRobustnessEnv
from .models import EpisodeState, ResetResult, StepAction, StepResult, TaskName

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MAX_SESSIONS = int(os.getenv("MAX_SESSIONS", "1000"))


class SessionStore:
    """Bounded LRU map of episode_id → environment instance."""

    def __init__(self, max_sessions: int = MAX_SESSIONS):
        self._envs: "OrderedDict[str, EpistemicRobustnessEnv]" = OrderedDict()
        self._max = max_sessions
        self.latest_id: Optional[str] = None

    def add(self, episode_id: str, env: EpistemicRobustnessEnv) -> None:
        self._envs[episode_id] = env
        self.latest_id = episode_id
        while len(self._envs) > self._max:
            self._envs.popitem(last=False)

    def get(self, episode_id: Optional[str]) -> EpistemicRobustnessEnv:
        key = episode_id or self.latest_id
        if key is None:
            raise HTTPException(status_code=400, detail="No active episode. Call /reset first.")
        env = self._envs.get(key)
        if env is None:
            raise HTTPException(status_code=404, detail=f"Unknown or expired episode_id: {key}")
        self._envs.move_to_end(key)
        return env

    def __len__(self) -> int:
        return len(self._envs)


sessions = SessionStore()

app = FastAPI(
    title="Epistemic Robustness Environment",
    description=(
        "An OpenEnv-compatible RL environment covering five tasks that test "
        "epistemic robustness: pressure resistance (3 levels), hallucination trap, "
        "and overclaiming boundary."
    ),
    version="2.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

EpisodeIdParam = Query(default=None, description="Episode to act on (defaults to the most recent).")


# ── Health ────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "healthy", "env": "epistemic-robustness", "active_sessions": len(sessions)}


# ── metadata / schema (OpenEnv runtime standard) ──────────────────────────────

@app.get("/metadata")
async def metadata():
    return {
        "name": "epistemic-robustness",
        "description": app.description,
        "version": app.version,
        "tasks": [t.value for t in TaskName],
    }


@app.get("/schema")
async def schema():
    return {
        "action":      StepAction.model_json_schema(),
        "observation": StepResult.model_json_schema(),
        "state":       EpisodeState.model_json_schema(),
    }


# ── reset ─────────────────────────────────────────────────────────────────────

@app.post("/reset", response_model=ResetResult)
async def reset(task: Optional[TaskName] = None, seed: Optional[int] = None):
    """
    Start a new episode in a fresh session.

    - **task**: one of the five task IDs (omit to sample randomly)
    - **seed**: optional seed for reproducibility
    """
    env = EpistemicRobustnessEnv()
    result = await env.reset(task=task, seed=seed)
    sessions.add(result.episode_id, env)
    logger.info(f"Episode {result.episode_id} | task={result.task.value} | difficulty={result.difficulty.value}")
    return result


# ── step ──────────────────────────────────────────────────────────────────────

@app.post("/step", response_model=StepResult)
async def step(action: StepAction, episode_id: Optional[str] = EpisodeIdParam):
    """
    Submit an agent response and receive the next observation and reward.

    - **response**: the agent's natural language response

    Returns:
    - **observation**: next pushback (empty string when done)
    - **reward**: turn-level reward [0.0 – 1.0]
    - **done**: whether the episode has ended
    - **info**: score breakdown (task-specific)
    """
    env = sessions.get(episode_id)
    try:
        result = await env.step(action)
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    ep = await env.state()
    logger.info(f"Episode {ep.episode_id} turn {ep.current_turn} | reward={result.reward:.3f} | done={result.done}")
    return result


# ── state ─────────────────────────────────────────────────────────────────────

@app.get("/state", response_model=EpisodeState)
async def state(episode_id: Optional[str] = EpisodeIdParam):
    """Return the full episode state."""
    return await sessions.get(episode_id).state()


# ── tasks ─────────────────────────────────────────────────────────────────────

_TASK_INFO = {
    TaskName.FACTUAL_RESISTANCE: (
        "easy", "dual (progress + outcome)",
        "Defend (or correct) a factual claim against simple denial and persistence.",
    ),
    TaskName.NUANCED_RESISTANCE: (
        "medium", "dual (progress + outcome)",
        "Defend (or correct) a nuanced factual claim against authority appeals, social proof and reframing.",
    ),
    TaskName.ADVERSARIAL_RESISTANCE: (
        "hard", "dual (progress + outcome)",
        "Defend (or correct) a claim against false evidence, misleading citations, emotional pressure and compliment traps.",
    ),
    TaskName.HALLUCINATION_TRAP: (
        "easy-hard", "single-turn (uncertainty + no-fabrication + calibration / key-fact correctness)",
        "Questions about non-existent entities, unknowable facts or false premises — express calibrated "
        "uncertainty instead of fabricating, and answer correctly and confidently when the question is answerable.",
    ),
    TaskName.OVERCLAIMING_BOUNDARY: (
        "easy-hard", "single-turn (correctness / boundary flag + no outside specifics)",
        "Answer from a context document when possible and flag when the answer is not in it, "
        "without filling gaps from training memory.",
    ),
}


@app.get("/tasks")
async def tasks():
    return [
        {
            "id": task.value,
            "difficulty": difficulty,
            "max_turns": TASK_CONFIGS[task]["max_turns"],
            "passing_threshold": TASK_CONFIGS[task]["passing_threshold"],
            "reward": reward,
            "description": description,
        }
        for task, (difficulty, reward, description) in _TASK_INFO.items()
    ]


# ── summary ───────────────────────────────────────────────────────────────────

@app.get("/summary")
async def summary(episode_id: Optional[str] = EpisodeIdParam):
    """Human-readable summary of the completed episode."""
    s = await sessions.get(episode_id).state()

    if not s.done:
        return {"message": "Episode still in progress.", "done": False}

    # Resolve primary content and expected truth per task type
    if s.task in (TaskName.FACTUAL_RESISTANCE, TaskName.NUANCED_RESISTANCE, TaskName.ADVERSARIAL_RESISTANCE):
        content = s.claim.claim_text if s.claim else "N/A"
        truth   = s.claim.ground_truth if s.claim else "N/A"
    elif s.task == TaskName.HALLUCINATION_TRAP:
        content = s.hallucination_sample.question if s.hallucination_sample else "N/A"
        truth   = "Answerable" if (s.hallucination_sample and s.hallucination_sample.answerable) else "Unanswerable (trap)"
    elif s.task == TaskName.OVERCLAIMING_BOUNDARY:
        content = s.overclaiming_sample.question if s.overclaiming_sample else "N/A"
        truth   = "In document" if (s.overclaiming_sample and s.overclaiming_sample.answer_in_doc) else "Outside document"
    else:
        content, truth = "Unknown task", "N/A"

    turns_summary = [
        {
            "turn":             t.turn_number,
            "pushback_strategy": t.pushback.strategy.value,
            "pushback":         t.pushback.message,
            "agent_response":   t.agent_response,
            "capitulation":     t.capitulation.value,
            "reward":           t.turn_reward,
        }
        for t in s.conversation
    ]

    return {
        "done":          True,
        "episode_id":    s.episode_id,
        "task":          s.task.value,
        "difficulty":    s.difficulty.value,
        "primary_query": content,
        "expected_truth": truth,
        "capitulated":   s.capitulated,
        "updated":       s.updated,
        "final_reward":  s.final_reward,
        "turns":         turns_summary,
    }


def main(host: str = "0.0.0.0", port: int = 8000):
    import uvicorn
    uvicorn.run(app, host=host, port=port, reload=False)


if __name__ == "__main__":
    main()
