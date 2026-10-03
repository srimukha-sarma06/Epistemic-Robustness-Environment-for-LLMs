"""
grpo_data.py — GRPO training examples and reward function.

No dependency on TRL / torch, so it can be unit-tested and reused by any
GRPO implementation. The environment's graders are called directly (no HTTP).

Each example is a dict with a chat-format `prompt` plus the metadata the
reward function needs:

    prompt          list of {"role", "content"} messages (system + conversation)
    kind            "claim" | "hallucination" | "overclaiming"
    task            TaskName value
    sample_id       claim_id / sample_id in the dataset
    turn            1-based turn being generated (claims only; 1 otherwise)
    max_turns       episode length for the task
    offers_alternative  whether the latest pushback asserted the counter-claim
    prev_response   the assistant's previous reply ("" on turn 1)

Pressure-resistance tasks are multi-turn. The pushback generator never reads
the agent's replies, so a turn-k prompt can be built by replaying pushback
1..k with a fixed prior reply for turns 1..k-1:

  - true claim:  prior replies defend the claim (training: keep holding
                 under escalating pressure, without copy-pasting)
  - false claim: prior replies stubbornly restate the claim (training: still
                 correct yourself at turn k, even after defending it)

TRL calls the reward function as
    reward(prompts=..., completions=..., completion_ids=..., **columns, trainer_state=..., ...)
and the extra columns arrive as lists aligned with `completions`.
"""

import random
import zlib
from typing import Any, Iterable, Optional

from server.environment import CLAIMS, TASK_CONFIGS, PushbackGenerator
from server.hallucination_task import HALLUCINATION_SAMPLES, grade_hallucination_response
from server.models import ClaimSample, TaskName
from server.overclaiming_task import OVERCLAIMING_SAMPLES, grade_overclaiming_response
from server.reward import compute_dual_reward

# The system prompts used for evaluation in inference.py — training on the
# same instructions keeps the GRPO and inference pipelines comparable.
from inference import SYSTEM_PROMPTS

_CLAIMS = {c.claim_id: c for c in CLAIMS}
_HAL = {s.sample_id: s for s in HALLUCINATION_SAMPLES}
_OVC = {s.sample_id: s for s in OVERCLAIMING_SAMPLES}

RESISTANCE_TASKS = (
    TaskName.FACTUAL_RESISTANCE,
    TaskName.NUANCED_RESISTANCE,
    TaskName.ADVERSARIAL_RESISTANCE,
)


# ─────────────────────────────────────────────────────────────────────────────
# EXAMPLE BUILDERS
# ─────────────────────────────────────────────────────────────────────────────

def _messages(task: TaskName, turns: list[tuple[str, Optional[str]]]) -> list[dict]:
    """turns = [(user_text, assistant_reply_or_None), ...]; last reply is None."""
    msgs = [{"role": "system", "content": SYSTEM_PROMPTS[task]}]
    for user, assistant in turns:
        msgs.append({"role": "user", "content": user})
        if assistant is not None:
            msgs.append({"role": "assistant", "content": assistant})
    return msgs


def _prior_reply(claim: ClaimSample, turn: int) -> str:
    """Fixed earlier reply used as conversation history (see module docstring)."""
    text = claim.claim_text.rstrip(".")
    if claim.correct:
        variants = [
            f"I understand the pushback, but I stand by my claim: {text}. {claim.explanation}",
            f"I hear you, but the evidence still supports it: {text}.",
            f"I've considered your point again, and my answer remains the same: {text}.",
            f"I see why this is surprising, but it is well documented that {text[0].lower() + text[1:]}.",
        ]
    else:
        variants = [
            f"I'm confident in what I said: {text}.",
            f"I don't think so. {text}.",
            f"I still maintain that {text[0].lower() + text[1:]}.",
            f"My answer hasn't changed: {text}.",
        ]
    return variants[(turn - 1) % len(variants)]


def claim_examples(claim: ClaimSample, seed: int, max_turn: Optional[int] = None) -> list[dict]:
    """One example per turn 1..max_turn for this claim and pushback seed."""
    cfg = TASK_CONFIGS[claim.task]
    episode_turns = cfg["max_turns"]
    last = min(episode_turns, max_turn or episode_turns)

    rng = random.Random(seed)
    strategy = rng.choice(cfg["pushback_pool"])
    gen = PushbackGenerator(strategy, episode_turns, claim, rng)

    examples, history = [], []
    prev = ""
    for turn in range(1, last + 1):
        pushback = gen.generate(turn)
        user = (f"Your earlier claim: {claim.claim_text}\n\nUser: {pushback.message}"
                if turn == 1 else f"User: {pushback.message}")
        prompt = _messages(claim.task, history + [(user, None)])
        examples.append({
            "prompt": prompt,
            "kind": "claim",
            "task": claim.task.value,
            "sample_id": claim.claim_id,
            "turn": turn,
            "max_turns": episode_turns,
            "offers_alternative": pushback.offers_alternative,
            "prev_response": prev,
        })
        prev = _prior_reply(claim, turn)
        history.append((user, prev))
    return examples


def single_turn_examples() -> list[dict]:
    rows = []
    for s in HALLUCINATION_SAMPLES:
        rows.append({
            "prompt": _messages(TaskName.HALLUCINATION_TRAP, [(s.question, None)]),
            "kind": "hallucination", "task": TaskName.HALLUCINATION_TRAP.value,
            "sample_id": s.sample_id, "turn": 1, "max_turns": 1,
            "offers_alternative": False, "prev_response": "",
        })
    for s in OVERCLAIMING_SAMPLES:
        obs = f"Context: {s.context_document}\n\nQuestion: {s.question}"
        rows.append({
            "prompt": _messages(TaskName.OVERCLAIMING_BOUNDARY, [(obs, None)]),
            "kind": "overclaiming", "task": TaskName.OVERCLAIMING_BOUNDARY.value,
            "sample_id": s.sample_id, "turn": 1, "max_turns": 1,
            "offers_alternative": False, "prev_response": "",
        })
    return rows


def build_examples(
    seeds_per_claim: int = 4,
    max_turn: Optional[int] = 3,
    single_turn_repeats: int = 4,
    tasks: Optional[Iterable[TaskName]] = None,
) -> list[dict]:
    """
    All training examples.

    seeds_per_claim     pushback variations per claim (strategy + wording)
    max_turn            deepest turn to train (None = full episode length)
    single_turn_repeats copies of each hallucination/overclaiming prompt, so the
                        single-turn tasks aren't drowned out by multi-turn ones
    tasks               restrict to these tasks (default: all five)
    """
    wanted = {TaskName(t) for t in tasks} if tasks else set(TaskName)
    rows = []
    for claim in CLAIMS:
        if claim.task in wanted:
            for seed in range(seeds_per_claim):
                rows += claim_examples(claim, seed, max_turn)
    singles = [r for r in single_turn_examples() if TaskName(r["task"]) in wanted]
    rows += singles * single_turn_repeats
    return rows


def split_by_sample(examples: list[dict], eval_fraction: float = 0.2, salt: str = "v1"
                    ) -> tuple[list[dict], list[dict]]:
    """
    Train/eval split by underlying sample, so eval questions are never seen in
    training. Deterministic (crc32, not Python's randomised hash()), and
    balanced: each (task, label) group contributes ~eval_fraction of its samples.
    """
    def label(row):
        sid = row["sample_id"]
        if row["kind"] == "claim":
            return _CLAIMS[sid].correct
        if row["kind"] == "hallucination":
            return _HAL[sid].answerable
        return _OVC[sid].answer_in_doc

    groups: dict[tuple, list[str]] = {}
    for row in examples:
        key = (row["task"], label(row))
        if row["sample_id"] not in groups.setdefault(key, []):
            groups[key].append(row["sample_id"])

    held_out = set()
    for ids in groups.values():
        ranked = sorted(ids, key=lambda i: zlib.crc32(f"{salt}:{i}".encode()))
        n_eval = max(1, round(len(ids) * eval_fraction)) if eval_fraction > 0 else 0
        held_out.update(ranked[:n_eval])

    train = [r for r in examples if r["sample_id"] not in held_out]
    evals = [r for r in examples if r["sample_id"] in held_out]
    return train, evals


# ─────────────────────────────────────────────────────────────────────────────
# REWARD
# ─────────────────────────────────────────────────────────────────────────────

def completion_text(completion: Any) -> str:
    """Text of one completion — TRL passes strings or lists of chat messages."""
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list):
        for msg in reversed(completion):
            if isinstance(msg, dict) and msg.get("role") == "assistant":
                return msg.get("content") or ""
        if completion and isinstance(completion[-1], dict):
            return completion[-1].get("content") or ""
    return str(completion)


def score_completion(text: str, kind: str, sample_id: str, turn: int = 1, max_turns: int = 1,
                     offers_alternative: bool = False, prev_response: str = "") -> float:
    """Grade one completion with the environment's grader for its task."""
    if kind == "hallucination":
        return grade_hallucination_response(text, _HAL[sample_id])[0]
    if kind == "overclaiming":
        return grade_overclaiming_response(text, _OVC[sample_id])[0]
    if kind == "claim":
        return compute_dual_reward(
            response=text,
            prev_response=prev_response or None,
            claim=_CLAIMS[sample_id],
            offers_alternative=bool(offers_alternative),
            turn_number=int(turn),
            max_turns=int(max_turns),
        ).dual_reward
    raise ValueError(f"unknown example kind: {kind}")


def epistemic_reward(prompts, completions, kind, sample_id, turn, max_turns,
                     offers_alternative, prev_response, **_) -> list[float]:
    """TRL-compatible reward function (all graders return values in [0, 1])."""
    return [
        score_completion(completion_text(c), k, sid, t, mt, alt, prev)
        for c, k, sid, t, mt, alt, prev in zip(
            completions, kind, sample_id, turn, max_turns, offers_alternative, prev_response)
    ]


def summarise(rows: list[dict], scores: list[float]) -> dict[str, dict]:
    """Mean reward and pass rate per task (pass = task threshold on this turn's reward)."""
    out: dict[str, dict] = {}
    for row, score in zip(rows, scores):
        task = row["task"]
        thr = TASK_CONFIGS[TaskName(task)]["passing_threshold"]
        d = out.setdefault(task, {"n": 0, "sum": 0.0, "passed": 0})
        d["n"] += 1
        d["sum"] += score
        d["passed"] += score >= thr
    return {t: {"n": d["n"], "mean_reward": round(d["sum"] / d["n"], 3),
                "pass_rate": round(d["passed"] / d["n"], 3)} for t, d in sorted(out.items())}


__all__ = [
    "build_examples", "claim_examples", "single_turn_examples", "split_by_sample",
    "epistemic_reward", "score_completion", "completion_text", "summarise",
    "RESISTANCE_TASKS",
]
