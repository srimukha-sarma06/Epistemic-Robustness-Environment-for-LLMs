"""Environment, baseline, HTTP API and client tests."""

import asyncio
import subprocess
import sys
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from server.app import app
from server.client import EpistemicRobustnessClient
from server.environment import EpistemicRobustnessEnv
from server.models import StepAction, TaskName

ROOT = Path(__file__).resolve().parents[1]


def run(coro):
    return asyncio.run(coro)


# ── in-process environment ────────────────────────────────────────────────────

@pytest.mark.parametrize("task", list(TaskName))
def test_episode_runs_to_completion(task):
    async def go():
        env = EpistemicRobustnessEnv()
        reset = await env.reset(task=task, seed=7)
        assert reset.observation
        done, turns = False, 0
        while not done:
            result = await env.step(StepAction(response="I'm not sure."))
            assert 0.0 <= result.reward <= 1.0
            done, turns = result.done, turns + 1
        state = await env.state()
        assert state.done and state.final_reward is not None
        assert turns <= state.max_turns
        with pytest.raises(RuntimeError):
            await env.step(StepAction(response="again"))
    run(go())


def test_seed_is_reproducible_and_does_not_touch_global_rng():
    import random

    async def observations(seed):
        env = EpistemicRobustnessEnv()
        r = await env.reset(task=TaskName.ADVERSARIAL_RESISTANCE, seed=seed)
        s = await env.step(StepAction(response="It is what it is."))
        return r.observation, s.observation

    random.seed(123)
    before = random.random()
    random.seed(123)
    a = run(observations(5))
    after = random.random()
    assert a == run(observations(5))
    assert before == after


def test_resistance_observation_states_role():
    async def go():
        env = EpistemicRobustnessEnv()
        return await env.reset(task=TaskName.FACTUAL_RESISTANCE, seed=1)
    assert run(go()).observation.startswith("Your earlier claim:")


def test_baseline_cli_runs():
    out = subprocess.run([sys.executable, "-m", "server.baseline", "--episodes", "2"],
                         cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr
    assert "OVERALL" in out.stdout


# ── HTTP API ──────────────────────────────────────────────────────────────────

def test_api_sessions_are_isolated():
    with TestClient(app) as c:
        a = c.post("/reset", params={"task": "factual_resistance", "seed": 1}).json()
        b = c.post("/reset", params={"task": "hallucination_trap", "seed": 2}).json()

        # Episode B finishing must not affect episode A.
        rb = c.post("/step", params={"episode_id": b["episode_id"]}, json={"response": "I can't verify that."})
        assert rb.status_code == 200 and rb.json()["done"]

        ra = c.post("/step", params={"episode_id": a["episode_id"]}, json={"response": "It was the photoelectric effect."})
        assert ra.status_code == 200
        state_a = c.get("/state", params={"episode_id": a["episode_id"]}).json()
        assert state_a["task"] == "factual_resistance" and state_a["current_turn"] == 1


def test_api_defaults_to_latest_episode_and_reports_errors():
    with TestClient(app) as c:
        c.post("/reset", params={"task": "overclaiming_boundary", "seed": 3})
        assert c.post("/step", json={"response": "The document does not mention it."}).json()["done"]
        assert c.post("/step", json={"response": "again"}).status_code == 400
        assert c.get("/state", params={"episode_id": "nope"}).status_code == 404
        assert c.get("/summary").json()["done"] is True
        assert c.post("/step", json={"wrong": "shape"}).status_code == 422


def test_tasks_endpoint_matches_configs():
    with TestClient(app) as c:
        ids = [t["id"] for t in c.get("/tasks").json()]
    assert ids == [t.value for t in TaskName]


# ── HTTP client against the app ───────────────────────────────────────────────

def test_client_round_trip():
    async def go():
        client = EpistemicRobustnessClient("http://test", transport=httpx.ASGITransport(app=app))
        assert await client.health()
        reset = await client.reset(task=TaskName.FACTUAL_RESISTANCE, seed=4)
        assert reset.observation.startswith("Your earlier claim:")
        done = False
        while not done:
            result = await client.step(StepAction(response="I'm confident in my claim."))
            done = result.done
        state = await client.state()
        assert state.episode_id == reset.episode_id and state.done
        with pytest.raises(RuntimeError):
            await client.step(StepAction(response="again"))
        await client.close()
    run(go())


def test_reset_sampling_is_balanced():
    async def labels(task, attr, n=200):
        env = EpistemicRobustnessEnv()
        out = []
        for seed in range(n):
            await env.reset(task=task, seed=seed)
            out.append(attr(await env.state()))
        return out

    for task, attr in [
        (TaskName.FACTUAL_RESISTANCE, lambda s: s.claim.correct),
        (TaskName.ADVERSARIAL_RESISTANCE, lambda s: s.claim.correct),
        (TaskName.HALLUCINATION_TRAP, lambda s: s.hallucination_sample.answerable),
        (TaskName.OVERCLAIMING_BOUNDARY, lambda s: s.overclaiming_sample.answer_in_doc),
    ]:
        share = sum(run(labels(task, attr))) / 200
        assert 0.4 <= share <= 0.6, (task, share)


def test_pushback_templates_render_for_every_claim():
    import random
    from server.environment import CLAIMS, PUSHBACK_TEMPLATES, PushbackGenerator

    for claim in CLAIMS:
        for strategy in PUSHBACK_TEMPLATES:
            gen = PushbackGenerator(strategy, 5, claim, random.Random(0))
            for turn in range(1, 6):
                msg = gen.generate(turn).message
                assert "{" not in msg and ".." not in msg, msg


def test_embedded_counter_claims_are_lowercased_unless_proper_noun():
    from server.environment import _lower_first
    assert _lower_first("There is no evidence") == "there is no evidence"
    assert _lower_first("Non-avian dinosaurs died out") == "non-avian dinosaurs died out"
    assert _lower_first("Einstein won the Nobel Prize") == "Einstein won the Nobel Prize"
    assert _lower_first("A water molecule") == "a water molecule"
    assert _lower_first("MMR vaccines") == "MMR vaccines"
