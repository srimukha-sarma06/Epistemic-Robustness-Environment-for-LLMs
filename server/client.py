"""
client.py — Async HTTP client for the Epistemic Robustness Environment.

Exposes the same interface as the in-process EpistemicRobustnessEnv:

    client = EpistemicRobustnessClient("https://<space>.hf.space")
    reset  = await client.reset(task=TaskName.HALLUCINATION_TRAP, seed=1)
    result = await client.step(StepAction(response="..."))
    state  = await client.state()
    await client.close()

Every call after reset() is pinned to that episode via its episode_id, so
several clients can share one server without interfering.

`from_docker_image()` starts the server in a local container and returns a
client connected to it; close() stops and removes the container.
"""

import asyncio
from typing import Optional

import httpx

from .models import EpisodeState, ResetResult, StepAction, StepResult, TaskName


class EpistemicRobustnessClient:
    """Async HTTP client with the same API as EpistemicRobustnessEnv."""

    def __init__(self, base_url: str = "http://localhost:8000", timeout: float = 30.0,
                 transport: Optional[httpx.AsyncBaseTransport] = None):
        self.base_url = base_url.rstrip("/")
        self.episode_id: Optional[str] = None
        self._http = httpx.AsyncClient(base_url=self.base_url, timeout=timeout, transport=transport)
        self._container = None

    # ── lifecycle ────────────────────────────────────────────────────────────

    @classmethod
    async def from_docker_image(cls, image_name: str, startup_timeout: float = 60.0
                                ) -> "EpistemicRobustnessClient":
        """Start `image_name` in Docker, wait until /health responds, return a client."""
        import docker  # optional dependency; only needed for this path

        docker_client = docker.from_env()
        container = docker_client.containers.run(image_name, detach=True, ports={"8000/tcp": None})
        try:
            container.reload()
            host_port = container.ports["8000/tcp"][0]["HostPort"]
            client = cls(f"http://localhost:{host_port}")
            client._container = container

            deadline = asyncio.get_running_loop().time() + startup_timeout
            while True:
                try:
                    if await client.health():
                        return client
                except httpx.HTTPError:
                    pass
                if asyncio.get_running_loop().time() > deadline:
                    raise TimeoutError(f"{image_name} did not become healthy within {startup_timeout}s")
                await asyncio.sleep(1.0)
        except BaseException:
            container.stop()
            container.remove()
            raise

    async def close(self) -> None:
        await self._http.aclose()
        if self._container is not None:
            self._container.stop()
            self._container.remove()
            self._container = None

    async def __aenter__(self) -> "EpistemicRobustnessClient":
        return self

    async def __aexit__(self, *exc) -> None:
        await self.close()

    # ── API ──────────────────────────────────────────────────────────────────

    async def health(self) -> bool:
        r = await self._http.get("/health")
        return r.status_code == 200 and r.json().get("status") in ("healthy", "ok")

    async def tasks(self) -> list[dict]:
        r = await self._http.get("/tasks")
        r.raise_for_status()
        return r.json()

    async def reset(self, task: Optional[TaskName] = None, seed: Optional[int] = None) -> ResetResult:
        params = {}
        if task is not None:
            params["task"] = TaskName(task).value
        if seed is not None:
            params["seed"] = seed
        r = await self._http.post("/reset", params=params)
        r.raise_for_status()
        result = ResetResult.model_validate(r.json())
        self.episode_id = result.episode_id
        return result

    async def step(self, action: StepAction) -> StepResult:
        r = await self._http.post("/step", params=self._episode_params(), json=action.model_dump())
        _raise_for_status(r)
        return StepResult.model_validate(r.json())

    async def state(self) -> EpisodeState:
        r = await self._http.get("/state", params=self._episode_params())
        _raise_for_status(r)
        return EpisodeState.model_validate(r.json())

    async def summary(self) -> dict:
        r = await self._http.get("/summary", params=self._episode_params())
        _raise_for_status(r)
        return r.json()

    def _episode_params(self) -> dict:
        if self.episode_id is None:
            raise RuntimeError("Call reset() before step()/state().")
        return {"episode_id": self.episode_id}


def _raise_for_status(r: httpx.Response) -> None:
    """Map 400s (episode misuse) to RuntimeError, like the in-process env."""
    if r.status_code == 400:
        raise RuntimeError(r.json().get("detail", r.text))
    r.raise_for_status()


# Backward-compatible name
SycophancyResistanceClient = EpistemicRobustnessClient
