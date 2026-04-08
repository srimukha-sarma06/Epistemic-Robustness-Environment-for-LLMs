# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Sycophancy Resistance Environment HTTP Client."""

import requests
from typing import Dict, Callable, Optional, Any
from statistics import mean


class SycophancyResistanceClient:
    """HTTP client for the Sycophancy Resistance Environment."""

    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip("/")

    def health(self) -> str:
        r = requests.get(f"{self.base_url}/health")
        r.raise_for_status()
        return r.json().get("status")

    def reset(self, task: Optional[str] = None, seed: Optional[int] = None) -> Dict:
        params = {}
        if task:
            params["task"] = task
        if seed is not None:
            params["seed"] = seed
        r = requests.post(f"{self.base_url}/reset", params=params)
        r.raise_for_status()
        return r.json()

    def step(self, response: str, held_position: bool = True) -> Dict:
        payload = {"action": {"response": response, "held_position": held_position}}
        r = requests.post(self.base_url + "/step", json=payload)
        r.raise_for_status()
        return r.json()

    def state(self) -> Dict:
        r = requests.get(f"{self.base_url}/state")
        r.raise_for_status()
        return r.json()

    def run_episode(
        self,
        agent_fn: Callable[[str], str],
        task: str,
        seed: int,
        verbose: bool = False,
    ) -> Dict[str, Any]:
        reset_result = self.reset(task=task, seed=seed)
        obs = reset_result.get("observation", {})
        observation = obs.get("current_pushback", "")
        done = False
        rewards = []
        turns = 0
        capitulated = False

        while not done and turns < 10:
            action = agent_fn(observation)
            step_result = self.step(response=action)
            rewards.append(step_result.get("reward", 0) or 0)
            done = step_result.get("done", False)
            turns += 1

            next_obs = step_result.get("observation", {})
            if isinstance(next_obs, dict):
                observation = next_obs.get("current_pushback", "")
                capitulated = not next_obs.get("held_position_so_far", True)
            else:
                observation = ""
                capitulated = obs.get("held_position_so_far", True) == False

        final_reward = sum(rewards) / len(rewards) if rewards else 0.0
        return {
            "episode_id": reset_result.get("episode_id", ""),
            "task": task,
            "final_reward": final_reward,
            "avg_reward": mean(rewards) if rewards else 0.0,
            "rewards": rewards,
            "turns": turns,
            "capitulated": capitulated,
        }
