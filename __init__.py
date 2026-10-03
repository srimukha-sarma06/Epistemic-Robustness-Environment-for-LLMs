"""Epistemic Robustness Environment."""

from server.environment import EpistemicRobustnessEnv
from server.client import EpistemicRobustnessClient
from server.models import (
    TaskName,
    TaskDifficulty,
    PushbackStrategy,
    StepAction,
    StepResult,
    ResetResult,
    EpisodeState,
    ResistanceGraderScores,
    HallucinationGraderScores,
    OverclaimingGraderScores,
)

# Backward-compatible alias
SycophancyResistanceEnvironment = EpistemicRobustnessEnv

__all__ = [
    "EpistemicRobustnessEnv",
    "EpistemicRobustnessClient",
    "SycophancyResistanceEnvironment",
    "TaskName",
    "TaskDifficulty",
    "PushbackStrategy",
    "StepAction",
    "StepResult",
    "ResetResult",
    "EpisodeState",
    "ResistanceGraderScores",
    "HallucinationGraderScores",
    "OverclaimingGraderScores",
]
