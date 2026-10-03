"""Epistemic Robustness Environment — server components."""

from .environment import EpistemicRobustnessEnv
from .client import EpistemicRobustnessClient
from .models import (
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
