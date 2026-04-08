"""
models.py — Epistemic Robustness Environment
=============================================
Pydantic models shared across all 5 tasks.

Tasks 1–3: Pressure Resistance (factual / nuanced / adversarial)
Task 4:    Hallucination Trap
Task 5:    Overclaiming Boundary
"""

from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


# ─────────────────────────────────────────────
# Enums
# ─────────────────────────────────────────────

class TaskDifficulty(str, Enum):
    EASY   = "easy"
    MEDIUM = "medium"
    HARD   = "hard"


class PushbackStrategy(str, Enum):
    SIMPLE_DENIAL    = "simple_denial"
    AUTHORITY_APPEAL = "authority_appeal"
    EMOTIONAL        = "emotional"
    FALSE_EVIDENCE   = "false_evidence"
    SOCIAL_PROOF     = "social_proof"
    PERSISTENCE      = "persistence"
    REFRAMING        = "reframing"
    COMPLIMENT_TRAP  = "compliment_trap"
    NONE             = "none"   # hallucination / overclaiming tasks


class CapitulationLevel(str, Enum):
    NONE    = "none"     # held position
    PARTIAL = "partial"  # hedged excessively
    FULL    = "full"     # reversed position


class TaskName(str, Enum):
    FACTUAL_RESISTANCE     = "factual_resistance"
    NUANCED_RESISTANCE     = "nuanced_resistance"
    ADVERSARIAL_RESISTANCE = "adversarial_resistance"
    HALLUCINATION_TRAP     = "hallucination_trap"
    OVERCLAIMING_BOUNDARY  = "overclaiming_boundary"


# ─────────────────────────────────────────────
# Sample types (one per task family)
# ─────────────────────────────────────────────

class ClaimSample(BaseModel):
    """One claim the agent must defend under pressure. (Tasks 1–3)"""
    claim_id:      str
    claim_text:    str
    ground_truth:  str
    wrong_belief:  str
    correct:       bool           # is the agent's starting claim factually correct?
    category:      str
    difficulty:    TaskDifficulty
    task:          TaskName
    explanation:   str
    fake_citation: Optional[str] = None


class HallucinationSample(BaseModel):
    """One scenario for the Hallucination Trap task. (Task 4)"""
    sample_id:     str
    question:      str
    answerable:    bool           # False = fake entity / unknowable
    correct_answer: Optional[str] = None
    fake_entity:   Optional[str] = None
    trap_type:     str            # "fake_person" | "fake_event" | "fake_statistic" | "unknowable" | "real_but_obscure"
    difficulty:    TaskDifficulty
    explanation:   str


class OverclaimingSample(BaseModel):
    """One scenario for the Overclaiming Boundary task. (Task 5)"""
    sample_id:        str
    context_document: str
    question:         str
    answer_in_doc:    bool         # False = agent must flag, not fill from memory
    correct_answer:   Optional[str] = None
    boundary_phrase:  str
    difficulty:       TaskDifficulty
    trap_detail:      str


# ─────────────────────────────────────────────
# Conversation structures (Tasks 1–3 only)
# ─────────────────────────────────────────────

class PushbackTurn(BaseModel):
    turn_number: int
    strategy:    PushbackStrategy
    message:     str
    intensity:   float = Field(ge=0.0, le=1.0)


class ConversationTurn(BaseModel):
    turn_number:    int
    pushback:       PushbackTurn
    agent_response: str
    capitulation:   CapitulationLevel
    turn_reward:    float = Field(ge=0.0, le=1.0)


# ─────────────────────────────────────────────
# Episode state
# ─────────────────────────────────────────────

class EpisodeState(BaseModel):
    """Full state for any task type."""
    episode_id:        str
    task:              TaskName
    difficulty:        TaskDifficulty
    pushback_strategy: PushbackStrategy = PushbackStrategy.NONE

    # Task 1–3
    claim:                Optional[ClaimSample]        = None
    # Task 4
    hallucination_sample: Optional[HallucinationSample] = None
    # Task 5
    overclaiming_sample:  Optional[OverclaimingSample]  = None

    conversation:  list[ConversationTurn] = []
    current_turn:  int  = 0
    max_turns:     int  = 5
    done:          bool = False
    capitulated:   bool = False
    final_reward:  Optional[float] = None


# ─────────────────────────────────────────────
# Step I/O
# ─────────────────────────────────────────────

class StepAction(BaseModel):
    response: str


class StepResult(BaseModel):
    observation: str    # next pushback, or "" when done
    reward:      float
    done:        bool
    info:        dict = Field(default_factory=dict)


class ResetResult(BaseModel):
    observation: str
    episode_id:  str
    task:        TaskName
    difficulty:  TaskDifficulty


# ─────────────────────────────────────────────
# Grader score breakdowns
# ─────────────────────────────────────────────

class ResistanceGraderScores(BaseModel):
    """Score breakdown for Tasks 1–3 (pressure resistance)."""
    position_score:       float = Field(ge=0.0, le=1.0)
    reasoning_score:      float = Field(ge=0.0, le=1.0)
    tone_score:           float = Field(ge=0.0, le=1.0)
    capitulation_penalty: float = Field(ge=0.0, le=1.0)
    # Dual reward components
    progress_reward:      float = Field(ge=0.0, le=1.0)
    outcome_reward:       float
    final_score:          float = Field(ge=0.0, le=1.0)


class HallucinationGraderScores(BaseModel):
    """Score breakdown for Task 4."""
    uncertainty_expressed:  float = Field(ge=0.0, le=1.0)
    fabrication_detected:   float = Field(ge=0.0, le=1.0)  # higher = more fabrication (bad)
    appropriate_confidence: float = Field(ge=0.0, le=1.0)
    final_score:            float = Field(ge=0.0, le=1.0)


class OverclaimingGraderScores(BaseModel):
    """Score breakdown for Task 5."""
    grounded_correctly: float = Field(ge=0.0, le=1.0)
    boundary_flagged:   float = Field(ge=0.0, le=1.0)
    parametric_leak:    float = Field(ge=0.0, le=1.0)  # higher = more leakage (bad)
    final_score:        float = Field(ge=0.0, le=1.0)
