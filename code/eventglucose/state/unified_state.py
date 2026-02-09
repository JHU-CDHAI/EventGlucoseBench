"""
Unified State for DIKW Agent System
====================================

Module Structure
----------------

┌─────────────────────────────────────────────────────────────────────┐
│                    unified_state.py                                 │
│              State Models for DIKW System                           │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  SUPPORTING MODELS                                                  │
│  ┌────────────────────────────────────────────────────────────┐    │
│  │  DIKWPlan ─────────> D/I/K/W instructions + plan_id        │    │
│  │  DIKWStep ─────────> current_level + status                │    │
│  │  DIKWStepResult ───> level results + brief_summary         │    │
│  │  DIKWMetadata ─────> workspace paths configuration         │    │
│  └────────────────────────────────────────────────────────────┘    │
│                                                                     │
│  TASK MODELS                                                        │
│  ┌────────────────────────────────────────────────────────────┐    │
│  │  Task ─────────────> Unified task for D/I/K/W levels       │    │
│  │    ├── level, task_idx, task_name                          │    │
│  │    ├── task_description, task_plan                         │    │
│  │    └── execution_mode: skip/run_code/run_openhands/...     │    │
│  │                                                            │    │
│  │  TaskResult ───────> Execution result with paths           │    │
│  │    ├── success, task_code_folder, task_report_folder       │    │
│  │    └── error_message, processing_time_seconds              │    │
│  └────────────────────────────────────────────────────────────┘    │
│                                                                     │
│  ROUTING MODELS                                                     │
│  ┌────────────────────────────────────────────────────────────┐    │
│  │  AgentRecommendation ──> from_level → recommended_next     │    │
│  │  DecisionAnalysisResult ──> LLM analysis of routing        │    │
│  │  RoutingDecision ──────> Audit trail for routing           │    │
│  └────────────────────────────────────────────────────────────┘    │
│                                                                     │
│  DECISION POINT MODELS                                              │
│  ┌────────────────────────────────────────────────────────────┐    │
│  │  ResultAnalysis ───────> Analysis of task-step results     │    │
│  │  DecisionRecommendation ──> LLM recommendation             │    │
│  │  HumanResponse ────────> Human's action + feedback         │    │
│  │  DecisionPoint ────────> Complete decision record          │    │
│  └────────────────────────────────────────────────────────────┘    │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                      DIKWUnifiedState                               │
│          Main State Schema (flows through all layers)               │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Section 1: DIKW ORCHESTRATOR                                       │
│  ┌────────────────────────────────────────────────────────────┐    │
│  │  metadata ─────────> DIKWMetadata (workspace paths)        │    │
│  │  current_dikw_plan ─> DIKWPlan (D/I/K/W instructions)      │    │
│  │  d/i/k/w_results ──> DIKWStepResult per level              │    │
│  │  is_complete ──────> bool (all levels done)                │    │
│  └────────────────────────────────────────────────────────────┘    │
│                                                                     │
│  Section 2: CURRENT LEVEL CONTEXT                                   │
│  ┌────────────────────────────────────────────────────────────┐    │
│  │  current_level ────> "P"/"D"/"I"/"K"/"W"                   │    │
│  │  current_instruction ─> str (from plan)                    │    │
│  │  current_task ─────> Task (active task)                    │    │
│  └────────────────────────────────────────────────────────────┘    │
│                                                                     │
│  Section 3: TASK MANAGEMENT                                         │
│  ┌────────────────────────────────────────────────────────────┐    │
│  │  data_tasks, information_tasks, ... ─> Historical tasks    │    │
│  │  current_*_tasks ──> Tasks for current step                │    │
│  │  completed_tasks, failed_tasks ─> Tracking lists           │    │
│  └────────────────────────────────────────────────────────────┘    │
│                                                                     │
│  Section 4: TASK EXECUTION                                          │
│  ┌────────────────────────────────────────────────────────────┐    │
│  │  task_processing_results ─> List[TaskResult] (Annotated)   │    │
│  │  task_execution_error ────> str (if any)                   │    │
│  │  openhand_sdk_result ─────> List[Any] (SDK results)        │    │
│  └────────────────────────────────────────────────────────────┘    │
│                                                                     │
│  Section 5: CONTROL FLOW                                            │
│  ┌────────────────────────────────────────────────────────────┐    │
│  │  next_action ─> Literal["route_to_planner", ...]           │    │
│  │  processing_complete ─> bool                               │    │
│  │  human_approval ──────> "approve"/"skip"/"abort"           │    │
│  └────────────────────────────────────────────────────────────┘    │
│                                                                     │
│  Section 6: MESSAGES & CONVERSATION                                 │
│  ┌────────────────────────────────────────────────────────────┐    │
│  │  messages ─────────> List[AnyMessage] (Annotated reducer)  │    │
│  │  decision_mode ────> "initial"/"review_plan"/...           │    │
│  │  last_agent_response, last_human_input                     │    │
│  └────────────────────────────────────────────────────────────┘    │
│                                                                     │
│  HELPER METHODS                                                     │
│  ┌────────────────────────────────────────────────────────────┐    │
│  │  get_current_tasks() ──> List[Task]                        │    │
│  │  get_pending_tasks() ──> List[Task]                        │    │
│  │  is_level_complete() ──> bool                              │    │
│  │  generate_level_summary() ─> str                           │    │
│  └────────────────────────────────────────────────────────────┘    │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘

State Flow
----------

::

    User Code ──> DIKWAgent.run()
                        │
                        ▼
    ┌───────────────────────────────────────────────────────────┐
    │                    DIKWUnifiedState                       │
    │                  (single state object)                    │
    └───────────────────────────────────────────────────────────┘
                        │
        ┌───────────────┼───────────────┐
        ▼               ▼               ▼
    Orchestrator ──> P/D/I/K/W ──> Task Execution
                    Agents          Subgraph

    SAME STATE FLOWS THROUGH ALL LAYERS (zero transformations)
"""

from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
from langgraph.graph.message import add_messages, AnyMessage
from operator import add
from datetime import datetime
from pathlib import Path

from typing import Annotated, Literal

# =============================================================================
# SUPPORTING MODELS (unchanged from current design)
# =============================================================================

class PlanTaskItem(BaseModel):
    """
    A task item in a DIKW plan.

    This is a lightweight task specification for planning purposes.
    When executed, PlanTaskItem gets converted to full Task object with execution fields.
    """
    task_name: str = Field(
        ...,
        min_length=1,
        description="Short identifier for this task (e.g., 'understand_columns', 'extract_patterns')"
    )
    task_description: str = Field(
        ...,
        min_length=1,
        description="Detailed description of what this task should accomplish"
    )
    task_plan: str = Field(
        default="",
        description="Step-by-step execution plan (optional, can be auto-generated)"
    )
    estimated_complexity: Literal["simple", "medium", "complex"] = Field(
        default="medium",
        description="Estimated complexity for resource allocation"
    )


class DIKWPlan(BaseModel):
    """
    DIKW Plan containing task lists for all four levels.

    Each level (D, I, K, W) has a list of PlanTaskItem objects that define
    what tasks should be executed at that level. This replaces the previous
    design where each level had a single instruction string.
    """
    D: List[PlanTaskItem] = Field(
        ...,
        min_length=1,
        description="Data-level tasks: data exploration, quality assessment, etc."
    )
    I: List[PlanTaskItem] = Field(
        ...,
        min_length=1,
        description="Information-level tasks: pattern extraction, statistical analysis, etc."
    )
    K: List[PlanTaskItem] = Field(
        ...,
        min_length=1,
        description="Knowledge-level tasks: rule extraction, causal analysis, etc."
    )
    W: List[PlanTaskItem] = Field(
        ...,
        min_length=1,
        description="Wisdom-level tasks: strategic recommendations, action plans, etc."
    )

    plan_id: str = Field(..., description="Unique identifier for this plan version")
    created_at: datetime = Field(default_factory=datetime.now)
    created_by: Literal["human", "template", "agent_revision"] = Field(
        default="template", description="How this plan was created"
    )
    revision_reason: Optional[str] = Field(
        default=None, description="Why this plan was revised (if applicable)"
    )

    @classmethod
    def create_default_template(cls, project_name: str) -> 'DIKWPlan':
        """Create default exploration template with standard tasks per level"""
        return cls(
            D=[
                PlanTaskItem(
                    task_name="understand_structure",
                    task_description=f"Understand the structure and schema of all data files for {project_name}",
                    estimated_complexity="simple"
                ),
                PlanTaskItem(
                    task_name="assess_quality",
                    task_description="Assess data quality: missing values, outliers, data types",
                    estimated_complexity="simple"
                ),
                PlanTaskItem(
                    task_name="create_summary",
                    task_description="Create summary statistics and basic visualizations",
                    estimated_complexity="medium"
                ),
            ],
            I=[
                PlanTaskItem(
                    task_name="extract_patterns",
                    task_description="Extract key patterns and trends from the data",
                    estimated_complexity="medium"
                ),
                PlanTaskItem(
                    task_name="analyze_correlations",
                    task_description="Analyze correlations and relationships between variables",
                    estimated_complexity="medium"
                ),
                PlanTaskItem(
                    task_name="identify_segments",
                    task_description="Identify segments or clusters in the data",
                    estimated_complexity="complex"
                ),
            ],
            K=[
                PlanTaskItem(
                    task_name="synthesize_rules",
                    task_description="Synthesize rules and principles from information patterns",
                    estimated_complexity="medium"
                ),
                PlanTaskItem(
                    task_name="map_relationships",
                    task_description="Map causal relationships between key factors",
                    estimated_complexity="complex"
                ),
            ],
            W=[
                PlanTaskItem(
                    task_name="generate_recommendations",
                    task_description="Generate actionable recommendations based on knowledge",
                    estimated_complexity="medium"
                ),
                PlanTaskItem(
                    task_name="create_action_plan",
                    task_description="Create prioritized action plan with implementation roadmap",
                    estimated_complexity="complex"
                ),
            ],
            plan_id=f"template_{project_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            created_by="template"
        )

    def get_level_tasks(self, level: str) -> List[PlanTaskItem]:
        """Get tasks for a specific level"""
        level_map = {"D": self.D, "I": self.I, "K": self.K, "W": self.W}
        return level_map.get(level, [])

    def get_level_instruction(self, level: str) -> str:
        """
        Get a combined instruction string for a level (for backward compatibility).

        This concatenates all task descriptions into a single instruction string.
        Used during transition period while updating all consumers.
        """
        tasks = self.get_level_tasks(level)
        if not tasks:
            return ""
        return " | ".join([f"({i+1}) {t.task_description}" for i, t in enumerate(tasks)])



class DIKWStep(BaseModel):
    """Current step being executed in the DIKW workflow"""
    current_level: Literal["P", "D", "I", "K", "W"] = Field(
        ..., description="Current level being processed"
    )
    current_instruction: str = Field(
        ..., description="Current instruction being executed"
    )
    status: Literal["pending", "in_progress", "completed", "failed"] = Field(
        default="pending", description="Status of current step"
    )
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None


class DIKWStepResult(BaseModel):
    """Result from processing one DIKW level"""
    level: Literal["P", "D", "I", "K", "W"]
    success: bool

    # Success results
    tasks: List[str] = Field(default_factory=list)
    reports_path: Optional[Path] = None
    code_path: Optional[Path] = None
    brief_summary: str = ""

    # LLM-powered intelligent analysis
    llm_analysis: Optional[str] = None
    key_accomplishments: List[str] = Field(default_factory=list)
    issues_found: List[str] = Field(default_factory=list)
    quality_assessment: Optional[Literal["excellent", "good", "needs_improvement", "poor"]] = None
    recommendations_for_next_level: Optional[str] = None

    # Failure results
    blocked_reason: Optional[str] = None
    suggestions_for_plan_revision: Optional[str] = None

    # Metadata
    agent_name: str = ""
    processing_time_seconds: float = 0.0
    timestamp: datetime = Field(default_factory=datetime.now)


# =============================================================================
# ROUTING MODELS - Agent Recommendations and Decision Analysis
# =============================================================================

class AgentRecommendation(BaseModel):
    """
    Recommendation from a level agent for next routing.

    Each level agent (P/D/I/K/W) produces a recommendation for which
    level should be executed next. This enables intelligent routing
    based on the agent's understanding of its own results.

    Default routing: P→D, D→I, I→K, K→W, W→END
    """
    from_level: Literal["P", "D", "I", "K", "W"] = Field(
        ..., description="Which level agent produced this recommendation"
    )
    recommended_next_level: Literal["P", "D", "I", "K", "W", "END"] = Field(
        ..., description="Recommended next level to execute"
    )
    confidence: float = Field(
        default=0.8, ge=0.0, le=1.0,
        description="Confidence in this recommendation (0.0-1.0)"
    )
    reasoning: str = Field(
        default="", description="Explanation for this recommendation"
    )
    alternative_paths: List[str] = Field(
        default_factory=list,
        description="Alternative routing options considered"
    )
    created_at: datetime = Field(default_factory=datetime.now)

    @classmethod
    def create_default(cls, from_level: str) -> 'AgentRecommendation':
        """Create default recommendation based on standard DIKW flow"""
        default_next = {
            "P": "D",  # Planner → Data
            "D": "I",  # Data → Information
            "I": "K",  # Information → Knowledge
            "K": "W",  # Knowledge → Wisdom
            "W": "END"  # Wisdom → Complete
        }
        return cls(
            from_level=from_level,
            recommended_next_level=default_next.get(from_level, "END"),
            confidence=0.9,
            reasoning=f"Standard DIKW flow: {from_level} → {default_next.get(from_level, 'END')}"
        )


class DecisionAnalysisResult(BaseModel):
    """
    Decision Agent's LLM analysis of routing options.

    The Decision Agent independently analyzes the current state and
    produces its own routing proposal. This may agree or disagree
    with the last level agent's recommendation.

    Two-decision system:
    1. Last Agent's recommendation (default)
    2. Decision Agent's analysis (can override)
    """
    analyzed_recommendation: Optional[AgentRecommendation] = Field(
        default=None, description="The recommendation that was analyzed"
    )
    decision_agent_proposal: Literal["P", "D", "I", "K", "W", "END"] = Field(
        ..., description="Decision Agent's proposed next level"
    )
    decision_confidence: float = Field(
        default=0.8, ge=0.0, le=1.0,
        description="Confidence in Decision Agent's proposal"
    )
    analysis_reasoning: str = Field(
        default="", description="Explanation for Decision Agent's analysis"
    )
    factors_considered: List[str] = Field(
        default_factory=list,
        description="Factors considered in analysis (plan progress, success rate, etc.)"
    )
    agrees_with_recommendation: bool = Field(
        default=True,
        description="Whether Decision Agent agrees with last agent's recommendation"
    )
    human_feedback_incorporated: Optional[str] = Field(
        default=None,
        description="Human feedback that was incorporated (if any)"
    )
    iteration_number: int = Field(
        default=1, ge=1,
        description="Which iteration of analysis (increases with each feedback round)"
    )
    created_at: datetime = Field(default_factory=datetime.now)


class RoutingDecision(BaseModel):
    """
    Final routing decision record for audit trail.

    Persisted to disk for tracking all routing decisions made during
    a DIKW workflow execution.
    """
    decision_id: str = Field(
        ..., description="Unique identifier for this decision"
    )
    from_level: Literal["P", "D", "I", "K", "W", "START"] = Field(
        ..., description="Level that just completed (or START for initial)"
    )
    to_level: Literal["P", "D", "I", "K", "W", "END"] = Field(
        ..., description="Level being routed to"
    )

    # Decision inputs
    agent_recommendation: Optional[AgentRecommendation] = Field(
        default=None, description="Recommendation from the completing agent"
    )
    decision_analysis: Optional[DecisionAnalysisResult] = Field(
        default=None, description="Decision Agent's analysis"
    )

    # Human involvement
    human_approved: bool = Field(
        default=False, description="Whether human explicitly approved"
    )
    human_feedback: Optional[str] = Field(
        default=None, description="Human feedback provided (if any)"
    )
    feedback_iterations: int = Field(
        default=0, description="Number of feedback rounds before approval"
    )

    # Final outcome
    final_decision_source: Literal["agent_recommendation", "decision_analysis", "human_override", "force_override"] = Field(
        ..., description="What determined the final routing"
    )

    # Metadata
    created_at: datetime = Field(default_factory=datetime.now)
    thread_id: Optional[str] = Field(
        default=None, description="LangGraph thread ID for this session"
    )


class DIKWMetadata(BaseModel):
    """Project metadata and paths for DIKW processing"""
    dikw_project_name: str = Field(..., min_length=1)
    dikw_project_folder: Path

    # LangGraph infrastructure (organized under langgraph/)
    dikw_langgraph_folder: Path  # Parent folder for all LangGraph state
    dikw_messages_path: Optional[Path] = None  # LangGraph conversation history
    dikw_memory_path: Optional[Path] = None  # LangGraph agent memory/context
    dikw_checkpoints_path: Optional[Path] = None  # LangGraph checkpoints

    # Workspace folders (data and outputs)
    dikw_run_workspace_folder: Path  # OpenHands SDK execution workspace
    dikw_project_source_folder: Path  # Raw input data (renamed from data_folder)
    dikw_project_source_raw_folder: Path  # Raw data subfolder

    dikw_project_code_folder: Path  # All code organized by level (data/, information/, knowledge/, wisdom/)
    dikw_project_report_folder: Path  # All reports organized by level (data/, information/, knowledge/, wisdom/)
    dikw_project_output_folder: Optional[Path] = None  # Optional: Legacy field for final deliverables (deprecated)

    # Task tracking folders (new structure for plan and step history)
    dikw_task_folder: Path  # Parent folder: workspace/task/
    dikw_task_plan_folder: Path  # Plan tracking: workspace/task/plan/
    dikw_task_plan_history_folder: Path  # Plan history: workspace/task/plan/history/
    dikw_task_step_folder: Path  # Step tracking: workspace/task/step/

    # Python environment
    dikw_venv_python: Optional[Path] = None  # Path to virtual environment python executable (defaults to sys.executable)

    def __init__(self, **data):
        """Initialize metadata with automatic langgraph subfolder setup"""
        # If langgraph_folder provided, auto-populate subfolders
        if 'dikw_langgraph_folder' in data:
            langgraph_folder = data['dikw_langgraph_folder']
            if 'dikw_messages_path' not in data:
                data['dikw_messages_path'] = langgraph_folder / "messages"
            if 'dikw_memory_path' not in data:
                data['dikw_memory_path'] = langgraph_folder / "memory"
            if 'dikw_checkpoints_path' not in data:
                data['dikw_checkpoints_path'] = langgraph_folder / "checkpoints"
        # If no langgraph_folder but project_folder exists, create default
        elif 'dikw_project_folder' in data and 'dikw_langgraph_folder' not in data:
            data['dikw_langgraph_folder'] = data['dikw_project_folder'] / "langgraph"
            langgraph_folder = data['dikw_langgraph_folder']
            if 'dikw_messages_path' not in data:
                data['dikw_messages_path'] = langgraph_folder / "messages"
            if 'dikw_memory_path' not in data:
                data['dikw_memory_path'] = langgraph_folder / "memory"
            if 'dikw_checkpoints_path' not in data:
                data['dikw_checkpoints_path'] = langgraph_folder / "checkpoints"

        # Auto-populate task tracking folders if project_folder exists
        if 'dikw_project_folder' in data:
            project_folder = data['dikw_project_folder']
            if 'dikw_task_folder' not in data:
                data['dikw_task_folder'] = project_folder / "task"
            task_folder = data['dikw_task_folder']
            if 'dikw_task_plan_folder' not in data:
                data['dikw_task_plan_folder'] = task_folder / "plan"
            if 'dikw_task_plan_history_folder' not in data:
                data['dikw_task_plan_history_folder'] = task_folder / "plan" / "history"
            if 'dikw_task_step_folder' not in data:
                data['dikw_task_step_folder'] = task_folder / "step"

        super().__init__(**data)


# =============================================================================
# TASK MODELS - UNIFIED ARCHITECTURE
# =============================================================================

class Task(BaseModel):
    """
    Unified task specification for all DIKW levels.

    This single class replaces DataTask, InformationTask, KnowledgeTask, WisdomTask.
    The level field determines which DIKW level this task belongs to.
    """
    level: Literal["D", "I", "K", "W"] = Field(
        ...,
        description="DIKW level: D=Data, I=Information, K=Knowledge, W=Wisdom"
    )
    task_idx: int = Field(
        ...,
        description="Sequential index for this task within its level"
    )
    task_name: str = Field(
        ...,
        description="Unique identifier for this task (e.g., 'd1_explore_dataset')"
    )
    task_description: str = Field(
        ...,
        description="Detailed description of what this task should accomplish"
    )
    task_plan: str = Field(
        ...,
        description="Step-by-step execution plan for this task"
    )
    estimated_complexity: Literal["simple", "medium", "complex"] = Field(
        default="medium",
        description="Estimated complexity level for resource allocation"
    )
    execution_mode: Optional[Literal["skip", "run_code", "run_openhands", "run_claude_agent", "run_reasoning"]] = Field(
        default=None,
        description="How to execute this task: skip, run existing code, use OpenHands SDK, use Claude Agent SDK, or LLM reasoning"
    )
    force_rerun: bool = Field(
        default=False,
        description="Force re-execution even if outputs exist (deletes existing report/code first)"
    )
    # Agent-level execution mode (inherited from LevelConfig)
    agent_execution_mode: Literal["sdk_full", "sdk_context", "llm_only"] = Field(
        default="sdk_full",
        description="""
        Agent execution mode (from LevelConfig):
        - sdk_full: SDK handles everything (D/I default)
        - sdk_context: SDK gathers context, then LLM reasons (K/W with context)
        - llm_only: Pure LLM reasoning (K/W default, no SDK)
        """
    )
    context_sources: List[str] = Field(
        default_factory=list,
        description="Folders to read for context when agent_execution_mode='sdk_context'"
    )
    # SDK summarization settings (inherited from LevelConfig)
    use_sdk_summarization: bool = Field(
        default=False,
        description="Use Claude SDK for intelligent context summarization"
    )
    max_summary_tokens: int = Field(
        default=8000,
        description="Maximum tokens for SDK summarization output"
    )
    created_at: datetime = Field(
        default_factory=datetime.now,
        description="When this task was created"
    )


# =============================================================================
# RESULT MODELS (from current X-agent states)
# =============================================================================

class TaskResult(BaseModel):
    """Generic result from processing any task (D/I/K/W)"""
    level: Literal["D", "I", "K", "W"]
    task_name: str
    success: bool

    # Output paths
    task_code_folder: Optional[Path] = None
    task_report_folder: Optional[Path] = None
    task_report_file: Optional[Path] = None
    task_summary: str = ""

    # Execution details
    processing_time_seconds: float = 0.0
    error_message: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)


# =============================================================================
# DECISION POINT MODELS - Human-in-the-Loop Review
# =============================================================================

class ResultAnalysis(BaseModel):
    """Analysis of task-step results for decision making."""
    level: Literal["P", "D", "I", "K", "W"] = Field(
        ..., description="DIKW level that was analyzed (P=Planner, D=Data, I=Information, K=Knowledge, W=Wisdom)"
    )
    success_rate: float = Field(
        ..., ge=0.0, le=1.0, description="Proportion of successful tasks (0.0-1.0)"
    )
    key_findings: List[str] = Field(
        default_factory=list, description="Key findings from successful tasks"
    )
    concerns: List[str] = Field(
        default_factory=list, description="Concerns from failed tasks or issues"
    )
    confidence_score: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Overall confidence in results (0.0-1.0)"
    )
    needs_revision: bool = Field(
        default=False, description="Whether results suggest plan revision is needed"
    )
    analysis_timestamp: datetime = Field(default_factory=datetime.now)


class DecisionRecommendation(BaseModel):
    """LLM-generated recommendation for human review."""
    recommendation: Literal["APPROVE", "DENY", "SUGGEST"] = Field(
        ..., description="Recommended action: APPROVE to continue, DENY to abort, SUGGEST for changes"
    )
    reasoning: str = Field(
        ..., min_length=1, description="Explanation for the recommendation"
    )
    suggested_changes: Optional[str] = Field(
        default=None, description="Specific changes suggested (for SUGGEST recommendation)"
    )
    confidence: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Confidence in this recommendation"
    )
    generated_by: str = Field(
        default="llm", description="What generated this recommendation (llm, rule-based)"
    )


class HumanResponse(BaseModel):
    """Human response to decision point."""
    action: Literal["APPROVE", "DENY", "SUGGEST"] = Field(
        ..., description="Human's chosen action"
    )
    feedback: Optional[str] = Field(
        default=None, description="Optional feedback or explanation"
    )
    suggested_changes: Optional[str] = Field(
        default=None, description="Specific changes requested (for SUGGEST action)"
    )
    response_timestamp: datetime = Field(default_factory=datetime.now)


class DecisionPoint(BaseModel):
    """Complete decision point record for human-in-the-loop review."""
    level: Literal["P", "D", "I", "K", "W"] = Field(
        ..., description="DIKW level this decision point is for (P=Planner, D=Data, I=Information, K=Knowledge, W=Wisdom)"
    )
    step_result: DIKWStepResult = Field(
        ..., description="Step result being reviewed"
    )
    analysis: ResultAnalysis = Field(
        ..., description="Analysis of the step results"
    )
    recommendation: Optional[DecisionRecommendation] = Field(
        default=None, description="LLM recommendation (set by prepare_recommendation)"
    )
    human_response: Optional[HumanResponse] = Field(
        default=None, description="Human response (set after interrupt)"
    )
    decision_outcome: Optional[Literal["continue", "revise", "abort"]] = Field(
        default=None, description="Final outcome after processing human response"
    )
    created_at: datetime = Field(default_factory=datetime.now)
    resolved_at: Optional[datetime] = Field(
        default=None, description="When the decision was resolved"
    )


# =============================================================================
# UNIFIED STATE DEFINITION
# =============================================================================

class DIKWUnifiedState(BaseModel):
    """
    Unified state for entire DIKW hierarchy - flows through all layers.

    This single state eliminates all state transformations:
    - No XAgentInput/Output conversions
    - No wrapper nodes needed
    - Direct field access everywhere

    Field Organization:
    1. DIKW Orchestrator fields (plan, routing, results)
    2. Workspace metadata (shared by all)
    3. Current level context (what's being processed now)
    4. X-Agent topic management (D/I/K/W topics)
    5. Task execution (parallel task results)
    6. Control flow (routing between nodes)
    7. Messages and timing
    """

    # ========== SECTION 1: DIKW ORCHESTRATOR FIELDS ==========

    # Project metadata (optional for backward compatibility)
    metadata: Optional[DIKWMetadata] = Field(
        default=None,
        description="Project/workspace metadata shared across nodes"
    )

    # Plan management
    current_dikw_plan: Optional[DIKWPlan] = Field(
        default=None,
        description="Current DIKW plan with D/I/K/W instructions"
    )
    plan_history: List[DIKWPlan] = Field(
        default_factory=list,
        description="History of all plans (for revision tracking)"
    )
    plan_generation_mode: Optional[str] = Field(
        default=None,
        description="Plan generation mode: 'generate', 'revise', or 'skip'"
    )
    suggestions_for_plan_revision: Optional[str] = Field(
        default=None,
        description="User feedback/suggestions for plan revision (from decision_agent)"
    )

    # High-level questions/goals (for automatic plan generation)
    high_level_questions: Optional[List[str]] = Field(
        default=None,
        description="User's high-level questions/goals (agent will decompose into DIKW plan)"
    )
    original_goal_statement: Optional[str] = Field(
        default=None,
        description="User's single goal statement (alternative to questions list)"
    )

    # Current execution step
    current_dikw_step: Optional[DIKWStep] = Field(
        default=None,
        description="Current step being executed (level, status, etc.)"
    )

    # Results from each DIKW level
    d_results: Optional[DIKWStepResult] = Field(
        default=None,
        description="Results from Data Agent processing"
    )
    i_results: Optional[DIKWStepResult] = Field(
        default=None,
        description="Results from Information Agent processing"
    )
    k_results: Optional[DIKWStepResult] = Field(
        default=None,
        description="Results from Knowledge Agent processing"
    )
    w_results: Optional[DIKWStepResult] = Field(
        default=None,
        description="Results from Wisdom Agent processing"
    )

    # Orchestration control
    is_complete: bool = Field(
        default=False,
        description="True when all DIKW levels have been processed"
    )
    waiting_for_human: bool = Field(
        default=False,
        description="True when waiting for human intervention"
    )
    pending_human_intervention: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Details of pending human intervention if any"
    )

    # ========== SECTION 1B: DECISION POINT FIELDS ==========

    # Current decision point being processed
    current_decision_point: Optional[DecisionPoint] = Field(
        default=None,
        description="Current decision point awaiting human review (after task-step completion)"
    )

    # Decision history (all decision points from this session)
    decision_history: List[DecisionPoint] = Field(
        default_factory=list,
        description="History of all decision points and their outcomes"
    )

    # Decision point control flags
    awaiting_human_decision: bool = Field(
        default=False,
        description="True when graph is paused waiting for human decision"
    )
    decision_outcome: Optional[Literal["continue", "revise", "abort"]] = Field(
        default=None,
        description="Outcome from the most recent decision point"
    )

    # Decision point configuration
    enable_decision_points: bool = Field(
        default=False,
        description="Whether to enable decision points after each level"
    )

    # ========== SECTION 1C: ROUTING DECISION FIELDS ==========

    # Last agent's recommendation (filled by each level agent after completion)
    last_agent_recommendation: Optional[AgentRecommendation] = Field(
        default=None,
        description="Recommendation from the last completed level agent for next routing"
    )

    # Decision agent's analysis (filled by propose_next_step_node)
    decision_analysis: Optional[DecisionAnalysisResult] = Field(
        default=None,
        description="Decision Agent's LLM analysis of routing options"
    )

    # Human-confirmed routing (set after feedback loop completes)
    confirmed_next_level: Optional[Literal["P", "D", "I", "K", "W", "END"]] = Field(
        default=None,
        description="Human-confirmed next level (after review/feedback)"
    )

    # Feedback iteration tracking (max 3 rounds per JL's specification)
    routing_feedback_iterations: int = Field(
        default=0,
        description="Number of feedback iterations in current routing decision (max 3)"
    )

    # Routing decision history (for audit trail, persisted to disk)
    routing_history: List[RoutingDecision] = Field(
        default_factory=list,
        description="History of all routing decisions made during this session"
    )

    # Force override flag (for testing purposes only)
    force_route_to_level: Optional[Literal["P", "D", "I", "K", "W", "END"]] = Field(
        default=None,
        description="Force routing to specific level (testing only, bypasses LLM analysis)"
    )

    # Default questions (for initial mode)
    default_questions: List[str] = Field(
        default_factory=list,
        description="Default analysis questions presented to user in initial mode"
    )

    # ========== SECTION 1D: WORKSPACE TOOLS FIELDS ==========

    # Pending tool call (set by process_response, executed by tool_call_node)
    pending_tool_call: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Pending tool call: {tool_name, tool_args} - set when user requests workspace operation"
    )

    # Last tool result (for display and context)
    tool_result: Optional[str] = Field(
        default=None,
        description="Result from the last workspace tool execution"
    )

    # ========== SECTION 2: CURRENT LEVEL CONTEXT ==========
    # NOTE: Workspace metadata is now managed via set_metadata() in helpers.py
    # This keeps configuration separate from runtime state

    current_level: Optional[Literal["P", "D", "I", "K", "W"]] = Field(
        default=None,
        description="Current DIKW level being processed (P=Planner, D=Data, I=Information, K=Knowledge, W=Wisdom)"
    )
    current_instruction: str = Field(
        default="",
        description="Current level's instruction from plan"
    )
    context_from_previous_level: str = Field(
        default="",
        description="Summary from previous level (empty for D-level)"
    )

    # Current task being processed (for Send() API with task subgraph)
    # Uses custom reducer to handle parallel execution - last value wins (doesn't matter which)
    current_task: Annotated[Optional[Task], lambda x, y: y if y is not None else x] = Field(
        default=None,
        description="Current task being processed (passed via Send() to task subgraph)"
    )

    # ========== SECTION 4: X-AGENT TASK MANAGEMENT ==========

    # Historical tasks (accumulated across all steps)
    # All use unified Task class with level attribute
    data_tasks: List[Task] = Field(
        default_factory=list,
        description="All data tasks accumulated across D-steps (all have level='D')"
    )
    information_tasks: List[Task] = Field(
        default_factory=list,
        description="All information tasks accumulated across I-steps (all have level='I')"
    )
    knowledge_tasks: List[Task] = Field(
        default_factory=list,
        description="All knowledge tasks accumulated across K-steps (all have level='K')"
    )
    wisdom_tasks: List[Task] = Field(
        default_factory=list,
        description="All wisdom tasks accumulated across W-steps (all have level='W')"
    )

    # Current step tasks (what's being processed NOW in current level)
    # All use unified Task class with level attribute
    current_data_tasks: List[Task] = Field(
        default_factory=list,
        description="Data tasks to process in current D-step (all have level='D')"
    )
    current_information_tasks: List[Task] = Field(
        default_factory=list,
        description="Information tasks to process in current I-step (all have level='I')"
    )
    current_knowledge_tasks: List[Task] = Field(
        default_factory=list,
        description="Knowledge tasks to process in current K-step (all have level='K')"
    )
    current_wisdom_tasks: List[Task] = Field(
        default_factory=list,
        description="Wisdom tasks to process in current W-step (all have level='W')"
    )

    # Task coordination (per current level)
    tasks_in_progress: List[str] = Field(
        default_factory=list,
        description="Task names currently being processed"
    )
    completed_tasks: List[str] = Field(
        default_factory=list,
        description="Task names that completed successfully"
    )
    failed_tasks: List[str] = Field(
        default_factory=list,
        description="Task names that failed"
    )

    # Human feedback for task review
    human_task_feedback: Optional[str] = Field(
        default=None,
        description="Human feedback on generated tasks for revision"
    )
    tasks_approved: bool = Field(
        default=False,
        description="Whether current tasks have been approved for execution"
    )
    all_tasks_complete: bool = Field(
        default=False,
        description="Whether all current tasks are complete"
    )

    # ========== SECTION 5: TASK EXECUTION (parallel results) ==========

    # Results accumulated from parallel task executions (via Send API)
    task_processing_results: Annotated[List[TaskResult], add] = Field(
        default_factory=list,
        description="Results from parallel task executions (reducer: add = append)"
    )

    # Current task state (for individual task execution)
    current_task_idx: Optional[int] = Field(
        default=None,
        description="Index of currently executing task"
    )
    current_task_name: Optional[str] = Field(
        default=None,
        description="Name of currently executing task"
    )

    # Task execution tracking (for current task)
    # Uses reducer for parallel execution - last non-None value wins
    task_execution_error: Annotated[Optional[str], lambda x, y: y if y is not None else x] = Field(
        default=None,
        description="Error message from current task execution (if any)"
    )
    # Uses reducer for parallel execution - last value wins
    task_processing_time: Annotated[Optional[float], lambda x, y: y if y is not None else x] = Field(
        default=None,
        description="Processing time for current task in seconds"
    )
    task_execution_output: Optional[str] = Field(
        default=None,
        description="Stdout/stderr output from current task execution"
    )

    # OpenHands SDK results (if using run_openhands mode)
    # Uses reducer for parallel execution - collects all SDK results
    openhand_sdk_result: Annotated[List[Any], add] = Field(
        default_factory=list,
        description="Full OpenHandSDKResults objects from SDK execution (list for parallel tasks)"
    )

    # ========== SECTION 6: CONTROL FLOW ==========

    next_action: Literal[
        # DIKW orchestrator actions (new architecture)
        "route_to_planner",
        "route_to_planner_revision",
        "route_to_data",
        "route_to_information",
        "route_to_knowledge",
        "route_to_wisdom",
        "route_to_decision",
        "route_to_conversation",  # DIKWAgent conversation flow
        # DIKW orchestrator actions (legacy)
        "initial_plan",
        "route_to_data_agent",
        "route_to_information_agent",
        "route_to_knowledge_agent",
        "route_to_wisdom_agent",
        "plan_revision",
        "human_input",
        # X-agent actions (shared pattern for all levels)
        "generate_tasks",
        "review_tasks",
        "execute_tasks",
        "aggregate_results",
        "complete",
        "level_complete",  # Level agent finished, return to orchestrator
        # Task agent actions
        "execute_task",
        "finalize_task",
        # Special actions
        "skip_to_end"
    ] = Field(
        default="initial_plan",
        description="Next action to take in workflow"
    )

    # Processing state
    # Uses reducer for parallel execution - True when any task sets it True
    # (aggregation logic checks if ALL tasks completed separately)
    processing_complete: Annotated[bool, lambda x, y: x or y] = Field(
        default=False,
        description="Whether current level processing is complete"
    )
    # Uses reducer for parallel execution - last non-empty value wins
    result_summary: Annotated[str, lambda x, y: y if y else x] = Field(
        default="",
        description="Summary of current level processing"
    )

    # Human-in-the-loop approval
    human_approval: Optional[Literal["approve", "skip", "abort"]] = Field(
        default=None,
        description="User's approval decision for task execution (approve/skip/abort)"
    )

    # Failure information (for plan revision)
    blocked_reason: Optional[str] = Field(
        default=None,
        description="Reason why processing was blocked/failed"
    )
    suggestions_for_plan_revision: Optional[str] = Field(
        default=None,
        description="Suggestions for revising the DIKW plan"
    )

    # ========== SECTION 7: MESSAGES AND TIMING ==========

    messages: Annotated[List[AnyMessage], add_messages] = Field(
        default_factory=list,
        description="Conversation history (reducers concatenate)"
    )

    # ========== SECTION 7B: DECISION AGENT CONVERSATION STATE ==========

    decision_mode: Optional[Literal["initial", "review_plan", "review_step", "final"]] = Field(
        default=None,
        description="Current mode of Decision Agent (conversational interface)"
    )

    conversation_history: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Structured conversation history with Decision Agent (role, content, timestamp)"
    )

    # Agent's last response (for display - propagates from subgraphs unlike messages)
    last_agent_response: Optional[str] = Field(
        default=None,
        description="Last response from agent to display to user (simple string, propagates from subgraphs)"
    )

    user_preferences: Dict[str, Any] = Field(
        default_factory=dict,
        description="User preferences learned during conversation (verbosity, focus areas, etc.)"
    )

    clarifications_given: List[str] = Field(
        default_factory=list,
        description="Topics already explained to user (avoid repetition)"
    )

    last_human_input: Optional[str] = Field(
        default=None,
        description="Most recent input from human in decision agent"
    )

    decision_agent_context: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional context for decision agent (previous agent, step number, etc.)"
    )

    last_activity_time: datetime = Field(
        default_factory=datetime.now,
        description="Timestamp of last state update"
    )
    processing_start_time: datetime = Field(
        default_factory=datetime.now,
        description="When current level processing started"
    )
    processing_end_time: Optional[datetime] = Field(
        default=None,
        description="When current level processing ended"
    )

    # ========== HELPER METHODS ==========

    def get_current_tasks(self) -> List[Any]:
        """Get current tasks for current level"""
        task_map = {
            "D": self.current_data_tasks,
            "I": self.current_information_tasks,
            "K": self.current_knowledge_tasks,
            "W": self.current_wisdom_tasks,
        }
        return task_map.get(self.current_level, [])

    def get_historical_tasks(self) -> List[Any]:
        """Get historical tasks for current level"""
        task_map = {
            "D": self.data_tasks,
            "I": self.information_tasks,
            "K": self.knowledge_tasks,
            "W": self.wisdom_tasks,
        }
        return task_map.get(self.current_level, [])

    def get_pending_tasks(self) -> List[Any]:
        """Get tasks that need to be executed"""
        current_tasks = self.get_current_tasks()
        processed_names = set(
            self.completed_tasks + self.failed_tasks + self.tasks_in_progress
        )
        return [t for t in current_tasks if self._get_task_name(t) not in processed_names]

    def get_ready_tasks(self) -> List[Any]:
        """Get tasks ready for execution (excludes skip mode)"""
        pending = self.get_pending_tasks()
        return [t for t in pending if getattr(t, 'execution_mode', None) != "skip"]

    def _get_task_name(self, task: Any) -> str:
        """Extract task name from task object (unified Task class uses 'task_name')"""
        return getattr(task, "task_name", "")

    def is_level_complete(self) -> bool:
        """Check if all current tasks are processed"""
        current_tasks = self.get_current_tasks()
        processed_count = len(self.completed_tasks) + len(self.failed_tasks)
        return processed_count >= len(current_tasks)

    def generate_level_summary(self) -> str:
        """Generate summary for current level to return to DIKW orchestrator"""
        level_name = {
            "P": "Planner",
            "D": "Data",
            "I": "Information",
            "K": "Knowledge",
            "W": "Wisdom"
        }[self.current_level]

        total_tasks = len(self.get_current_tasks())
        completed = len([r for r in self.task_processing_results if r.success and r.level == self.current_level])
        failed = len([r for r in self.task_processing_results if not r.success and r.level == self.current_level])

        summary = f"{level_name} processing: {completed}/{total_tasks} tasks completed"
        if failed > 0:
            summary += f", {failed} failed"

        # Add key findings from successful tasks
        successful_summaries = [
            r.task_summary
            for r in self.task_processing_results
            if r.success and r.level == self.current_level and r.task_summary
        ]

        if successful_summaries:
            summary += f". Key findings: {'; '.join(successful_summaries[:3])}"

        return summary

    def save_tasks_for_level(self, level: str, metadata: DIKWMetadata) -> Optional[Path]:
        """
        Save tasks for specific DIKW level to JSON file

        Args:
            level: DIKW level (D/I/K/W)
            metadata: Workspace metadata (from config)
        """
        try:
            level_mapping = {
                "D": ("data_tasks", "data_tasks.json"),
                "I": ("information_tasks", "information_tasks.json"),
                "K": ("knowledge_tasks", "knowledge_tasks.json"),
                "W": ("wisdom_tasks", "wisdom_tasks.json")
            }

            if level not in level_mapping:
                return None

            task_attr, filename = level_mapping[level]
            tasks_file = metadata.dikw_project_folder / filename

            import json
            tasks_list = getattr(self, task_attr)
            tasks_data = [
                task.model_dump() if hasattr(task, 'model_dump') else task
                for task in tasks_list
            ]

            with open(tasks_file, 'w') as f:
                json.dump(tasks_data, f, indent=2)
            return tasks_file
        except Exception:
            return None

    class Config:
        """Pydantic configuration"""
        arbitrary_types_allowed = True
        validate_assignment = True
        use_enum_values = True


# =============================================================================
# INPUT STATE (for initial graph invocation)
# =============================================================================

# (Removed DIKWInputState) — unified DIKWUnifiedState is used directly for graph inputs
