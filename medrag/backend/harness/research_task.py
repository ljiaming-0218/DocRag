from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timezone
from uuid import uuid4

from services.knowledge_base_service import resolve_knowledge_base_scope


CURRENT_RESEARCH_CONFIG = "h3-v4"
DEFAULT_RESEARCH_WALL_SECONDS = 5400
DEFAULT_RESEARCH_MODEL_CALL_SECONDS = 600


class ResearchIndexUnavailableError(RuntimeError):
    """A task cannot freeze a usable index generation."""


@dataclass
class ResearchTask:
    task_id: str
    owner_id: str
    kb_id: str
    document_ids: tuple[str, ...]
    index_generations: dict[str, str]
    created_at: datetime
    query: str = ""
    document_filenames: dict[str, str] = field(default_factory=dict)
    status: str = "running"
    started_at: datetime | None = None
    finished_at: datetime | None = None
    stop_reason: str | None = None
    model_turns: int = 0
    model_actions: int = 0
    tool_calls: int = 0
    evidence: dict[str, dict] = field(default_factory=dict)
    notes: list[dict] = field(default_factory=list)
    report: dict | None = None
    events: list[dict] = field(default_factory=list)
    config_version: str = CURRENT_RESEARCH_CONFIG
    pending_event: dict | None = None
    pending_action: dict | None = None
    last_result: dict | None = None
    empty_searches: int = 0
    coverage_attempted: bool = False
    coverage_queue: list[str] = field(default_factory=list)
    parent_task_id: str | None = None
    followup_context: dict | None = None
    inherited_evidence_ids: list[str] = field(default_factory=list)
    duplicate_searches: int = 0
    # Missing fields in older persisted tasks retain their original limits.
    wall_time_seconds: float = 180
    model_call_seconds: float = 45


def inherit_research_context(task: ResearchTask, parent: ResearchTask) -> None:
    """Copy a bounded snapshot; reports remain unverified and counters stay fresh."""
    if task.owner_id != parent.owner_id or task.kb_id != parent.kb_id:
        raise PermissionError("follow-up must use the same owner and knowledge base")
    if set(task.document_ids) != set(parent.document_ids) or task.index_generations != parent.index_generations:
        raise ResearchIndexUnavailableError("follow-up document scope or index generation changed")
    task.parent_task_id = parent.task_id
    report = parent.report or {}
    task.followup_context = {
        "root_question": (parent.followup_context or {}).get("root_question", parent.query),
        "previous_question": parent.query,
        "previous_report_title": str(report.get("title", ""))[:200],
        "previous_report_claims_unverified": [
            {"text": str(claim.get("text", ""))[:500], "status": claim.get("status")}
            for claim in report.get("claims", [])[:6]
        ],
    }
    cited = {eid for claim in report.get("claims", []) for eid in claim.get("evidence_ids", [])}
    items = list(parent.evidence.values())
    items = [item for item in items if item["evidence_id"] not in cited] + [
        item for item in items if item["evidence_id"] in cited
    ]
    for item in items[-20:]:
        if item.get("document_id") not in task.document_ids or item.get("generation_id") != task.index_generations[item["document_id"]]:
            raise ResearchIndexUnavailableError("inherited evidence no longer matches the frozen index")
        evidence = deepcopy(item)
        evidence["evidence_id"] = f"{task.task_id}:{item['chunk_id']}"
        task.evidence[evidence["evidence_id"]] = evidence
    task.inherited_evidence_ids = list(task.evidence)


async def create_research_task(
    owner_id: str,
    kb_id: str,
    query: str,
    selected_document_ids: list[str] | None = None,
) -> ResearchTask:
    """Resolve the caller-owned KB once and freeze its exact document scope."""
    query = query.strip()
    if not query or len(query) > 2000:
        raise ValueError("research query must be between 1 and 2000 characters")
    scope = await resolve_knowledge_base_scope(
        owner_id, kb_id, selected_document_ids
    )
    generations = scope["index_generations"]
    if any(not generations.get(document_id) for document_id in scope["document_ids"]):
        raise ResearchIndexUnavailableError(
            "research requires an explicit index generation for every document"
        )
    return ResearchTask(
        task_id=str(uuid4()),
        owner_id=owner_id,
        kb_id=scope["kb_id"],
        document_ids=tuple(scope["document_ids"]),
        index_generations={
            document_id: generations[document_id]
            for document_id in scope["document_ids"]
        },
        created_at=datetime.now(timezone.utc),
        query=query,
        wall_time_seconds=DEFAULT_RESEARCH_WALL_SECONDS,
        model_call_seconds=DEFAULT_RESEARCH_MODEL_CALL_SECONDS,
        document_filenames={
            document_id: scope.get("document_filenames", {}).get(document_id, document_id)
            for document_id in scope["document_ids"]
        },
    )
