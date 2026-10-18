from asyncio import to_thread
from datetime import datetime, timezone

from harness.research_task import ResearchTask, DEFAULT_RESEARCH_WALL_SECONDS
from services.knowledge_base_service import resolve_knowledge_base_scope
from services.vector_store_service import has_chunks


MAX_MODEL_ACTIONS = 8
MAX_TOOL_CALLS = 12
MAX_WALL_SECONDS = DEFAULT_RESEARCH_WALL_SECONDS
MAX_QUERY_CHARS = 1000
MAX_RESULTS = 10
ALLOWED_ACTIONS = {
    "search_documents": {"query", "n_results", "document_ids"},
    "read_evidence": {"evidence_id"},
    "save_research_note": {"question", "conclusion", "evidence_ids", "open_questions"},
    "finish_report": {"title", "claims"},
}


class ResearchStoppedError(RuntimeError):
    pass


def build_run_limitations(task: ResearchTask) -> list[dict]:
    """Describe this execution's retrieval coverage, not properties of a paper."""
    covered = {item["document_id"] for item in task.evidence.values()}
    missing = [document_id for document_id in task.document_ids if document_id not in covered]
    return [{"code": "retrieval_gap", "document_ids": missing}] if missing else []


def _stop(task: ResearchTask, reason: str) -> None:
    task.status = "stopped"
    task.stop_reason = reason
    raise ResearchStoppedError(reason)


async def ensure_task_scope(task: ResearchTask) -> None:
    try:
        scope = await resolve_knowledge_base_scope(
            task.owner_id, task.kb_id, list(task.document_ids)
        )
    except (LookupError, PermissionError, ValueError, RuntimeError):
        _stop(task, "document_scope_unavailable")
    if scope["index_generations"] != task.index_generations:
        _stop(task, "index_generation_changed")
    for document_id in task.document_ids:
        try:
            available = await to_thread(
                has_chunks, task.owner_id, document_id,
                task.index_generations[document_id],
            )
        except RuntimeError:
            _stop(task, "index_generation_unavailable")
        if not available:
            _stop(task, "index_generation_unavailable")


def _nonempty(value: object, name: str, limit: int) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > limit:
        raise ValueError(f"{name} must be nonempty and at most {limit} characters")
    return value.strip()


def _evidence_ids(task: ResearchTask, value: object) -> list[str]:
    if not isinstance(value, list) or len(value) > 20:
        raise ValueError("evidence_ids must be a list of at most 20 IDs")
    if any(not isinstance(item, str) or item not in task.evidence for item in value):
        raise ValueError("evidence_id is outside this task's evidence ledger")
    return list(dict.fromkeys(value))


def _search_documents(
    task: ResearchTask, query: str, n_results: int, document_ids: list[str],
) -> list[dict]:
    # Import lazily: the existing search service loads the local reranker.
    from services.search_service import search_relevant_chunks_for_documents

    return search_relevant_chunks_for_documents(
        task.owner_id, document_ids, query, n_results,
        dict(task.index_generations),
    )


async def execute_action(
    task: ResearchTask, action: dict, *, model_proposed: bool = True,
    budget_reserved: bool = False,
) -> dict:
    """Validate an action; identity and scope never come from its arguments."""
    if task.status != "running":
        raise ResearchStoppedError(task.stop_reason or task.status)
    if not isinstance(action, dict) or action.get("name") not in ALLOWED_ACTIONS:
        raise ValueError("unsupported research action")
    name = action["name"]
    args = action.get("args")
    if not isinstance(args, dict) or set(args) - ALLOWED_ACTIONS[name]:
        raise ValueError("unsupported research action arguments")
    if not budget_reserved and ((model_proposed and task.model_actions >= MAX_MODEL_ACTIONS) or task.tool_calls >= MAX_TOOL_CALLS):
        _stop(task, "budget_exhausted")
    if (datetime.now(timezone.utc) - (task.started_at or task.created_at)).total_seconds() >= task.wall_time_seconds:
        _stop(task, "wall_time_exhausted")
    await ensure_task_scope(task)
    if not budget_reserved:
        if model_proposed:
            task.model_actions += 1
        task.tool_calls += 1

    if name == "search_documents":
        query = _nonempty(args.get("query"), "query", MAX_QUERY_CHARS)
        n_results = args.get("n_results", 3)
        if type(n_results) is not int or not 1 <= n_results <= MAX_RESULTS:
            raise ValueError("n_results must be between 1 and 10")
        document_ids = args.get("document_ids")
        if document_ids is None:
            document_ids = list(task.document_ids)
        if (not isinstance(document_ids, list) or not document_ids
                or any(not isinstance(item, str) for item in document_ids)
                or len(set(document_ids)) != len(document_ids)
                or not set(document_ids).issubset(task.document_ids)):
            raise ValueError("document_ids must be a nonempty subset of this task")
        chunks = await to_thread(_search_documents, task, query, n_results, document_ids)
        await ensure_task_scope(task)
        if (datetime.now(timezone.utc) - (task.started_at or task.created_at)).total_seconds() >= task.wall_time_seconds:
            _stop(task, "wall_time_exhausted")
        results = []
        for chunk in chunks[:n_results]:
            metadata = chunk.get("元数据") or {}
            document_id = metadata.get("document_id")
            chunk_id = chunk.get("chunk_id")
            if (metadata.get("user_id") != task.owner_id
                    or document_id not in document_ids
                    or metadata.get("index_generation_id") != task.index_generations[document_id]
                    or not isinstance(chunk_id, str) or not chunk_id
                    or not isinstance(chunk.get("文本块"), str)):
                _stop(task, "retrieval_scope_mismatch")
            evidence_id = f"{task.task_id}:{chunk_id}"
            evidence = {
                "evidence_id": evidence_id,
                "chunk_id": chunk_id,
                "document_id": document_id,
                "generation_id": task.index_generations[document_id],
                "page_number": metadata.get("page_number"),
                "text": chunk["文本块"],
                "retrieval_sources": list(chunk.get("retrieval_sources") or []),
                "rerank_score": chunk.get("rerank_score"),
            }
            results.append(evidence)
        task.evidence.update({item["evidence_id"]: item for item in results})
        return {"evidence": results}

    if name == "read_evidence":
        evidence_id = _nonempty(args.get("evidence_id"), "evidence_id", 300)
        if evidence_id not in task.evidence:
            raise ValueError("evidence_id is outside this task's evidence ledger")
        return dict(task.evidence[evidence_id])

    if name == "save_research_note":
        note = {
            "question": _nonempty(args.get("question"), "question", 1000),
            "conclusion": _nonempty(args.get("conclusion"), "conclusion", 4000),
            "evidence_ids": _evidence_ids(task, args.get("evidence_ids")),
            "open_questions": args.get("open_questions", []),
        }
        if not isinstance(note["open_questions"], list) or len(note["open_questions"]) > 10:
            raise ValueError("open_questions must be a list of at most 10")
        note["open_questions"] = [
            _nonempty(item, "open_question", 1000) for item in note["open_questions"]
        ]
        if not note["evidence_ids"] and not note["open_questions"]:
            raise ValueError("an unsupported conclusion must record an open question")
        task.notes.append(note)
        return {"note_index": len(task.notes) - 1}

    if not (task.parent_task_id and task.inherited_evidence_ids and
            all(eid in task.evidence for eid in task.inherited_evidence_ids)) and not any(
            event.get("name") == "search_documents" and event.get("result") is not None
            for event in task.events):
        raise ValueError("report requires a completed search step")
    title = _nonempty(args.get("title"), "title", 200)
    claims = args.get("claims")
    if not isinstance(claims, list) or not claims or len(claims) > 30:
        raise ValueError("claims must contain between 1 and 30 items")
    checked_claims = []
    for claim in claims:
        if not isinstance(claim, dict) or set(claim) - {"text", "evidence_ids", "status", "role"} or not {"text", "evidence_ids", "status"}.issubset(claim):
            raise ValueError("each claim needs text, evidence_ids and status")
        status = claim["status"]
        evidence_ids = _evidence_ids(task, claim["evidence_ids"])
        if status not in {"supported", "unknown"}:
            raise ValueError("claim status must be supported or unknown")
        if status == "supported" and not evidence_ids:
            raise ValueError("supported claim requires evidence")
        if status == "unknown" and evidence_ids:
            raise ValueError("unknown claim cannot cite supporting evidence")
        role = claim.get("role", "finding")
        if role not in {"finding", "comparison", "limitation"}:
            raise ValueError("invalid claim role")
        checked_claims.append({
            "text": _nonempty(claim["text"], "claim text", 2000),
            "status": status,
            "evidence_ids": evidence_ids,
            "role": role,
        })
    cited_documents = {
        task.evidence[evidence_id]["document_id"]
        for claim in checked_claims
        for evidence_id in claim["evidence_ids"]
    }
    cross_document_claim_count = sum(
        claim["role"] == "comparison" and claim["status"] == "supported" and
        len({task.evidence[evidence_id]["document_id"] for evidence_id in claim["evidence_ids"]}) >= 2
        for claim in checked_claims
    )
    comparison_requested = len(task.document_ids) >= 2 and any(
        word in task.query.casefold() for word in ("比较", "对比", "异同", "关联", "联系", "compare", "comparison", "contrast", "relationship")
    )
    gaps = []
    if comparison_requested:
        if missing := [document_id for document_id in task.document_ids if document_id not in cited_documents]:
            gaps.append({"code": "uncited_documents", "document_ids": missing})
        if not cross_document_claim_count:
            gaps.append({"code": "missing_cross_document_comparison"})
        if not any(
            claim["role"] == "limitation" and claim["status"] == "supported"
            and len({task.evidence[evidence_id]["document_id"]
                     for evidence_id in claim["evidence_ids"]}) >= 2
            for claim in checked_claims
        ):
            gaps.append({"code": "missing_comparison_limitations"})
    task.report = {
        "title": title,
        "claims": checked_claims,
        "run_limitations": build_run_limitations(task),
        "verification_status": "needs_semantic_review",
        "cross_document_claim_count": cross_document_claim_count,
        "question_coverage": {"status": "partial" if gaps else "structurally_complete", "gaps": gaps},
        "document_coverage": {
            "cited_document_ids": [
                document_id for document_id in task.document_ids
                if document_id in cited_documents
            ],
            "uncited_document_ids": [
                document_id for document_id in task.document_ids
                if document_id not in cited_documents
            ],
        },
    }
    task.status = "partial" if gaps else "completed"
    task.stop_reason = "report_incomplete" if gaps else None
    return dict(task.report)
