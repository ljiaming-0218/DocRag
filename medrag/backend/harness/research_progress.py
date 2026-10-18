"""Bounded, reproducible context from persisted notes and original evidence."""

MAX_PROGRESS_ITEMS = 6
MAX_PROGRESS_REFERENCES = 4
MAX_CONTEXT_EVIDENCE = 12
MAX_CONTEXT_TEXT_CHARS = 12000
EVIDENCE_PREVIEW_CHARS = 800


def build_evidence_context(task):
    """Keep short originals and read excerpts, with balanced document coverage."""
    read_ids = {event.get("args_summary", {}).get("evidence_id") for event in task.events
                if event.get("name") == "read_evidence" and event.get("result") is not None
                and not event.get("error_code")}
    groups = [[item for item in reversed(list(task.evidence.values())) if item["document_id"] == doc]
              for doc in task.document_ids]
    selected = [group.pop(0) for group in groups if group][:MAX_CONTEXT_EVIDENCE]
    for event in reversed(task.events):
        eid = event.get("args_summary", {}).get("evidence_id")
        if eid in read_ids and eid in task.evidence and len(selected) < MAX_CONTEXT_EVIDENCE:
            if not any(item["evidence_id"] == eid for item in selected):
                selected.append(task.evidence[eid])
    while len(selected) < MAX_CONTEXT_EVIDENCE and any(groups):
        for group in groups:
            if group and len(selected) < MAX_CONTEXT_EVIDENCE:
                item = group.pop(0)
                if not any(prior["evidence_id"] == item["evidence_id"] for prior in selected):
                    selected.append(item)
    result, remaining = [], MAX_CONTEXT_TEXT_CHARS
    for item in selected:
        eid = item["evidence_id"]
        limit = min(2500 if eid in read_ids else EVIDENCE_PREVIEW_CHARS, remaining)
        if limit <= 0:
            break
        text = item["text"][:limit]
        remaining -= len(text)
        result.append({"evidence_id": eid, "document_id": item["document_id"],
                       "page_number": item.get("page_number"), "preview": text,
                       "is_truncated": len(text) < len(item["text"]), "has_been_read": eid in read_ids})
    return result


def build_research_progress(task):
    # The latest note for the same literal subquestion supersedes its draft.
    latest, seen = [], set()
    for note in reversed(task.notes):
        key = " ".join(note["question"].split())
        if key in seen:
            continue
        seen.add(key)
        latest.append(note)
    selected = latest[:MAX_PROGRESS_ITEMS]
    items = []
    for note in selected:
        ids = [eid for eid in note["evidence_ids"] if eid in task.evidence]
        items.append({
            "question": note["question"][:300],
            "draft_conclusion": note["conclusion"][:500],
            "state": "open" if note["open_questions"] or not ids else "recorded_unverified",
            "evidence": [{"evidence_id": eid, "document_id": task.evidence[eid]["document_id"],
                          "page_number": task.evidence[eid].get("page_number")}
                         for eid in ids[:MAX_PROGRESS_REFERENCES]],
            "open_questions": [question[:300] for question in note["open_questions"][:3]],
            "omitted_evidence_count": max(0, len(ids) - MAX_PROGRESS_REFERENCES),
            "omitted_open_question_count": max(0, len(note["open_questions"]) - 3),
        })
    current = next((item for item in items if item["state"] == "open"), None)
    question = (current["open_questions"][0] if current and current["open_questions"]
                else current["question"] if current else task.query[:300])
    read_ids = {event.get("args_summary", {}).get("evidence_id") for event in task.events
                if event.get("name") == "read_evidence" and event.get("result") is not None
                and not event.get("error_code")}
    candidates = ([ref["evidence_id"] for ref in current["evidence"]] if current
                  else list(task.evidence)[-8:])
    if task.config_version == "h3-v4":
        visible = {item["evidence_id"]: item for item in build_evidence_context(task)}
        candidates = [ref["evidence_id"] for ref in current["evidence"]] if current else []
        unread = next((eid for eid in candidates if eid not in read_ids
                       and (eid not in visible or visible[eid]["is_truncated"])), None)
    else:
        unread = next((eid for eid in candidates if eid not in read_ids
                       and len(task.evidence[eid]["text"]) > 300), None)
    if unread:
        next_step = {"action": "read_evidence", "evidence_id": unread,
                     "reason": "Only a preview is in context; inspect the original evidence before concluding."}
    elif current or not task.evidence:
        next_step = {"action": "search_documents", "question": question,
                     "reason": "An explicit subquestion remains open; search for evidence addressing it."}
    elif not items:
        next_step = {"action": "save_research_note", "question": question,
                     "reason": "Record a draft conclusion, evidence and remaining questions before reporting."}
    else:
        next_step = {"action": "review_and_finish",
                     "reason": "Review the original question and cited evidence; recorded drafts are not verified answers."}
    return {"current_question": question, "subquestions": items,
            "omitted_note_count": len(task.notes) - len(selected),
            "verification_status": "unverified_model_notes", "suggested_next_step": next_step}
