"""Direct time-budget acceptance with fixtures; no model calls or pytest."""
import asyncio
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import sys
from unittest.mock import AsyncMock, Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "medrag" / "backend"))
from harness import research_runner, research_task, research_tools


async def smoke():
    scope = {"kb_id": "kb", "document_ids": ["doc"], "index_generations": {"doc": "gen"}}
    with patch.object(research_task, "resolve_knowledge_base_scope", AsyncMock(return_value=scope)):
        task = await research_task.create_research_task("owner", "kb", "Question")
    assert task.wall_time_seconds == 5400 and task.model_call_seconds == 600
    task.started_at = datetime.now(timezone.utc) - timedelta(seconds=200)
    restored = research_task.ResearchTask(**asdict(task))
    assert 5199 < research_runner._remaining_seconds(restored) <= 5200
    old = asdict(task)
    old.pop("wall_time_seconds")
    old.pop("model_call_seconds")
    legacy = research_task.ResearchTask(**old)
    assert legacy.wall_time_seconds == 180 and legacy.model_call_seconds == 45
    assert research_runner._remaining_seconds(legacy) < 0
    with patch.object(research_tools, "ensure_task_scope", AsyncMock()), patch.object(
            research_tools, "_search_documents", Mock(return_value=[])) as search:
        await research_tools.execute_action(restored, {"name": "search_documents", "args": {"query": "method"}})
        assert search.call_count == 1
        try:
            await research_tools.execute_action(legacy, {"name": "search_documents", "args": {"query": "method"}})
            raise AssertionError("expired legacy task allowed")
        except research_tools.ResearchStoppedError:
            assert legacy.stop_reason == "wall_time_exhausted"
        restored.events = [{"name": "search_documents", "result": {"evidence_ids": []}}]
        response = [{"type": "function", "name": "finish_report", "arguments": json.dumps({
            "title": "Unknown", "findings": [], "unresolved": [{"text": "No supported answer"}]})}]
        with patch.object(research_runner, "_propose_action", Mock(return_value=response)) as propose:
            await research_runner.run_research_task(restored)
        assert restored.report and propose.call_args.args[1] == 600
    print(json.dumps({"result": "passed", "services": "fixtures", "new_limits": [5400, 600],
        "legacy_limits": [180, 45], "past_180_seconds_allowed": True, "persisted_limits_preserved": True,
        "model_timeout_applied": True}))


if __name__ == "__main__":
    asyncio.run(smoke())
