from unittest.mock import AsyncMock, Mock

import pytest

from services import chat_service
from services.agent_router_service import route_task
from services.prompt_service import (
    build_source_check_prompt,
    build_term_prompt,
)


@pytest.mark.parametrize(
    ("query", "expected_task"),
    [
        ("生成这篇文献的阅读报告", "report"),
        ("这个结论的引用来源是什么？", "source_check"),
        ("提取本文的关键词和专业术语", "term"),
        ("总结这篇文献", "summary"),
        ("LoRA 的核心机制是什么？", "qa"),
    ],
)
def test_route_task_selects_expected_tool(
    query: str,
    expected_task: str,
):
    assert route_task(query) == expected_task


def test_source_check_has_priority_over_summary():
    assert route_task("总结这个结论的证据来源") == "source_check"


@pytest.mark.parametrize(
    ("prompt_builder", "query"),
    [
        (build_term_prompt, "解释 LoRA 术语"),
        (build_source_check_prompt, "该结论有依据吗？"),
    ],
)
def test_agent_tool_prompt_renders_real_source(
    prompt_builder,
    query: str,
):
    sources = [{
        "文本块": (
            "LoRA freezes pretrained weights and updates "
            "low-rank matrices."
        ),
        "距离": 0.2,
        "元数据": {
            "page_number": 1,
            "chunk_index": 2,
        },
    }]

    prompt = prompt_builder(
        query,
        sources,
        "general",
        [],
    )

    assert query in prompt
    assert "LoRA freezes pretrained weights" in prompt


def make_context() -> dict:
    return {
        "conversation_id": "conversation-1",
        "document_id": "document-1",
        "user_id": "user-1",
        "history": [],
        "query": "query",
        "user_type": "general",
        "rewritten_query": "rewritten query",
        "retrieval_queries": ["rewritten query"],
        "user_message": {"message_id": "user-message-1"},
        "sources": [{"文本块": "source text"}],
        "sources_count": 1,
        "prompt": "qa prompt",
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("task_type", "builder_name", "operation"),
    [
        ("term", "build_term_prompt", "term"),
        (
            "source_check",
            "build_source_check_prompt",
            "source_check",
        ),
    ],
)
async def test_agent_tool_uses_specialized_prompt_and_saves_task_type(
    monkeypatch,
    task_type: str,
    builder_name: str,
    operation: str,
):
    context = make_context()
    prompt_builder = Mock(return_value=f"{task_type} prompt")
    generate_answer = Mock(return_value=f"{task_type} answer")
    create_message = AsyncMock(return_value={
        "message_id": "assistant-message-1",
    })
    set_message_status = AsyncMock()

    monkeypatch.setattr(
        chat_service,
        "route_task",
        Mock(return_value=task_type),
    )
    monkeypatch.setattr(
        chat_service,
        "prepare_ask_context",
        AsyncMock(return_value=context),
    )
    monkeypatch.setattr(
        chat_service,
        builder_name,
        prompt_builder,
    )
    monkeypatch.setattr(
        chat_service,
        "generate_answer",
        generate_answer,
    )
    monkeypatch.setattr(
        chat_service,
        "create_message",
        create_message,
    )
    monkeypatch.setattr(
        chat_service,
        "set_message_status",
        set_message_status,
    )

    result = await chat_service.ask_conversation(
        "user-1",
        "conversation-1",
        "query",
    )

    assert result["task_type"] == task_type
    assert result["answer"] == f"{task_type} answer"
    prompt_builder.assert_called_once_with(
        "rewritten query",
        context["sources"],
        "general",
        [],
    )
    generate_answer.assert_called_once_with(
        f"{task_type} prompt",
        operation=operation,
    )
    assert create_message.await_args.kwargs["task_type"] == task_type
    assert create_message.await_args.kwargs["sources"] == context["sources"]
    set_message_status.assert_awaited_once_with(
        "user-message-1",
        "completed",
    )
