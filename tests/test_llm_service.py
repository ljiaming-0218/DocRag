from types import SimpleNamespace
from unittest.mock import Mock

from services import llm_service as service


def configure_roles(monkeypatch):
    values = {
        "REWRITE_LLM_PROVIDER": "rewrite-provider",
        "REWRITE_LLM_API_KEY": "rewrite-key",
        "REWRITE_LLM_BASE_URL": "https://rewrite.example/v1",
        "REWRITE_LLM_MODEL": "rewrite-model",
        "REWRITE_LLM_TIMEOUT_SECONDS": 20,
        "REWRITE_LLM_MAX_RETRIES": 0,
        "ANSWER_LLM_PROVIDER": "answer-provider",
        "ANSWER_LLM_API_KEY": "answer-key",
        "ANSWER_LLM_BASE_URL": "https://answer.example/v1",
        "ANSWER_LLM_MODEL": "answer-model",
        "ANSWER_LLM_TIMEOUT_SECONDS": 120,
        "ANSWER_LLM_MAX_RETRIES": 1,
    }
    for name, value in values.items():
        monkeypatch.setattr(service, name, value)
    monkeypatch.setattr(service, "_clients", {})


def make_client_response(content):
    create = Mock(return_value=SimpleNamespace(
        choices=[SimpleNamespace(
            message=SimpleNamespace(content=content),
        )],
    ))
    client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=create),
        ),
    )
    return client, create


def test_resolve_llm_role_routes_operations():
    assert service.resolve_llm_role("query_rewrite") == "rewrite"
    assert service.resolve_llm_role("summary_query") == "rewrite"
    assert service.resolve_llm_role("task_route") == "rewrite"
    assert service.resolve_llm_role("qa") == "answer"
    assert service.resolve_llm_role("summary") == "answer"
    assert service.resolve_llm_role("report") == "answer"


def test_create_llm_client_caches_each_role_separately(monkeypatch):
    configure_roles(monkeypatch)
    openai_calls = []

    class FakeOpenAI:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            openai_calls.append(kwargs)

    monkeypatch.setattr(service, "OpenAI", FakeOpenAI)

    rewrite_client = service.create_llm_client("rewrite")
    same_rewrite_client = service.create_llm_client("rewrite")
    answer_client = service.create_llm_client("answer")

    assert rewrite_client is same_rewrite_client
    assert rewrite_client is not answer_client
    assert len(openai_calls) == 2
    assert rewrite_client.kwargs == {
        "api_key": "rewrite-key",
        "base_url": "https://rewrite.example/v1",
        "timeout": 20,
        "max_retries": 0,
    }
    assert answer_client.kwargs == {
        "api_key": "answer-key",
        "base_url": "https://answer.example/v1",
        "timeout": 120,
        "max_retries": 1,
    }


def test_generate_answer_uses_rewrite_model(monkeypatch):
    configure_roles(monkeypatch)
    client, create = make_client_response("rewritten query")
    create_client = Mock(return_value=client)
    monkeypatch.setattr(service, "create_llm_client", create_client)

    result = service.generate_answer("prompt", operation="query_rewrite")

    assert result == "rewritten query"
    create_client.assert_called_once_with("rewrite")
    assert create.call_args.kwargs["model"] == "rewrite-model"


def test_generate_answer_uses_answer_model(monkeypatch):
    configure_roles(monkeypatch)
    client, create = make_client_response("final answer")
    create_client = Mock(return_value=client)
    monkeypatch.setattr(service, "create_llm_client", create_client)

    result = service.generate_answer("prompt", operation="summary")

    assert result == "final answer"
    create_client.assert_called_once_with("answer")
    assert create.call_args.kwargs["model"] == "answer-model"
