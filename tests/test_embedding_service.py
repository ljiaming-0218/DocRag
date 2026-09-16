from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from services import embedding_service as service


def make_chunks(count: int) -> list[dict]:
    return [
        {"文本块": f"chunk {index}"}
        for index in range(count)
    ]


def test_embed_chunks_processes_batches(monkeypatch):
    calls = []

    def fake_get_embeddings(texts):
        calls.append(list(texts))
        return [[float(text.rsplit(" ", 1)[-1])] for text in texts]

    monkeypatch.setattr(
        service,
        "get_embeddings",
        fake_get_embeddings,
    )
    chunks = make_chunks(5)

    result = service.embed_chunks(chunks, batch_size=2)

    assert result is chunks
    assert calls == [
        ["chunk 0", "chunk 1"],
        ["chunk 2", "chunk 3"],
        ["chunk 4"],
    ]
    assert [chunk["embedding"] for chunk in result] == [
        [0.0],
        [1.0],
        [2.0],
        [3.0],
        [4.0],
    ]


def test_embed_chunks_rejects_invalid_batch_size(monkeypatch):
    get_embeddings = Mock()
    monkeypatch.setattr(service, "get_embeddings", get_embeddings)

    with pytest.raises(ValueError, match="batch_size 必须大于 0"):
        service.embed_chunks(make_chunks(1), batch_size=0)

    get_embeddings.assert_not_called()


def test_embed_chunks_rejects_vector_count_mismatch(monkeypatch):
    monkeypatch.setattr(
        service,
        "get_embeddings",
        Mock(return_value=[[0.1]]),
    )

    with pytest.raises(RuntimeError, match="返回数量与文本块数量不一致"):
        service.embed_chunks(make_chunks(2), batch_size=2)


def test_embed_chunks_stops_after_failed_batch(monkeypatch):
    get_embeddings = Mock(
        side_effect=[
            [[0.1], [0.2]],
            RuntimeError("model failure"),
        ],
    )
    monkeypatch.setattr(
        service,
        "get_embeddings",
        get_embeddings,
    )
    chunks = make_chunks(5)

    with pytest.raises(RuntimeError, match="model failure"):
        service.embed_chunks(chunks, batch_size=2)

    assert get_embeddings.call_count == 2
    assert chunks[0]["embedding"] == [0.1]
    assert chunks[1]["embedding"] == [0.2]
    assert "embedding" not in chunks[2]


def test_embedding_client_is_initialized_once(monkeypatch):
    created_clients = []

    class FakeOpenAI:
        def __init__(self, **kwargs):
            created_clients.append(kwargs)

    monkeypatch.setattr(service, "_client", None)
    monkeypatch.setattr(service, "OpenAI", FakeOpenAI)
    monkeypatch.setattr(service, "EMBEDDING_API_KEY", "test-key")
    monkeypatch.setattr(
        service,
        "EMBEDDING_API_BASE",
        "https://api.example.com/v1",
    )

    first_client = service.get_embedding_client()
    second_client = service.get_embedding_client()

    assert first_client is second_client
    assert created_clients == [{
        "api_key": "test-key",
        "base_url": "https://api.example.com/v1",
        "timeout": service.EMBEDDING_TIMEOUT_SECONDS,
        "max_retries": service.EMBEDDING_MAX_RETRIES,
    }]


def test_get_embeddings_restores_api_result_order(monkeypatch):
    create = Mock(return_value=SimpleNamespace(data=[
        SimpleNamespace(index=1, embedding=[0.3, 0.4]),
        SimpleNamespace(index=0, embedding=[0.1, 0.2]),
    ]))
    client = SimpleNamespace(
        embeddings=SimpleNamespace(create=create),
    )
    monkeypatch.setattr(
        service,
        "get_embedding_client",
        Mock(return_value=client),
    )
    monkeypatch.setattr(service, "EMBEDDING_DIMENSION", 2)

    result = service.get_embeddings(["first", "second"])

    assert result == [[0.1, 0.2], [0.3, 0.4]]
    create.assert_called_once_with(
        model=service.EMBEDDING_MODEL,
        input=["first", "second"],
    )


def test_get_embeddings_rejects_dimension_mismatch(monkeypatch):
    client = SimpleNamespace(
        embeddings=SimpleNamespace(
            create=Mock(return_value=SimpleNamespace(data=[
                SimpleNamespace(index=0, embedding=[0.1, 0.2]),
            ])),
        ),
    )
    monkeypatch.setattr(
        service,
        "get_embedding_client",
        Mock(return_value=client),
    )
    monkeypatch.setattr(service, "EMBEDDING_DIMENSION", 1024)

    with pytest.raises(RuntimeError, match="Embedding 维度与配置不一致"):
        service.get_embeddings(["text"])


def test_get_embeddings_rejects_invalid_response_indices(monkeypatch):
    client = SimpleNamespace(
        embeddings=SimpleNamespace(
            create=Mock(return_value=SimpleNamespace(data=[
                SimpleNamespace(index=0, embedding=[0.1]),
                SimpleNamespace(index=0, embedding=[0.2]),
            ])),
        ),
    )
    monkeypatch.setattr(
        service,
        "get_embedding_client",
        Mock(return_value=client),
    )
    monkeypatch.setattr(service, "EMBEDDING_DIMENSION", 1)

    with pytest.raises(RuntimeError, match="返回索引与输入顺序不一致"):
        service.get_embeddings(["first", "second"])


def test_embedding_client_requires_api_key(monkeypatch):
    monkeypatch.setattr(service, "_client", None)
    monkeypatch.setattr(service, "EMBEDDING_API_KEY", "")

    with pytest.raises(RuntimeError, match="EMBEDDING_API_KEY 未配置"):
        service.get_embedding_client()
