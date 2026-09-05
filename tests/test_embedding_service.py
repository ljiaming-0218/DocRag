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
