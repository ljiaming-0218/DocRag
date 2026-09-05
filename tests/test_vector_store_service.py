from unittest.mock import Mock

import pytest

from services import vector_store_service as service


def make_chunk(
    *,
    user_id: str = "user-1",
    document_id: str = "document-1",
    page_number: int = 1,
    chunk_index: int = 0,
    text: str = "chunk text",
    index_generation_id: str | None = None,
) -> dict:
    chunk = {
        "user_id": user_id,
        "document_id": document_id,
        "页码": page_number,
        "块索引": chunk_index,
        "文本块": text,
        "embedding": [0.1, 0.2],
        "提取方式": "text",
        "图片数量": 0,
    }
    if index_generation_id is not None:
        chunk["index_generation_id"] = index_generation_id
    return chunk


class FakeCollection:
    def __init__(
        self,
        existing_ids: set[str] | None = None,
        *,
        fail_upsert: bool = False,
        fail_upsert_call: int | None = None,
        missing_after_upsert: set[str] | None = None,
    ) -> None:
        self.ids = set(existing_ids or set())
        self.fail_upsert = fail_upsert
        self.fail_upsert_call = fail_upsert_call
        self.missing_after_upsert = set(missing_after_upsert or set())
        self.upsert_calls: list[dict] = []
        self.delete_calls: list[list[str]] = []

    def get(
        self,
        *,
        ids=None,
        where=None,
        include=None,
        limit=None,
    ) -> dict:
        if ids is not None:
            visible_ids = (
                self.ids.intersection(ids)
                - self.missing_after_upsert
            )
            return {"ids": sorted(visible_ids)}

        result_ids = sorted(self.ids)
        if limit is not None:
            result_ids = result_ids[:limit]
        return {"ids": result_ids}

    def upsert(self, **kwargs) -> None:
        self.upsert_calls.append(kwargs)
        if self.fail_upsert or (
            self.fail_upsert_call == len(self.upsert_calls)
        ):
            raise RuntimeError("simulated upsert failure")
        self.ids.update(kwargs["ids"])

    def delete(self, *, ids) -> None:
        deleted_ids = list(ids)
        self.delete_calls.append(deleted_ids)
        self.ids.difference_update(deleted_ids)


def install_fake_chroma(monkeypatch, collection: FakeCollection):
    client = Mock()
    client.get_or_create_collection.return_value = collection
    client.get_collection.return_value = collection
    persistent_client = Mock(return_value=client)
    monkeypatch.setattr(
        service.chromadb,
        "PersistentClient",
        persistent_client,
    )
    return persistent_client


def test_build_chunk_id_is_stable_and_scoped():
    chunk = make_chunk(
        user_id="user-a",
        document_id="document-a",
        page_number=3,
        chunk_index=7,
    )

    assert service.build_chunk_id(chunk) == (
        "user-a_document-a_page_3_chunk_7"
    )


def test_build_chunk_id_scopes_versioned_chunk_to_generation():
    chunk = make_chunk(
        page_number=3,
        chunk_index=7,
        index_generation_id="generation-new",
    )

    assert service.build_chunk_id(chunk) == (
        "user-1_document-1_generation-new_page_3_chunk_7"
    )

def test_build_chunk_metadata_supports_legacy_chunk():
    chunk = make_chunk(page_number=3, chunk_index=7)

    metadata = service.build_chunk_metadata(chunk)

    assert metadata == {
        "page_number": 3,
        "chunk_index": 7,
        "document_id": "document-1",
        "user_id": "user-1",
        "extraction_method": "text",
        "image_count": 0,
        "chunk_strategy": "fixed",
    }


def test_build_chunk_metadata_includes_recursive_strategy():
    chunk = make_chunk(page_number=3, chunk_index=7)
    chunk["strategy"] = "recursive"

    metadata = service.build_chunk_metadata(chunk)

    assert metadata["chunk_strategy"] == "recursive"


def test_empty_chunks_return_zero_without_opening_chroma(monkeypatch):
    persistent_client = Mock()
    monkeypatch.setattr(
        service.chromadb,
        "PersistentClient",
        persistent_client,
    )

    assert service.save_chunks([]) == 0
    persistent_client.assert_not_called()


def test_first_save_upserts_without_deleting(monkeypatch):
    collection = FakeCollection()
    install_fake_chroma(monkeypatch, collection)
    chunks = [
        make_chunk(chunk_index=0),
        make_chunk(chunk_index=1),
    ]

    saved_count = service.save_chunks(chunks)

    assert saved_count == 2
    assert len(collection.upsert_calls) == 1
    assert collection.delete_calls == []
    assert collection.ids == {
        service.build_chunk_id(chunks[0]),
        service.build_chunk_id(chunks[1]),
    }


def test_save_chunks_writes_in_batches(monkeypatch):
    collection = FakeCollection()
    install_fake_chroma(monkeypatch, collection)
    chunks = [
        make_chunk(chunk_index=index)
        for index in range(5)
    ]

    saved_count = service.save_chunks(chunks, batch_size=2)

    assert saved_count == 5
    assert [
        len(item["ids"])
        for item in collection.upsert_calls
    ] == [2, 2, 1]
    assert collection.ids == {
        service.build_chunk_id(chunk)
        for chunk in chunks
    }


def test_save_chunks_writes_chunk_strategy(monkeypatch):
    collection = FakeCollection()
    install_fake_chroma(monkeypatch, collection)
    chunk = make_chunk()
    chunk["strategy"] = "recursive"

    service.save_chunks([chunk])

    metadata = collection.upsert_calls[0]["metadatas"][0]
    assert metadata["chunk_strategy"] == "recursive"


def test_save_chunks_writes_index_metadata(monkeypatch):
    collection = FakeCollection()
    install_fake_chroma(monkeypatch, collection)
    chunk = make_chunk()
    chunk.update({
        "chunk_size": 500,
        "chunk_overlap": 50,
        "chunk_version": "recursive:v1",
        "embedding_model": "BAAI/bge-small-zh-v1.5",
        "embedding_version": "v1",
        "index_version": "v1",
        "index_fingerprint": "fingerprint-1",
    })

    service.save_chunks([chunk])

    metadata = collection.upsert_calls[0]["metadatas"][0]
    assert metadata["index_fingerprint"] == "fingerprint-1"
    assert metadata["chunk_version"] == "recursive:v1"
    assert metadata["embedding_model"] == "BAAI/bge-small-zh-v1.5"


def test_versioned_save_keeps_previous_generation(monkeypatch):
    old_id = (
        "user-1_document-1_generation-old_page_1_chunk_0"
    )
    collection = FakeCollection({old_id})
    install_fake_chroma(monkeypatch, collection)
    chunk = make_chunk(index_generation_id="generation-new")

    saved_count = service.save_chunks([chunk])

    assert saved_count == 1
    assert collection.delete_calls == []
    assert old_id in collection.ids
    assert service.build_chunk_id(chunk) in collection.ids
    metadata = collection.upsert_calls[0]["metadatas"][0]
    assert metadata["index_generation_id"] == "generation-new"


def test_reindex_deletes_only_stale_chunk_ids(monkeypatch):
    kept_chunk = make_chunk(chunk_index=0)
    new_chunk = make_chunk(chunk_index=1)
    kept_id = service.build_chunk_id(kept_chunk)
    stale_id = "user-1_document-1_page_1_chunk_99"
    collection = FakeCollection({kept_id, stale_id})
    install_fake_chroma(monkeypatch, collection)

    saved_count = service.save_chunks([kept_chunk, new_chunk])

    assert saved_count == 2
    assert collection.delete_calls == [[stale_id]]
    assert collection.ids == {
        kept_id,
        service.build_chunk_id(new_chunk),
    }


def test_upsert_failure_does_not_delete_old_chunks(monkeypatch):
    old_id = "user-1_document-1_page_1_chunk_0"
    collection = FakeCollection(
        {old_id},
        fail_upsert=True,
    )
    install_fake_chroma(monkeypatch, collection)

    with pytest.raises(RuntimeError, match="simulated upsert failure"):
        service.save_chunks([make_chunk(chunk_index=1)])

    assert collection.delete_calls == []
    assert collection.ids == {old_id}


def test_second_batch_failure_can_be_retried(monkeypatch):
    old_id = "user-1_document-1_page_1_chunk_99"
    collection = FakeCollection(
        {old_id},
        fail_upsert_call=2,
    )
    install_fake_chroma(monkeypatch, collection)
    chunks = [
        make_chunk(chunk_index=index)
        for index in range(5)
    ]

    with pytest.raises(RuntimeError, match="simulated upsert failure"):
        service.save_chunks(chunks, batch_size=2)

    assert collection.delete_calls == []
    assert old_id in collection.ids

    collection.fail_upsert_call = None
    saved_count = service.save_chunks(chunks, batch_size=2)

    assert saved_count == 5
    assert collection.delete_calls == [[old_id]]
    assert collection.ids == {
        service.build_chunk_id(chunk)
        for chunk in chunks
    }


def test_save_chunks_rejects_invalid_batch_size(monkeypatch):
    persistent_client = Mock()
    monkeypatch.setattr(
        service.chromadb,
        "PersistentClient",
        persistent_client,
    )

    with pytest.raises(ValueError, match="batch_size 必须大于 0"):
        service.save_chunks([make_chunk()], batch_size=0)

    persistent_client.assert_not_called()


def test_save_chunks_rejects_duplicate_chunk_ids(monkeypatch):
    collection = FakeCollection()
    install_fake_chroma(monkeypatch, collection)
    chunk = make_chunk()

    with pytest.raises(ValueError, match="重复的 chunk_id"):
        service.save_chunks([chunk, dict(chunk)])

    assert collection.upsert_calls == []


def test_verification_failure_does_not_delete_old_chunks(monkeypatch):
    old_id = "user-1_document-1_page_1_chunk_99"
    new_chunk = make_chunk(chunk_index=0)
    new_id = service.build_chunk_id(new_chunk)
    collection = FakeCollection(
        {old_id},
        missing_after_upsert={new_id},
    )
    install_fake_chroma(monkeypatch, collection)

    with pytest.raises(RuntimeError, match="写入校验失败"):
        service.save_chunks([new_chunk])

    assert collection.delete_calls == []
    assert old_id in collection.ids


@pytest.mark.parametrize(
    ("changed_field", "changed_value", "expected_message"),
    [
        ("document_id", "document-2", "document_id"),
        ("user_id", "user-2", "user_id"),
    ],
)
def test_save_rejects_mixed_chunk_ownership(
    monkeypatch,
    changed_field,
    changed_value,
    expected_message,
):
    persistent_client = Mock()
    monkeypatch.setattr(
        service.chromadb,
        "PersistentClient",
        persistent_client,
    )
    first = make_chunk(chunk_index=0)
    second = make_chunk(chunk_index=1)
    second[changed_field] = changed_value

    with pytest.raises(ValueError, match=expected_message):
        service.save_chunks([first, second])

    persistent_client.assert_not_called()


@pytest.mark.parametrize(
    ("user_id", "document_id", "query_embedding", "n_results", "message"),
    [
        ("", "document-1", [0.1], 3, "user_id 不能为空"),
        ("user-1", "", [0.1], 3, "document_id 不能为空"),
        ("user-1", "document-1", [], 3, "query_embedding 不能为空"),
        ("user-1", "document-1", [0.1], 0, "n_results 必须大于 0"),
    ],
)
def test_query_rejects_invalid_input_before_opening_chroma(
    monkeypatch,
    user_id,
    document_id,
    query_embedding,
    n_results,
    message,
):
    persistent_client = Mock()
    monkeypatch.setattr(
        service.chromadb,
        "PersistentClient",
        persistent_client,
    )

    with pytest.raises(ValueError, match=message):
        service.query_chunks(
            user_id,
            document_id,
            query_embedding,
            n_results,
        )

    persistent_client.assert_not_called()


def test_query_filters_active_generation(monkeypatch):
    collection = Mock()
    collection.query.return_value = {
        "ids": [[]],
        "documents": [[]],
        "metadatas": [[]],
        "distances": [[]],
    }
    client = Mock()
    client.get_collection.return_value = collection
    monkeypatch.setattr(
        service.chromadb,
        "PersistentClient",
        Mock(return_value=client),
    )

    service.query_chunks(
        "user-1",
        "document-1",
        [0.1],
        3,
        "generation-active",
    )

    assert collection.query.call_args.kwargs["where"] == {
        "$and": [
            {"user_id": "user-1"},
            {
                "$and": [
                    {"document_id": "document-1"},
                    {
                        "index_generation_id": (
                            "generation-active"
                        )
                    },
                ]
            },
        ]
    }


def test_multi_document_query_filters_each_active_generation(monkeypatch):
    collection = Mock()
    collection.query.return_value = {
        "ids": [[]],
        "documents": [[]],
        "metadatas": [[]],
        "distances": [[]],
    }
    client = Mock()
    client.get_collection.return_value = collection
    monkeypatch.setattr(
        service.chromadb,
        "PersistentClient",
        Mock(return_value=client),
    )

    service.query_chunks_by_documents(
        "user-1",
        ["document-1", "document-2"],
        [0.1],
        5,
        {
            "document-1": "generation-a",
            "document-2": "generation-b",
        },
    )

    where = collection.query.call_args.kwargs["where"]
    assert where["$and"][0] == {"user_id": "user-1"}
    assert where["$and"][1] == {
        "$or": [
            {
                "$and": [
                    {"document_id": "document-1"},
                    {"index_generation_id": "generation-a"},
                ]
            },
            {
                "$and": [
                    {"document_id": "document-2"},
                    {"index_generation_id": "generation-b"},
                ]
            },
        ]
    }


def test_real_chroma_query_returns_only_active_generations(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setattr(service, "CHROMA_DIR", tmp_path)
    chunks = [
        make_chunk(
            document_id="document-1",
            text="old document one",
            index_generation_id="generation-old-1",
        ),
        make_chunk(
            document_id="document-1",
            text="active document one",
            index_generation_id="generation-active-1",
        ),
        make_chunk(
            document_id="document-2",
            text="old document two",
            index_generation_id="generation-old-2",
        ),
        make_chunk(
            document_id="document-2",
            text="active document two",
            index_generation_id="generation-active-2",
        ),
    ]
    for chunk in chunks:
        service.save_chunks([chunk])

    result = service.query_chunks_by_documents(
        "user-1",
        ["document-1", "document-2"],
        [0.1, 0.2],
        10,
        {
            "document-1": "generation-active-1",
            "document-2": "generation-active-2",
        },
    )

    assert set(result["documents"][0]) == {
        "active document one",
        "active document two",
    }
    assert {
        metadata["index_generation_id"]
        for metadata in result["metadatas"][0]
    } == {
        "generation-active-1",
        "generation-active-2",
    }


def test_failed_generation_cleanup_keeps_active_generation(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setattr(service, "CHROMA_DIR", tmp_path)
    failed_chunk = make_chunk(
        text="failed generation",
        index_generation_id="generation-failed",
    )
    active_chunk = make_chunk(
        text="active generation",
        index_generation_id="generation-active",
    )
    service.save_chunks([failed_chunk])
    service.save_chunks([active_chunk])

    deleted = service.delete_index_generation(
        "user-1",
        "document-1",
        "generation-failed",
    )

    assert deleted == 1
    active_result = service.query_chunks(
        "user-1",
        "document-1",
        [0.1, 0.2],
        3,
        "generation-active",
    )
    failed_result = service.query_chunks(
        "user-1",
        "document-1",
        [0.1, 0.2],
        3,
        "generation-failed",
    )
    assert active_result["documents"][0] == ["active generation"]
    assert failed_result["documents"][0] == []


def test_get_chunks_by_documents_uses_scoped_filter(monkeypatch):
    class ReadCollection:
        def __init__(self):
            self.get_kwargs = None

        def get(self, **kwargs):
            self.get_kwargs = kwargs
            return {
                "ids": ["chunk-1", "chunk-2"],
                "documents": ["RAG evidence", "LoRA evidence"],
                "metadatas": [
                    {
                        "user_id": "user-1",
                        "document_id": "document-1",
                    },
                    {
                        "user_id": "user-1",
                        "document_id": "document-2",
                    },
                ],
            }

    collection = ReadCollection()
    client = Mock()
    client.get_collection.return_value = collection
    monkeypatch.setattr(
        service.chromadb,
        "PersistentClient",
        Mock(return_value=client),
    )

    result = service.get_chunks_by_documents(
        " user-1 ",
        ["document-1", " document-2 ", "document-1"],
    )

    assert collection.get_kwargs == {
        "where": {
            "$and": [
                {"user_id": "user-1"},
                {
                    "document_id": {
                        "$in": ["document-1", "document-2"],
                    }
                },
            ]
        },
        "include": ["documents", "metadatas"],
    }
    assert result == [
        {
            "chunk_id": "chunk-1",
            "文本块": "RAG evidence",
            "元数据": {
                "user_id": "user-1",
                "document_id": "document-1",
            },
        },
        {
            "chunk_id": "chunk-2",
            "文本块": "LoRA evidence",
            "元数据": {
                "user_id": "user-1",
                "document_id": "document-2",
            },
        },
    ]


@pytest.mark.parametrize(
    ("user_id", "document_ids", "message"),
    [
        ("", ["document-1"], "user_id cannot be empty"),
        ("user-1", [], "document_ids cannot be empty"),
        ("user-1", [""], "document_id cannot be empty"),
    ],
)
def test_get_chunks_by_documents_rejects_invalid_scope(
    monkeypatch,
    user_id,
    document_ids,
    message,
):
    persistent_client = Mock()
    monkeypatch.setattr(
        service.chromadb,
        "PersistentClient",
        persistent_client,
    )

    with pytest.raises(ValueError, match=message):
        service.get_chunks_by_documents(user_id, document_ids)

    persistent_client.assert_not_called()
