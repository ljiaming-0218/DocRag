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
) -> dict:
    return {
        "user_id": user_id,
        "document_id": document_id,
        "页码": page_number,
        "块索引": chunk_index,
        "文本块": text,
        "embedding": [0.1, 0.2],
        "提取方式": "text",
        "图片数量": 0,
    }


class FakeCollection:
    def __init__(
        self,
        existing_ids: set[str] | None = None,
        *,
        fail_upsert: bool = False,
        missing_after_upsert: set[str] | None = None,
    ) -> None:
        self.ids = set(existing_ids or set())
        self.fail_upsert = fail_upsert
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
        if self.fail_upsert:
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
