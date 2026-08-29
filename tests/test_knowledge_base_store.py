from unittest.mock import AsyncMock

import pytest

from stores import knowledge_base_store


class FakeAsyncCursor:
    def __init__(self, items: list[dict]):
        self._items = iter(items)

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            return next(self._items)
        except StopIteration as error:
            raise StopAsyncIteration from error


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "store_function, arguments",
    [
        (
            knowledge_base_store.find_documents_in_knowledge_base,
            ("user-1", "kb-1", 20),
        ),
        (
            knowledge_base_store.find_knowledge_base_document_records,
            ("user-1", "kb-1"),
        ),
    ],
)
async def test_knowledge_base_aggregate_is_awaited(
    monkeypatch,
    store_function,
    arguments,
):
    expected = [{"document_id": "document-1"}]
    aggregate = AsyncMock(return_value=FakeAsyncCursor(expected))
    database = {
        "knowledge_base_documents": type(
            "FakeCollection",
            (),
            {"aggregate": aggregate},
        )()
    }
    monkeypatch.setattr(
        knowledge_base_store,
        "get_database",
        lambda: database,
    )

    result = await store_function(*arguments)

    assert result == expected
    aggregate.assert_awaited_once()
