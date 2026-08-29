from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.testclient import TestClient

from api_errors import (
    APIError,
    api_error_handler,
    request_validation_error_handler,
)
from routers import knowledge_base_router


@pytest.fixture
def client():
    app = FastAPI()
    app.include_router(knowledge_base_router.router)
    app.add_exception_handler(APIError, api_error_handler)
    app.add_exception_handler(
        RequestValidationError,
        request_validation_error_handler,
    )
    with TestClient(app) as test_client:
        yield test_client


def test_create_knowledge_base_contract(client, monkeypatch):
    create = AsyncMock(return_value={
        "kb_id": "kb-1",
        "user_id": "user-1",
        "name": "Research Papers",
        "description": "RAG papers",
        "created_at": "2026-08-28T10:00:00Z",
        "updated_at": "2026-08-28T10:00:00Z",
    })
    monkeypatch.setattr(
        knowledge_base_router,
        "create_knowledge_base",
        create,
    )

    response = client.post(
        "/knowledge-bases",
        json={
            "user_id": "user-1",
            "name": "Research Papers",
            "description": "RAG papers",
        },
    )

    assert response.status_code == 201
    assert response.json()["kb_id"] == "kb-1"
    create.assert_awaited_once_with(
        "user-1",
        "Research Papers",
        "RAG papers",
    )


def test_create_knowledge_base_rejects_empty_name(client, monkeypatch):
    create = AsyncMock()
    monkeypatch.setattr(
        knowledge_base_router,
        "create_knowledge_base",
        create,
    )

    response = client.post(
        "/knowledge-bases",
        json={
            "user_id": "user-1",
            "name": "",
        },
    )

    assert response.status_code == 422
    create.assert_not_awaited()


def test_get_knowledge_base_maps_cross_user_access_to_403(
    client,
    monkeypatch,
):
    monkeypatch.setattr(
        knowledge_base_router,
        "get_knowledge_base",
        AsyncMock(side_effect=PermissionError("forbidden")),
    )

    response = client.get(
        "/knowledge-bases/kb-1",
        params={"user_id": "user-2"},
    )

    assert response.status_code == 403
    assert response.json() == {
        "error": "KNOWLEDGE_BASE_FORBIDDEN",
        "message": "forbidden",
    }


def test_get_knowledge_base_maps_missing_resource_to_404(
    client,
    monkeypatch,
):
    monkeypatch.setattr(
        knowledge_base_router,
        "get_knowledge_base",
        AsyncMock(side_effect=LookupError("not found")),
    )

    response = client.get(
        "/knowledge-bases/kb-404",
        params={"user_id": "user-1"},
    )

    assert response.status_code == 404
    assert response.json() == {
        "error": "KNOWLEDGE_BASE_NOT_FOUND",
        "message": "not found",
    }


def test_add_document_to_knowledge_base_contract(client, monkeypatch):
    add_document = AsyncMock(return_value={
        "kb_id": "kb-1",
        "document_id": "document-1",
        "filename": "paper.pdf",
        "linked": True,
        "knowledge_base_name": "Research Papers",
    })
    monkeypatch.setattr(
        knowledge_base_router,
        "add_document_to_knowledge_base",
        add_document,
    )

    response = client.post(
        "/knowledge-bases/kb-1/documents/document-1",
        json={"user_id": "user-1"},
    )

    assert response.status_code == 200
    assert response.json()["linked"] is True
    add_document.assert_awaited_once_with(
        "user-1",
        "kb-1",
        "document-1",
    )


def test_list_knowledge_base_documents_contract(client, monkeypatch):
    list_documents = AsyncMock(return_value=[{
        "document_id": "document-1",
        "kb_id": "kb-1",
        "filename": "paper.pdf",
    }])
    monkeypatch.setattr(
        knowledge_base_router,
        "list_knowledge_base_documents",
        list_documents,
    )

    response = client.get(
        "/knowledge-bases/kb-1/documents",
        params={
            "user_id": "user-1",
            "limit": 20,
        },
    )

    assert response.status_code == 200
    assert response.json()[0]["document_id"] == "document-1"
    list_documents.assert_awaited_once_with(
        "user-1",
        "kb-1",
        20,
    )


def test_remove_unlinked_document_maps_to_404(client, monkeypatch):
    monkeypatch.setattr(
        knowledge_base_router,
        "remove_document_from_knowledge_base",
        AsyncMock(side_effect=LookupError("document is not linked")),
    )

    response = client.delete(
        "/knowledge-bases/kb-1/documents/document-1",
        params={"user_id": "user-1"},
    )

    assert response.status_code == 404
    assert response.json()["error"] == "KNOWLEDGE_BASE_NOT_FOUND"
