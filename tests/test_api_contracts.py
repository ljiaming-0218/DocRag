from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from routers import (
    conversation_router,
    pdf_router,
    user_router,
)
from services.llm_service import LLMServiceError

@pytest.fixture
def client():
    app = FastAPI()
    app.include_router(user_router.router)
    app.include_router(conversation_router.router)
    app.include_router(pdf_router.router)

    with TestClient(app) as test_client:
        yield test_client


def test_create_user_returns_user_contract(
    client: TestClient,
    monkeypatch,
):
    create_user = AsyncMock(return_value={
        "user_id": "user-1",
        "username": "test-user",
        "default_user_type": "general",
        "created": True,
    })
    monkeypatch.setattr(
        user_router,
        "create_or_get_user",
        create_user,
    )

    response = client.post(
        "/users",
        json={
            "username": "test-user",
            "default_user_type": "general",
        },
    )

    assert response.status_code == 200
    assert response.json() == {
        "user_id": "user-1",
        "username": "test-user",
        "default_user_type": "general",
        "created": True,
    }
    create_user.assert_awaited_once_with(
        "test-user",
        "general",
    )


def test_create_user_rejects_empty_username(
    client: TestClient,
    monkeypatch,
):
    create_user = AsyncMock()
    monkeypatch.setattr(
        user_router,
        "create_or_get_user",
        create_user,
    )

    response = client.post(
        "/users",
        json={
            "username": "",
            "default_user_type": "general",
        },
    )

    assert response.status_code == 422
    create_user.assert_not_awaited()


def test_list_messages_maps_permission_error_to_403(
    client: TestClient,
    monkeypatch,
):
    monkeypatch.setattr(
        conversation_router,
        "list_messages",
        AsyncMock(
            side_effect=PermissionError(
                "当前用户无权访问该会话"
            )
        ),
    )

    response = client.get(
        "/conversations/conversation-1/messages",
        params={"user_id": "user-2"},
    )

    assert response.status_code == 403
    assert "无权访问" in response.json()["detail"]


def test_ask_maps_llm_rate_limit_to_503(
    client: TestClient,
    monkeypatch,
):
    monkeypatch.setattr(
        conversation_router,
        "ask_conversation",
        AsyncMock(
            side_effect=LLMServiceError(
                "LLM_RATE_LIMITED",
                "大模型服务繁忙，请稍后重试",
                503,
            )
        ),
    )

    response = client.post(
        "/conversations/conversation-1/ask",
        json={
            "user_id": "user-1",
            "query": "总结这篇文档",
            "history_limit": 6,
            "n_results": 3,
        },
    )

    assert response.status_code == 503
    assert response.json()["detail"] == {
        "error": "LLM_RATE_LIMITED",
        "message": "大模型服务繁忙，请稍后重试",
    }


def test_index_pdf_maps_validation_error_to_400(
    client: TestClient,
    monkeypatch,
):
    index_document = AsyncMock(
        side_effect=ValueError(
            "文件内容不是有效的 PDF"
        )
    )
    monkeypatch.setattr(
        pdf_router,
        "index_document",
        index_document,
    )

    response = client.post(
        "/pdf/index",
        params={
            "user_id": "user-1",
            "chunk_size": 500,
            "chunk_overlap": 50,
        },
        files={
            "file": (
                "invalid.pdf",
                b"not a real pdf",
                "application/pdf",
            )
        },
    )

    assert response.status_code == 400
    assert "有效的 PDF" in response.json()["detail"]
    index_document.assert_awaited_once()
