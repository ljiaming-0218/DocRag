from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api_errors import APIError, api_error_handler
from dependencies.auth import get_current_user_id
from routers import auth_router, conversation_router


@pytest.fixture
def auth_client():
    app = FastAPI()
    app.include_router(auth_router.router)
    app.add_exception_handler(APIError, api_error_handler)
    with TestClient(app) as client:
        yield client


def test_login_maps_invalid_credentials_to_401(
    auth_client,
    monkeypatch,
):
    monkeypatch.setattr(
        auth_router,
        "login_user",
        AsyncMock(
            side_effect=auth_router.AuthenticationError(
                "用户名或密码错误"
            )
        ),
    )

    response = auth_client.post(
        "/auth/login",
        json={"username": "alice", "password": "password-123"},
    )

    assert response.status_code == 401
    assert response.json()["error"] == "INVALID_CREDENTIALS"


def test_protected_endpoint_requires_bearer_token():
    app = FastAPI()
    app.include_router(conversation_router.router)
    app.add_exception_handler(APIError, api_error_handler)

    with TestClient(app) as client:
        response = client.get(
            "/conversations",
            params={"user_id": "user-1"},
        )

    assert response.status_code == 401
    assert response.json()["error"] == "AUTHENTICATION_REQUIRED"


def test_forged_user_id_is_rejected_before_service(monkeypatch):
    app = FastAPI()
    app.include_router(conversation_router.router)
    app.add_exception_handler(APIError, api_error_handler)
    app.dependency_overrides[get_current_user_id] = lambda: "user-1"
    list_conversations = AsyncMock(return_value=[])
    monkeypatch.setattr(
        conversation_router,
        "list_conversations",
        list_conversations,
    )

    with TestClient(app) as client:
        response = client.get(
            "/conversations",
            params={"user_id": "user-2"},
        )

    assert response.status_code == 403
    assert response.json()["error"] == "USER_IDENTITY_MISMATCH"
    list_conversations.assert_not_awaited()
