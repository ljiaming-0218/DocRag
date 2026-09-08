from datetime import datetime, timezone
from unittest.mock import AsyncMock, Mock

import pytest

from services import auth_service


def make_user() -> dict:
    now = datetime.now(timezone.utc)
    return {
        "_id": "user-1",
        "username": "alice",
        "password_hash": "stored-hash",
        "default_user_type": "general",
        "created_at": now,
        "updated_at": now,
    }


def test_access_token_uses_immutable_user_id_as_subject(monkeypatch):
    monkeypatch.setattr(
        auth_service,
        "JWT_SECRET_KEY",
        "test-secret-key-with-at-least-32-bytes",
    )

    token = auth_service.create_access_token("user-1")

    assert auth_service.decode_access_token(token) == "user-1"


@pytest.mark.asyncio
async def test_register_hashes_password_before_storage(monkeypatch):
    insert = AsyncMock()
    monkeypatch.setattr(
        auth_service,
        "find_user_by_username",
        AsyncMock(return_value=None),
    )
    monkeypatch.setattr(auth_service, "insert_user", insert)
    monkeypatch.setattr(
        auth_service,
        "hash_password",
        Mock(return_value="hashed-password"),
    )
    monkeypatch.setattr(
        auth_service,
        "create_access_token",
        Mock(return_value="token-1"),
    )

    result = await auth_service.register_user(
        "alice",
        "password-123",
        "general",
    )

    stored_user = insert.await_args.args[0]
    assert stored_user["password_hash"] == "hashed-password"
    assert "password" not in stored_user
    assert result["access_token"] == "token-1"
    assert "password_hash" not in result["user"]


@pytest.mark.asyncio
async def test_register_rejects_existing_username(monkeypatch):
    monkeypatch.setattr(
        auth_service,
        "find_user_by_username",
        AsyncMock(return_value=make_user()),
    )

    with pytest.raises(auth_service.UserAlreadyExistsError):
        await auth_service.register_user(
            "alice",
            "password-123",
        )


@pytest.mark.asyncio
async def test_login_returns_token_for_valid_password(monkeypatch):
    monkeypatch.setattr(
        auth_service,
        "find_user_by_username",
        AsyncMock(return_value=make_user()),
    )
    monkeypatch.setattr(
        auth_service,
        "verify_password",
        Mock(return_value=True),
    )
    monkeypatch.setattr(
        auth_service,
        "create_access_token",
        Mock(return_value="token-1"),
    )

    result = await auth_service.login_user("alice", "password-123")

    assert result["access_token"] == "token-1"
    assert result["user"]["user_id"] == "user-1"


@pytest.mark.asyncio
async def test_login_hides_whether_username_or_password_is_wrong(monkeypatch):
    monkeypatch.setattr(
        auth_service,
        "find_user_by_username",
        AsyncMock(return_value=None),
    )

    with pytest.raises(
        auth_service.AuthenticationError,
        match="用户名或密码错误",
    ):
        await auth_service.login_user("unknown", "password-123")


def test_access_token_rejects_short_secret(monkeypatch):
    monkeypatch.setattr(auth_service, "JWT_SECRET_KEY", "short")

    with pytest.raises(RuntimeError, match="至少为 32 字节"):
        auth_service.create_access_token("user-1")
