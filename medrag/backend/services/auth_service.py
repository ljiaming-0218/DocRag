from datetime import datetime, timedelta, timezone
from uuid import uuid4

import jwt
from jwt.exceptions import InvalidTokenError
from pwdlib import PasswordHash
from pymongo.errors import DuplicateKeyError

from config import (
    JWT_ACCESS_TOKEN_EXPIRE_MINUTES,
    JWT_ALGORITHM,
    JWT_SECRET_KEY,
)
from services.user_service import ALLOWED_USER_TYPES, format_user_response
from stores.user_store import (
    find_user_by_id,
    find_user_by_username,
    insert_user,
)


password_hasher = PasswordHash.recommended()


class AuthenticationError(ValueError):
    pass


class UserAlreadyExistsError(ValueError):
    pass


def hash_password(password: str) -> str:
    return password_hasher.hash(password)


def verify_password(password: str, password_hash: str) -> bool:
    try:
        return password_hasher.verify(password, password_hash)
    except Exception:
        return False


def _validate_password(password: str) -> str:
    if len(password) < 8:
        raise ValueError("密码长度不能少于 8 位")
    if len(password) > 128:
        raise ValueError("密码长度不能超过 128 位")
    return password


def _normalize_username(username: str) -> str:
    username = username.strip()
    if not username:
        raise ValueError("用户名不能为空")
    if len(username) > 30:
        raise ValueError("用户名长度不能超过 30")
    return username


def _normalize_user_type(user_type: str) -> str:
    user_type = user_type.strip().lower()
    if user_type not in ALLOWED_USER_TYPES:
        raise ValueError(f"非法的用户类型: {user_type}")
    return user_type


def create_access_token(user_id: str) -> str:
    if len(JWT_SECRET_KEY) < 32:
        raise RuntimeError("JWT_SECRET_KEY 必须配置且长度至少为 32 字节")

    now = datetime.now(timezone.utc)
    payload = {
        "sub": user_id,
        "iat": now,
        "exp": now + timedelta(
            minutes=JWT_ACCESS_TOKEN_EXPIRE_MINUTES,
        ),
    }
    return jwt.encode(
        payload,
        JWT_SECRET_KEY,
        algorithm=JWT_ALGORITHM,
    )


def decode_access_token(token: str) -> str:
    if len(JWT_SECRET_KEY) < 32:
        raise RuntimeError("JWT_SECRET_KEY 必须配置且长度至少为 32 字节")

    try:
        payload = jwt.decode(
            token,
            JWT_SECRET_KEY,
            algorithms=[JWT_ALGORITHM],
        )
    except InvalidTokenError as error:
        raise AuthenticationError("无效或已过期的访问令牌") from error

    user_id = payload.get("sub")
    if not isinstance(user_id, str) or not user_id.strip():
        raise AuthenticationError("访问令牌缺少用户身份")
    return user_id


def _build_auth_response(user: dict) -> dict:
    return {
        "access_token": create_access_token(user["_id"]),
        "token_type": "bearer",
        "user": format_user_response(user, False),
    }


async def register_user(
    username: str,
    password: str,
    default_user_type: str = "general",
) -> dict:
    username = _normalize_username(username)
    password = _validate_password(password)
    default_user_type = _normalize_user_type(default_user_type)

    if await find_user_by_username(username):
        raise UserAlreadyExistsError("用户名已存在")

    now = datetime.now(timezone.utc)
    user = {
        "_id": str(uuid4()),
        "username": username,
        "password_hash": hash_password(password),
        "default_user_type": default_user_type,
        "created_at": now,
        "updated_at": now,
    }
    try:
        await insert_user(user)
    except DuplicateKeyError as error:
        raise UserAlreadyExistsError("用户名已存在") from error

    return _build_auth_response(user)


async def login_user(username: str, password: str) -> dict:
    username = _normalize_username(username)
    _validate_password(password)

    user = await find_user_by_username(username)
    password_hash = user.get("password_hash") if user else None
    if (
        not user
        or not isinstance(password_hash, str)
        or not verify_password(password, password_hash)
    ):
        raise AuthenticationError("用户名或密码错误")

    return _build_auth_response(user)


async def get_user_from_token(token: str) -> dict:
    user_id = decode_access_token(token)
    user = await find_user_by_id(user_id)
    if not user:
        raise AuthenticationError("访问令牌对应的用户不存在")
    return format_user_response(user, False)
