from typing import Annotated

from fastapi import Depends
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from api_errors import APIError
from services.auth_service import AuthenticationError, get_user_from_token


bearer_scheme = HTTPBearer(auto_error=False)


async def get_current_user(
    credentials: Annotated[
        HTTPAuthorizationCredentials | None,
        Depends(bearer_scheme),
    ],
) -> dict:
    if credentials is None or credentials.scheme.lower() != "bearer":
        raise APIError(401, "AUTHENTICATION_REQUIRED", "请先登录")

    try:
        return await get_user_from_token(credentials.credentials)
    except AuthenticationError as error:
        raise APIError(401, "INVALID_ACCESS_TOKEN", str(error)) from error
    except RuntimeError as error:
        raise APIError(503, "AUTH_CONFIGURATION_ERROR", str(error)) from error


async def get_current_user_id(
    user: Annotated[dict, Depends(get_current_user)],
) -> str:
    return user["user_id"]


def require_matching_user(
    authenticated_user_id: str,
    requested_user_id: str,
) -> str:
    if requested_user_id.strip() != authenticated_user_id:
        raise APIError(
            403,
            "USER_IDENTITY_MISMATCH",
            "请求中的用户身份与访问令牌不一致",
        )
    return authenticated_user_id
