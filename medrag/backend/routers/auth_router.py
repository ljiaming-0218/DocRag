from typing import Annotated

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from api_errors import APIError
from dependencies.auth import get_current_user
from services.auth_service import (
    AuthenticationError,
    UserAlreadyExistsError,
    login_user,
    register_user,
)


router = APIRouter(prefix="/auth", tags=["auth"])


class RegisterRequest(BaseModel):
    username: str = Field(min_length=1, max_length=30)
    password: str = Field(min_length=8, max_length=128)
    default_user_type: str = "general"


class LoginRequest(BaseModel):
    username: str = Field(min_length=1, max_length=30)
    password: str = Field(min_length=8, max_length=128)


@router.post("/register", status_code=201)
async def register_endpoint(request: RegisterRequest) -> dict:
    try:
        return await register_user(
            request.username,
            request.password,
            request.default_user_type,
        )
    except UserAlreadyExistsError as error:
        raise APIError(409, "USERNAME_ALREADY_EXISTS", str(error)) from error
    except RuntimeError as error:
        raise APIError(503, "AUTH_CONFIGURATION_ERROR", str(error)) from error
    except ValueError as error:
        raise APIError(400, "INVALID_REGISTRATION", str(error)) from error


@router.post("/login")
async def login_endpoint(request: LoginRequest) -> dict:
    try:
        return await login_user(request.username, request.password)
    except AuthenticationError as error:
        raise APIError(401, "INVALID_CREDENTIALS", str(error)) from error
    except RuntimeError as error:
        raise APIError(503, "AUTH_CONFIGURATION_ERROR", str(error)) from error
    except ValueError as error:
        raise APIError(400, "INVALID_LOGIN_REQUEST", str(error)) from error


@router.get("/me")
async def current_user_endpoint(
    user: Annotated[dict, Depends(get_current_user)],
) -> dict:
    return user
