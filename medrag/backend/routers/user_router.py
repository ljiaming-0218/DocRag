
from fastapi import APIRouter, Query
from pydantic import BaseModel, Field

from api_errors import APIError
from services.document_service import list_user_documents
from services.user_service import create_or_get_user

router = APIRouter(
    prefix="/users",
    tags=["users"],
)

class CreateUserRequest(BaseModel):
    username: str = Field(min_length=1, max_length=30)
    default_user_type: str = "general"


@router.post("")
async def create_user_endpoint(user_request: CreateUserRequest) -> dict:
    try:
        user = await create_or_get_user(
            user_request.username,
            user_request.default_user_type,
        )
    except ValueError as error:
        raise APIError(
            status_code=400,
            code="INVALID_USER_REQUEST",
            message=str(error),
        ) from error

    return user


@router.get("/{user_id}/documents")
async def list_user_documents_endpoint(
    user_id: str,
    limit: int = Query(default=50, ge=1, le=100),
) -> list[dict]:
    try:
        documents = await list_user_documents(user_id, limit)
    except ValueError as error:
        raise APIError(
            status_code=404,
            code="USER_NOT_FOUND",
            message=str(error),
        ) from error

    return documents
