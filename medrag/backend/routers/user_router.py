
from typing import Annotated

from fastapi import APIRouter, Depends, Query

from api_errors import APIError
from dependencies.auth import get_current_user_id, require_matching_user
from services.document_service import list_user_documents

router = APIRouter(
    prefix="/users",
    tags=["users"],
)

@router.get("/{user_id}/documents")
async def list_user_documents_endpoint(
    user_id: str,
    authenticated_user_id: Annotated[str, Depends(get_current_user_id)],
    limit: int = Query(default=50, ge=1, le=100),
) -> list[dict]:
    try:
        user_id = require_matching_user(authenticated_user_id, user_id)
        documents = await list_user_documents(user_id, limit)
    except APIError:
        raise
    except ValueError as error:
        raise APIError(
            status_code=404,
            code="USER_NOT_FOUND",
            message=str(error),
        ) from error

    return documents
