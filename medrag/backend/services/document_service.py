from datetime import datetime, timezone
from uuid import uuid4

from services.user_service import get_existing_user
from stores.document_store import (
    find_document_by_user_and_hash,
    find_document_by_user_and_id,
    find_documents_by_user,
    insert_document,
    update_document_index_state,
    update_document_language,
    update_document_processing_state,
)


PROCESSING_STATUSES = {"pending", "processing", "completed", "failed"}
PROCESSING_STAGES = {
    "pending",
    "parsing",
    "chunking",
    "embedding",
    "indexing",
    "completed",
}


class DocumentNotReadyError(RuntimeError):
    def __init__(
        self,
        document_id: str,
        processing_status: str,
        processing_stage: str,
        message: str,
    ) -> None:
        super().__init__(message)
        self.document_id = document_id
        self.processing_status = processing_status
        self.processing_stage = processing_stage


def get_document_processing_status(document: dict) -> str:
    processing_status = document.get("processing_status")
    if processing_status:
        return processing_status
    if document.get("index_fingerprint") or document.get("indexed_at"):
        return "completed"
    return "pending"


def has_active_index(document: dict) -> bool:
    """Return whether the document has a queryable index generation."""
    if document.get("active_index_generation_id"):
        return True

    processing_status = document.get("processing_status")
    if processing_status not in {None, "completed"}:
        return False
    return bool(
        document.get("index_fingerprint")
        or document.get("indexed_at")
    )


def ensure_document_ready(document: dict) -> None:
    document_id = str(document.get("_id", ""))
    processing_status = get_document_processing_status(document)
    processing_stage = document.get(
        "processing_stage",
        processing_status,
    )
    if processing_status == "completed" or has_active_index(document):
        return

    if processing_status == "failed":
        message = "文档索引失败，请重新索引后再提问"
    else:
        message = (
            f"文档索引尚未完成，当前阶段: {processing_stage}，"
            "请稍后再试"
        )
    raise DocumentNotReadyError(
        document_id,
        processing_status,
        processing_stage,
        message,
    )


async def get_or_create_document(user_id, filename, document_hash) -> dict:
    user_id = user_id.strip()
    if not user_id:
        raise ValueError("user_id 不能为空")
    if not filename:
        raise ValueError("filename 不能为空")
    document_hash = document_hash.strip()
    if not document_hash:
        raise ValueError("document_hash 不能为空")
    
    existing_document=True
    document = await find_document_by_user_and_hash(user_id, document_hash)
    if document is None:
        now = datetime.now(timezone.utc)
        existing_document=False
        document_id = str(uuid4())
        document = {
            "_id": document_id,
            "user_id": user_id,
            "filename": filename,
            "document_hash": document_hash,
            "created_at": now,
            "updated_at": now,
            "language": "unknown",
            "index_fingerprint": None,
            "active_index_generation_id": None,
            "index_config": None,
            "indexed_at": None,
            "processing_status": "pending",
            "processing_stage": "pending",
            "processing_started_at": None,
            "processed_at": None,
        }
        await insert_document(document)
        return {
            "document_id": document["_id"],
            "user_id": document["user_id"],
            "filename": document["filename"],
            "document_hash": document["document_hash"],
            "existing_document": existing_document,
            "created_at": document["created_at"],
            "updated_at": document["updated_at"],
            "language": document.get("language", "unknown"),
            "index_fingerprint": document.get("index_fingerprint"),
            "active_index_generation_id": document.get(
                "active_index_generation_id"
            ),
            "index_config": document.get("index_config"),
            "indexed_at": document.get("indexed_at"),
            "processing_status": document.get(
                "processing_status",
                "pending",
            ),
            "processing_stage": document.get(
                "processing_stage",
                "pending",
            ),
        }


    return {
        "document_id": document["_id"],
        "user_id": document["user_id"],
        "filename": document["filename"],
        "document_hash": document["document_hash"],
        "existing_document": existing_document,
        "created_at": document["created_at"],
        "updated_at": document["updated_at"],
        "language": document.get("language", "unknown"),
        "index_fingerprint": document.get("index_fingerprint"),
        "active_index_generation_id": document.get(
            "active_index_generation_id"
        ),
        "index_config": document.get("index_config"),
        "indexed_at": document.get("indexed_at"),
        "processing_status": document.get(
            "processing_status",
            "completed" if document.get("index_fingerprint") else "pending",
        ),
        "processing_stage": document.get(
            "processing_stage",
            "completed" if document.get("index_fingerprint") else "pending",
        ),
    }


async def get_existing_document_for_user(user_id, document_id) -> dict:
    user_id = user_id.strip()
    if not user_id:
        raise ValueError("user_id 不能为空")
    
    document_id = document_id.strip()
    if not document_id:
        raise ValueError("document_id 不能为空")
    
    document = await find_document_by_user_and_id(user_id, document_id)
    if not document:
        raise ValueError("文档不存在或不属于当前用户")
    return document

async def set_document_language(
    user_id: str,
    document_id: str,
    language: str,
) -> None:
    if language not in {"zh", "en", "unknown"}:
        raise ValueError("非法的文档语言")

    updated = await update_document_language(
        user_id,
        document_id,
        language,
    )

    if not updated:
        raise ValueError("文档不存在或不属于当前用户")


async def set_document_index_state(
    user_id: str,
    document_id: str,
    index_fingerprint: str,
    index_config: dict,
    indexed_at: datetime,
    active_index_generation_id: str | None = None,
) -> None:
    if active_index_generation_id is None:
        updated = await update_document_index_state(
            user_id,
            document_id,
            index_fingerprint,
            index_config,
            indexed_at,
        )
    else:
        updated = await update_document_index_state(
            user_id,
            document_id,
            index_fingerprint,
            index_config,
            indexed_at,
            active_index_generation_id=active_index_generation_id,
        )
    if not updated:
        raise ValueError("文档不存在或不属于当前用户")


async def set_document_processing_state(
    user_id: str,
    document_id: str,
    processing_status: str,
    processing_stage: str,
    *,
    error_stage: str | None = None,
    error_code: str | None = None,
    error_message: str | None = None,
) -> None:
    if processing_status not in PROCESSING_STATUSES:
        raise ValueError("非法的文档处理状态")
    if processing_stage not in PROCESSING_STAGES:
        raise ValueError("非法的文档处理阶段")
    if processing_status == "failed":
        if not error_stage or not error_code:
            raise ValueError("失败状态必须包含错误阶段和错误代码")
    elif error_stage or error_code or error_message:
        raise ValueError("非失败状态不能包含错误信息")

    now = datetime.now(timezone.utc)
    updated = await update_document_processing_state(
        user_id,
        document_id,
        processing_status,
        processing_stage,
        now,
        processing_started_at=(
            now
            if processing_status == "processing"
            and processing_stage == "parsing"
            else None
        ),
        error_stage=error_stage,
        error_code=error_code,
        error_message=error_message,
        clear_error=processing_status != "failed",
    )
    if not updated:
        raise ValueError("文档不存在或不属于当前用户")

async def list_user_documents(
    user_id: str,
    limit: int = 50,
) -> list[dict]:
    user_id = user_id.strip()
    if not user_id:
        raise ValueError("user_id 不能为空")
    if limit < 1 or limit > 100:
        raise ValueError("limit 必须在 1 到 100 之间")

    await get_existing_user(user_id)


    documents = await find_documents_by_user(user_id, limit)

    return [
        {
            "document_id": document["_id"],
            "user_id": document["user_id"],
            "filename": document["filename"],
            "document_hash": document["document_hash"],
            "language": document.get("language", "unknown"),
            "created_at": document["created_at"],
            "updated_at": document["updated_at"],
            "processing_status": document.get(
                "processing_status",
                "completed" if document.get("index_fingerprint") else "pending",
            ),
            "processing_stage": document.get(
                "processing_stage",
                "completed" if document.get("index_fingerprint") else "pending",
            ),
            "error_stage": document.get("error_stage"),
            "error_code": document.get("error_code"),
            "error_message": document.get("error_message"),
        }
        for document in documents
    ]
