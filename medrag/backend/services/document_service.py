from datetime import datetime, timezone
from uuid import uuid4

from stores.document_store import find_document_by_user_and_hash, find_document_by_user_and_id, insert_document, update_document_language

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