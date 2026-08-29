from datetime import datetime, timezone
from uuid import uuid4

from pymongo.errors import DuplicateKeyError

from services.document_service import get_existing_document_for_user
from services.user_service import get_existing_user
from stores.knowledge_base_store import (
    delete_knowledge_base,
    delete_knowledge_base_document,
    delete_knowledge_base_links,
    find_documents_in_knowledge_base,
    find_knowledge_base_document_records,
    find_knowledge_base_by_id,
    find_knowledge_base_by_user_and_name,
    find_knowledge_bases_by_user,
    insert_knowledge_base,
    touch_knowledge_base,
    update_knowledge_base,
    upsert_knowledge_base_document,
)


def _serialize_knowledge_base(knowledge_base: dict) -> dict:
    return {
        "kb_id": knowledge_base["_id"],
        "user_id": knowledge_base["user_id"],
        "name": knowledge_base["name"],
        "description": knowledge_base.get("description", ""),
        "created_at": knowledge_base["created_at"],
        "updated_at": knowledge_base["updated_at"],
    }


def _clean_required(value: str, field_name: str) -> str:
    value = value.strip()
    if not value:
        raise ValueError(f"{field_name} cannot be empty")
    return value


async def get_knowledge_base_for_user(user_id: str, kb_id: str) -> dict:
    user_id = _clean_required(user_id, "user_id")
    kb_id = _clean_required(kb_id, "kb_id")

    knowledge_base = await find_knowledge_base_by_id(kb_id)
    if knowledge_base is None:
        raise LookupError("knowledge base not found")
    if knowledge_base["user_id"] != user_id:
        raise PermissionError("knowledge base does not belong to this user")
    return knowledge_base


async def create_knowledge_base(
    user_id: str,
    name: str,
    description: str = "",
) -> dict:
    user_id = _clean_required(user_id, "user_id")
    name = _clean_required(name, "name")
    description = description.strip()
    await get_existing_user(user_id)

    existing = await find_knowledge_base_by_user_and_name(user_id, name)
    if existing is not None:
        raise ValueError("knowledge base name already exists")

    now = datetime.now(timezone.utc)
    knowledge_base = {
        "_id": str(uuid4()),
        "user_id": user_id,
        "name": name,
        "description": description,
        "created_at": now,
        "updated_at": now,
    }
    try:
        await insert_knowledge_base(knowledge_base)
    except DuplicateKeyError as error:
        raise ValueError("knowledge base name already exists") from error
    return _serialize_knowledge_base(knowledge_base)


async def list_knowledge_bases(
    user_id: str,
    limit: int = 50,
) -> list[dict]:
    user_id = _clean_required(user_id, "user_id")
    if limit < 1 or limit > 100:
        raise ValueError("limit must be between 1 and 100")
    await get_existing_user(user_id)

    knowledge_bases = await find_knowledge_bases_by_user(user_id, limit)
    return [_serialize_knowledge_base(item) for item in knowledge_bases]


async def get_knowledge_base(user_id: str, kb_id: str) -> dict:
    knowledge_base = await get_knowledge_base_for_user(user_id, kb_id)
    return _serialize_knowledge_base(knowledge_base)


async def edit_knowledge_base(
    user_id: str,
    kb_id: str,
    name: str | None = None,
    description: str | None = None,
) -> dict:
    knowledge_base = await get_knowledge_base_for_user(user_id, kb_id)
    if name is None and description is None:
        raise ValueError("name or description is required")

    updates = {"updated_at": datetime.now(timezone.utc)}
    if name is not None:
        name = _clean_required(name, "name")
        existing = await find_knowledge_base_by_user_and_name(user_id, name)
        if existing is not None and existing["_id"] != kb_id:
            raise ValueError("knowledge base name already exists")
        updates["name"] = name
    if description is not None:
        updates["description"] = description.strip()

    try:
        updated = await update_knowledge_base(kb_id, user_id, updates)
    except DuplicateKeyError as error:
        raise ValueError("knowledge base name already exists") from error
    if not updated:
        raise LookupError("knowledge base not found")

    knowledge_base.update(updates)
    return _serialize_knowledge_base(knowledge_base)


async def remove_knowledge_base(user_id: str, kb_id: str) -> dict:
    await get_knowledge_base_for_user(user_id, kb_id)
    deleted = await delete_knowledge_base(kb_id, user_id)
    if not deleted:
        raise LookupError("knowledge base not found")
    deleted_links = await delete_knowledge_base_links(kb_id, user_id)
    return {
        "kb_id": kb_id,
        "deleted": True,
        "deleted_document_links": deleted_links,
    }


async def add_document_to_knowledge_base(
    user_id: str,
    kb_id: str,
    document_id: str,
) -> dict:
    knowledge_base = await get_knowledge_base_for_user(user_id, kb_id)
    document_id = _clean_required(document_id, "document_id")
    document = await get_existing_document_for_user(user_id, document_id)

    now = datetime.now(timezone.utc)
    linked = await upsert_knowledge_base_document(
        {
            "_id": str(uuid4()),
            "user_id": user_id,
            "kb_id": kb_id,
            "document_id": document_id,
            "added_at": now,
        }
    )
    if linked:
        await touch_knowledge_base(kb_id, user_id, now)

    return {
        "kb_id": kb_id,
        "document_id": document_id,
        "filename": document["filename"],
        "linked": linked,
        "knowledge_base_name": knowledge_base["name"],
    }


async def list_knowledge_base_documents(
    user_id: str,
    kb_id: str,
    limit: int = 100,
) -> list[dict]:
    await get_knowledge_base_for_user(user_id, kb_id)
    if limit < 1 or limit > 100:
        raise ValueError("limit must be between 1 and 100")

    items = await find_documents_in_knowledge_base(user_id, kb_id, limit)
    return [
        {
            "document_id": item["document"]["_id"],
            "user_id": item["document"]["user_id"],
            "kb_id": kb_id,
            "filename": item["document"]["filename"],
            "document_hash": item["document"]["document_hash"],
            "language": item["document"].get("language", "unknown"),
            "added_at": item["added_at"],
        }
        for item in items
    ]


async def resolve_knowledge_base_scope(
    user_id: str,
    kb_id: str,
    selected_document_ids: list[str] | None = None,
) -> dict:
    knowledge_base = await get_knowledge_base_for_user(user_id, kb_id)
    records = await find_knowledge_base_document_records(user_id, kb_id)

    documents_by_id = {
        item["document"]["_id"]: item["document"]
        for item in records
    }
    if not documents_by_id:
        raise ValueError("knowledge base has no documents")

    if selected_document_ids is None:
        document_ids = list(documents_by_id)
    else:
        document_ids = []
        for document_id in selected_document_ids:
            document_id = _clean_required(document_id, "document_id")
            if document_id not in document_ids:
                document_ids.append(document_id)
        if not document_ids:
            raise ValueError("selected_document_ids cannot be empty")

        unavailable = set(document_ids) - set(documents_by_id)
        if unavailable:
            raise PermissionError(
                "selected documents are not available in this knowledge base"
            )

    selected_documents = [
        documents_by_id[document_id]
        for document_id in document_ids
    ]
    languages = {
        document.get("language", "unknown")
        for document in selected_documents
        if document.get("language", "unknown") != "unknown"
    }
    language = languages.pop() if len(languages) == 1 else "unknown"

    return {
        "kb_id": knowledge_base["_id"],
        "knowledge_base_name": knowledge_base["name"],
        "document_ids": document_ids,
        "document_filenames": {
            document["_id"]: document["filename"]
            for document in selected_documents
        },
        "language": language,
    }


async def remove_document_from_knowledge_base(
    user_id: str,
    kb_id: str,
    document_id: str,
) -> dict:
    await get_knowledge_base_for_user(user_id, kb_id)
    document_id = _clean_required(document_id, "document_id")
    await get_existing_document_for_user(user_id, document_id)

    deleted = await delete_knowledge_base_document(
        user_id,
        kb_id,
        document_id,
    )
    if not deleted:
        raise LookupError("document is not linked to this knowledge base")

    await touch_knowledge_base(
        kb_id,
        user_id,
        datetime.now(timezone.utc),
    )
    return {
        "kb_id": kb_id,
        "document_id": document_id,
        "deleted": True,
    }
