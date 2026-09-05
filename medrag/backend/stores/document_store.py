

from datetime import datetime

from stores.database import get_database


async def find_document_by_user_and_hash(user_id, document_hash) -> dict:
    database = get_database()
    document_collection = database["documents"]

    document = await document_collection.find_one({"user_id": user_id, "document_hash": document_hash})
    return document

async def find_document_by_user_and_id(user_id, document_id) -> dict:
    database = get_database()
    document_collection = database["documents"]

    document = await document_collection.find_one({"user_id": user_id, "_id": document_id})
    return document

async def insert_document(document) -> None:
    database = get_database()
    document_collection = database["documents"]
    await document_collection.insert_one(document)

async def find_documents_by_user(
    user_id: str,
    limit: int = 50,
) -> list[dict]:
    database = get_database()
    collection = database["documents"]

    cursor = (
        collection.find({"user_id": user_id})
        .sort("updated_at", -1)
        .limit(limit)
    )

    documents = []
    async for document in cursor:
        documents.append(document)

    return documents

async def create_document_indexes() -> None:
    database = get_database()
    document_collection = database["documents"]

    await document_collection.create_index(
        [("user_id", 1), ("document_hash", 1)],
        name="user_id_document_hash_unique_idx",
        unique=True
    )
    await document_collection.create_index(
        [
            ("user_id", 1),
            ("updated_at", -1),
        ],
        name="user_id_updated_at_idx",
    )

async def update_document_language(
    user_id: str,
    document_id: str,
    language: str,
) -> bool:
    database = get_database()
    collection = database["documents"]

    result = await collection.update_one(
        {
            "_id": document_id,
            "user_id": user_id,
        },
        {
            "$set": {
                "language": language,
            }
        },
    )

    return result.matched_count == 1


async def update_document_index_state(
    user_id: str,
    document_id: str,
    index_fingerprint: str,
    index_config: dict,
    indexed_at: datetime,
    active_index_generation_id: str | None = None,
) -> bool:
    database = get_database()
    collection = database["documents"]

    completed_fields = {
        "index_fingerprint": index_fingerprint,
        "index_config": index_config,
        "indexed_at": indexed_at,
        "processing_status": "completed",
        "processing_stage": "completed",
        "processed_at": indexed_at,
        "updated_at": indexed_at,
    }
    if active_index_generation_id is not None:
        completed_fields["active_index_generation_id"] = (
            active_index_generation_id
        )

    result = await collection.update_one(
        {
            "_id": document_id,
            "user_id": user_id,
        },
        {
            "$set": completed_fields,
            "$unset": {
                "error_stage": "",
                "error_code": "",
                "error_message": "",
            },
        },
    )
    return result.matched_count == 1


async def update_document_processing_state(
    user_id: str,
    document_id: str,
    processing_status: str,
    processing_stage: str,
    updated_at: datetime,
    *,
    processing_started_at: datetime | None = None,
    error_stage: str | None = None,
    error_code: str | None = None,
    error_message: str | None = None,
    clear_error: bool = False,
) -> bool:
    database = get_database()
    collection = database["documents"]

    fields = {
        "processing_status": processing_status,
        "processing_stage": processing_stage,
        "updated_at": updated_at,
    }
    if processing_started_at is not None:
        fields["processing_started_at"] = processing_started_at
    if error_stage is not None:
        fields["error_stage"] = error_stage
    if error_code is not None:
        fields["error_code"] = error_code
    if error_message is not None:
        fields["error_message"] = error_message

    update = {"$set": fields}
    if clear_error:
        update["$unset"] = {
            "error_stage": "",
            "error_code": "",
            "error_message": "",
        }

    result = await collection.update_one(
        {
            "_id": document_id,
            "user_id": user_id,
        },
        update,
    )
    return result.matched_count == 1
