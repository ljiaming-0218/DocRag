

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

async def find_documents_by_user(user_id) -> list:
    database = get_database()
    document_collection = database["documents"]

    cursor = document_collection.find({"user_id": user_id})
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
