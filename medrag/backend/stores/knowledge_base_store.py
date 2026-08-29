from datetime import datetime

from stores.database import get_database


async def insert_knowledge_base(knowledge_base: dict) -> None:
    database = get_database()
    await database["knowledge_bases"].insert_one(knowledge_base)


async def find_knowledge_base_by_id(kb_id: str) -> dict | None:
    database = get_database()
    return await database["knowledge_bases"].find_one({"_id": kb_id})


async def find_knowledge_base_by_user_and_name(
    user_id: str,
    name: str,
) -> dict | None:
    database = get_database()
    return await database["knowledge_bases"].find_one(
        {
            "user_id": user_id,
            "name": name,
        }
    )


async def find_knowledge_bases_by_user(
    user_id: str,
    limit: int,
) -> list[dict]:
    database = get_database()
    cursor = (
        database["knowledge_bases"]
        .find({"user_id": user_id})
        .sort("updated_at", -1)
        .limit(limit)
    )

    results = []
    async for knowledge_base in cursor:
        results.append(knowledge_base)
    return results


async def update_knowledge_base(
    kb_id: str,
    user_id: str,
    updates: dict,
) -> bool:
    database = get_database()
    result = await database["knowledge_bases"].update_one(
        {
            "_id": kb_id,
            "user_id": user_id,
        },
        {"$set": updates},
    )
    return result.matched_count == 1


async def delete_knowledge_base(kb_id: str, user_id: str) -> bool:
    database = get_database()
    result = await database["knowledge_bases"].delete_one(
        {
            "_id": kb_id,
            "user_id": user_id,
        }
    )
    return result.deleted_count == 1


async def delete_knowledge_base_links(kb_id: str, user_id: str) -> int:
    database = get_database()
    result = await database["knowledge_base_documents"].delete_many(
        {
            "kb_id": kb_id,
            "user_id": user_id,
        }
    )
    return result.deleted_count


async def upsert_knowledge_base_document(link: dict) -> bool:
    database = get_database()
    result = await database["knowledge_base_documents"].update_one(
        {
            "user_id": link["user_id"],
            "kb_id": link["kb_id"],
            "document_id": link["document_id"],
        },
        {"$setOnInsert": link},
        upsert=True,
    )
    return result.upserted_id is not None


async def delete_knowledge_base_document(
    user_id: str,
    kb_id: str,
    document_id: str,
) -> bool:
    database = get_database()
    result = await database["knowledge_base_documents"].delete_one(
        {
            "user_id": user_id,
            "kb_id": kb_id,
            "document_id": document_id,
        }
    )
    return result.deleted_count == 1


async def find_documents_in_knowledge_base(
    user_id: str,
    kb_id: str,
    limit: int,
) -> list[dict]:
    database = get_database()
    pipeline = [
        {
            "$match": {
                "user_id": user_id,
                "kb_id": kb_id,
            }
        },
        {"$sort": {"added_at": -1}},
        {"$limit": limit},
        {
            "$lookup": {
                "from": "documents",
                "localField": "document_id",
                "foreignField": "_id",
                "as": "document",
            }
        },
        {"$unwind": "$document"},
        {"$match": {"document.user_id": user_id}},
    ]

    results = []
    cursor = await database["knowledge_base_documents"].aggregate(pipeline)
    async for item in cursor:
        results.append(item)
    return results


async def find_knowledge_base_document_records(
    user_id: str,
    kb_id: str,
) -> list[dict]:
    database = get_database()
    pipeline = [
        {
            "$match": {
                "user_id": user_id,
                "kb_id": kb_id,
            }
        },
        {
            "$lookup": {
                "from": "documents",
                "localField": "document_id",
                "foreignField": "_id",
                "as": "document",
            }
        },
        {"$unwind": "$document"},
        {"$match": {"document.user_id": user_id}},
    ]

    results = []
    cursor = await database["knowledge_base_documents"].aggregate(pipeline)
    async for item in cursor:
        results.append(item)
    return results


async def touch_knowledge_base(
    kb_id: str,
    user_id: str,
    updated_at: datetime,
) -> None:
    database = get_database()
    await database["knowledge_bases"].update_one(
        {
            "_id": kb_id,
            "user_id": user_id,
        },
        {"$set": {"updated_at": updated_at}},
    )


async def create_knowledge_base_indexes() -> None:
    database = get_database()
    knowledge_bases = database["knowledge_bases"]
    links = database["knowledge_base_documents"]

    await knowledge_bases.create_index(
        [("user_id", 1), ("name", 1)],
        name="user_name_unique_idx",
        unique=True,
    )
    await knowledge_bases.create_index(
        [("user_id", 1), ("updated_at", -1)],
        name="user_updated_at_idx",
    )
    await links.create_index(
        [("user_id", 1), ("kb_id", 1), ("document_id", 1)],
        name="user_kb_document_unique_idx",
        unique=True,
    )
    await links.create_index(
        [("user_id", 1), ("kb_id", 1), ("added_at", -1)],
        name="user_kb_added_at_idx",
    )
