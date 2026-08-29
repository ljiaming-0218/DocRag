from stores.database import get_database


from datetime import datetime


async def insert_conversation(conversation: dict) -> None:
    database = get_database()
    conversation_collection = database["conversations"]

    await conversation_collection.insert_one(conversation)
    

async def find_conversation_by_id(conversation_id: str) -> dict | None:
    database = get_database()
    conversation_collection = database["conversations"]

    conversation = await conversation_collection.find_one({"_id":conversation_id})
    return conversation

async def find_conversations(
    user_id: str,
    document_id: str | None = None,
    limit: int = 50,
    kb_id: str | None = None,
) -> list[dict]:
    database = get_database()
    conversation_collection = database["conversations"]

    query = {"user_id": user_id}

    if document_id:
        query["document_id"] = document_id
    if kb_id:
        query["kb_id"] = kb_id

    cursor = conversation_collection.find(query)
    

    cursor = cursor.sort("updated_at", -1)
    cursor = cursor.limit(limit)

    conversations = []

    async for conversation in cursor:
        conversations.append(conversation)

    return conversations

async def create_conversation_indexes() -> None:
    database = get_database()
    conversation_collection = database["conversations"]

    await conversation_collection.create_index(
        [
            ("user_id", 1),
            ("document_id", 1),
            ("updated_at", -1),
        ],
        name="user_document_updated_at_idx",
    )
    await conversation_collection.create_index(
        [
            ("user_id", 1),
            ("kb_id", 1),
            ("updated_at", -1),
        ],
        name="user_kb_updated_at_idx",
    )
    

async def update_conversation_updated_at(conversation_id: str, updated_at: datetime,) -> None:
    database = get_database()
    conversation_collection = database["conversations"]

    await conversation_collection.update_one(
        {"_id": conversation_id},
        {"$set": {"updated_at": updated_at}},
    )
