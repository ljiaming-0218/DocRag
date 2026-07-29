from stores.database import get_database


async def insert_message(message: dict) -> None:
    # 获取数据库
    database = get_database()
    # 获取 messages 集合
    message_collection = database["messages"]
    # 异步插入消息
    await message_collection.insert_one(message)


async def find_messages_by_conversation(conversation_id: str) -> list[dict]:
    database = get_database()

    message_collection = database["messages"]

    cursor = message_collection.find({"conversation_id": conversation_id})
    cursor = cursor.sort("created_at", 1)

    messages = []

    async for message in cursor:
        messages.append(message)
    return messages

async def create_message_indexes() -> None:
    database = get_database()
    message_collection = database["messages"]

    await message_collection.create_index(
        [
            ("conversation_id", 1),
            ("created_at", 1),
        ],
        name="conversation_id_created_at_idx",
    )


async def find_recent_messages_by_conversation(conversation_id, limit) -> list[dict]:
    database = get_database()
    message_collection = database["messages"]

    cursor = message_collection.find({"conversation_id": conversation_id})
    cursor = cursor.sort("created_at", -1).limit(limit)

    messages = []

    async for message in cursor:
        messages.append(message)
    messages.reverse()
    return messages
